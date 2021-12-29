import cv2
from matplotlib import image 
import numpy as np
import albumentations as A
import tensorflow as tf
from pypylon import pylon
import argparse
import os

from measure_cutter import Ruletka




def efnet_lstm(backbone='0',seq_length=16,input_size=224):
    if backbone == '0':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB0
    elif backbone == '1':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB1
    elif backbone == '2':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB2
    elif backbone == '3':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB3
    elif backbone == '4':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB4
    elif backbone == '5':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB5
    elif backbone == '6':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB6
    elif backbone == '7':
        EFNet = tf.keras.applications.efficientnet.EfficientNetB7
        
    bottleneck = EFNet(weights='imagenet', include_top=False, pooling='avg')
    inp = tf.keras.layers.Input((seq_length, input_size, input_size, 3))
    x = tf.keras.layers.TimeDistributed(bottleneck)(inp)
    x = tf.keras.layers.LSTM(128)(x)
    x = tf.keras.layers.Dense(64, activation='elu')(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inp,x)

    return model




class Recorder:
    def __init__(self,seq_length,input_size):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.num_images_to_grab = seq_length
        self.input_size = input_size

        self.camera.Open()
        self.camera.ExposureAuto.SetValue('Once')


    def rec(self):
        '''
        returns list with images 
        '''
        self.camera.StartGrabbingMax(self.num_images_to_grab)

        sequence = []
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(5000,pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():

                img = grab_result.Array
                # print(img.shape)
                # img = A.Resize(height=self.input_size, width=self.input_size)(image = img)['image']
                sequence.append(np.dstack((img,img,img)))

        self.camera.Close()

        return sequence


if __name__=='__main__':
    # _________________BIENIE PART___________________

    # real_time = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--real-time',action='store_true')

    args = parser.parse_args()

    path = 'images'
    # length of sequence of images(as used during training step) 
    seq_length = 16
    # image size
    input_size = 224

    if args.real_time:
        print(f"args.real_time : {args.real_time}")
        recorder = Recorder(seq_length=seq_length,input_size=input_size)

        # record images
        images = recorder.rec()
    else:
        im_paths = [os.path.join(path,pth) for pth in os.listdir(path)]
        images = []

        for pth in im_paths[:seq_length]:
            img = cv2.cvtColor(cv2.imread(pth),cv2.COLOR_BGR2RGB)
            images.append(img)



    backbone = '0'
    classes = {0:'default', 1:'bienie'}

    DATASET_PATH = ''
    WEIGHTS_PATH = ''

    model = efnet_lstm(backbone=backbone, seq_length=seq_length, input_size=input_size)
    # model.load_weights(f"{WEIGHTS_PATH}/best.hdf5")

    

    inference_images = []
    for i in range(len(images)):
        img = A.Resize(height=input_size, width=input_size)(image = images[i])['image']
        inference_images.append(img)

    # predictions
    pred = model.predict(np.expand_dims(inference_images,axis=0))

    if pred>0.5:
        print(f"There is throb with probability : {pred}")
    else:
        print(f"There is no throb with probability: {1-pred}")

    # ___________________MEASUREMENT PART_________________

    ruletka = Ruletka()

    algo_params = {
    #edge detection params
    'canny_lower_thresh' : 121,
    'canny_upper_thresh' : 226,
    'canny_l2' : True,
    # hough lines params(ref object and cutter)
    'ref_thr' : 11,
    'ref_mll' : 10,
    'ref_mlg' : 2,

    'cut_thr' : 23,
    'cut_mll' : 3,
    'cut_mlg' : 0,
    
    }

    REFERENCE_DIAMETER = 50
    length,width = ruletka.measure(images,algo_params,REFERENCE_DIAMETER)



    print(f"length(mm) : {length} | width(mm) : {width}")