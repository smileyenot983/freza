from measure_cutter import Ruletka

import cv2
from matplotlib import image 
import numpy as np

import argparse
import os



if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--real-time',action='store_true')

    args = parser.parse_args()

    path = 'images'




    ruletka = Ruletka()

    algo_params = {
    #edge detection params
    'canny_lower_thresh' : 100,
    'canny_upper_thresh' : 200,
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

    im_paths = [os.path.join(path,pth) for pth in os.listdir(path)]
    images = []

    for pth in im_paths:
        img = cv2.cvtColor(cv2.imread(pth),cv2.COLOR_BGR2RGB)
        images.append(img)

    length,width = ruletka.measure_2(images,algo_params,REFERENCE_DIAMETER)



    print(f"length(mm) : {length} | width(mm) : {width}")