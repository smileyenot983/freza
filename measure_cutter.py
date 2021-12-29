import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import *


class Ruletka():
    '''
    this class implements complete pipeline used for measuring cutter width and length
    
    '''
    def __init__(self) -> None:
        super().__init__()

    def measure(self,images,algo_params,reference_diameter):
    
        '''

        '''

        canny_lower_thresh = algo_params['canny_lower_thresh']
        canny_upper_thresh = algo_params['canny_upper_thresh']
        canny_l2 = algo_params['canny_l2']

        ref_thr = algo_params['ref_thr']
        ref_mll = algo_params['ref_mll']
        ref_mlg = algo_params['ref_mlg']

        cut_thr = algo_params['cut_thr']
        cut_mll = algo_params['cut_mll']
        cut_mlg = algo_params['cut_mlg']


        binary_images = []
        for image in images:
            im = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            (T,thresh) = cv2.threshold(im,0,255,cv2.THRESH_OTSU)
            binary_images.append(thresh)



        # creating a resulting image by overlapping all valid images
        res_img = 255*np.ones(binary_images[0].shape,dtype='uint8')

        for i in range(len(binary_images)):
            res_img = np.minimum(res_img,binary_images[i])
        
        cv2.imwrite('res_img.jpg',res_img)

        # detecting reference object and cutter by template matching
        reference_coords = template_matching(
            img_path='res_img.jpg',
            template_path='templates/ref_template.png',
            vis=False
        )

        cutter_coords = template_matching(
            img_path='res_img.jpg',
            template_path='templates/obj_template.png',
            vis=False
        )



        # edge detection
        canny_lower_thresh = 50
        canny_upper_thresh = 200
        canny_l2 = True

        img_edg = cv2.Canny(res_img,canny_lower_thresh,canny_upper_thresh,L2gradient = canny_l2,apertureSize = 3)

        ref_edg = img_edg[reference_coords[1]:reference_coords[3],
                        reference_coords[0]:reference_coords[2]]

        cut_edg = img_edg[cutter_coords[1]:cutter_coords[3],
                        cutter_coords[0]:cutter_coords[2]]

        REF_SHIFT = ref_edg.shape[0]//2

        upper_part = ref_edg[:REF_SHIFT,:]
        lower_part = ref_edg[REF_SHIFT:,:]

        left_up_line, right_up_line = hough_lines(upper_part,ref_thr,ref_mll,ref_mlg,only_vertical=True,only_edges=True)
        left_low_line, right_low_line = hough_lines(lower_part,ref_thr,ref_mll,ref_mlg,only_vertical=True,only_edges=True)

        upper_circle_diameter = abs(left_up_line[0]-right_up_line[0])
        lower_circle_diameter = abs(left_low_line[0]-right_low_line[0])
        # measured diameter with caliper = 50mm
        # print(f"Upper circle diameter(px) : {upper_circle_diameter}")

        # print(f"Lower circle diameter(px) : {lower_circle_diameter}")

        if upper_circle_diameter==lower_circle_diameter:
            pixel_to_mm = reference_diameter/upper_circle_diameter
        else:
            # print('The instrument has some angle(lower_circle_diameter != upper_circle_diameter)')
            pixel_to_mm = reference_diameter/upper_circle_diameter



        lines_cutter_all = hough_lines(cut_edg,20,5,cut_mlg)
        lines_cutter_vertical = hough_lines(cut_edg,cut_thr,cut_mll,cut_mlg,only_vertical=True)



        # calculating width of the cutter
        CUTTER_MID = cut_edg.shape[1]//2
        width_line, width = self.cutter_width(lines_cutter_vertical,CUTTER_MID)


        # calculating length of the cutter
        length_line,length = self.cutter_length(lines_cutter_all)


        cv2.line(cut_edg,(width_line[0],width_line[1]),(width_line[2],width_line[3]),(255,255,255),2)

        cv2.line(cut_edg,(length_line[0],length_line[1]),(length_line[2],length_line[3]),(255,255,255),2)

        width_real = round(width*pixel_to_mm,3)
        length_real = round(length*pixel_to_mm,3)

        length_log = f'Estimated length(px) : {length} | estimated length(mm) : {length_real}'
        width_log = f'Estimated width(px) : {width} | estimated width(mm) : {width_real}'

        print(length_log)
        print(width_log)


        # plt.imshow(cut_edg)
        # plt.text(cut_edg.shape[1]+10,0,length_log)
        # plt.text(cut_edg.shape[1]+10,10,width_log)
        # plt.title('cutter length and width lines')

        # plt.show()

        return length_real,width_real



    def cutter_length(self,lines):
        top_line = lines[0]
        bot_line = lines[0]

        top_y = np.max([top_line[1],top_line[3]])
        bot_y = np.min([bot_line[1],bot_line[3]])

        for line in lines:
            # line = [start_x,start_y,end_x,end_y]
            if line[1] < bot_y:
                bot_line = line
                bot_y = line[1]
            if line[3] < bot_y:
                bot_line = line
                bot_y = line[3]

            if line[1] > top_y:
                top_line = line
                top_y = line[1]

            if line[3] > top_y:
                top_line = line
                top_y = line[3]

        estimated_length = abs(top_y-bot_y)

        return (bot_line[0],bot_y,bot_line[0],top_y),estimated_length

    def cutter_width(self,lines,mid_line):
        '''
        finds 2 top lines: on left and right side and calculate distance between them

        inputs:
        lines - vertical lines
        mid_line - approximate x coordinate which divides cutter into left and right parts 
        '''

        def top_line(lines):
            top_line = lines[0]
            top = np.max([top_line[1],top_line[3]])

            for line in lines:
                if line[1] > top:
                    top = line[1]
                    top_line = line
                if line[3] > top:
                    top = line[3]
                    top_line  = line

            return top_line


        left_side_lines = []
        right_side_lines = []

        for line in lines:
            if line[0]<mid_line:
                left_side_lines.append(line)
            else:
                right_side_lines.append(line)

        left_top_line = top_line(left_side_lines)
        right_top_line = top_line(right_side_lines)

        estimated_width = abs(left_top_line[0]-right_top_line[0])
        
        return (left_top_line[0],left_top_line[1],right_top_line[0],left_top_line[1]),estimated_width