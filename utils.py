
import numpy as np
import matplotlib.pyplot as plt
import cv2


def centroid(img):
    '''
    calculates image centroid
    centroid = middle point of all white pixels
    '''

    coords = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] == 255:
                coords.append([i,j])

    coords_np = np.array(coords) # shape = [n_coords,2]
    return np.average(coords_np,axis=0)


def plot_multiple(images,labels=None):
    fig,ax = plt.subplots(1,len(images))

    for i in range(len(images)):
        ax[i].imshow(images[i])
        if labels is not None:
            ax[i].set_title(labels[i])
        
    plt.draw()

def template_matching(img_path='thresholded_img.jpg',template_path='template.jpg',vis=False):
    '''
    img_path - path to image where you will search for some pattern
    template_path - path to image with pattern
    both images should be thresholded
    '''

    img_gray = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(cv2.imread(template_path),cv2.COLOR_BGR2GRAY)
    # template matching(it might be applied here because template and image will have same scale, same angle of view)

    # print(f"img_gray.shape:{img_gray.shape}")
    # print(f"template_gray.shape:{template_gray.shape}")

    res = cv2.matchTemplate(img_gray,template_gray,cv2.TM_CCOEFF_NORMED)

    # get value(x,y) with highest match 
    (min_val,max_val,min_loc,max_loc) = cv2.minMaxLoc(res)

    (start_x,start_y) = max_loc
    end_x = start_x + template_gray.shape[1]
    end_y = start_y + template_gray.shape[0]

    # img_matching = np.copy(img_gray)
    cv2.rectangle(img_gray,(start_x,start_y),(end_x,end_y),(255,255,255),3)

    if vis:
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(img_gray[start_y:end_y,start_x:end_x])
        ax[0].set_title('zoomed template')
        ax[1].imshow(img_gray)
        ax[1].set_title('template on image')
        plt.draw()

    return start_x,start_y,end_x,end_y

def hough_lines(img,thr,min_line_length,max_line_gap,only_vertical=False,only_horizontal=False,only_edges=False):
    '''
    the algorithm is probabilistic hough transform(it takes only subset of points in order to compute faster)


    Searching vertical lines with Hough transform:
    1. detect all lines
    2. if only_horizontal ->leave only horizontal lines, only_vertical -> leave only vertical lines
    3. only_edges -> leave only leftmost and right most if only_vertical, topmost and bottommost if only_horizontal

    inputs:
    thr - threshold(min number for hough transform accumulator) 
    min_line_length - minimum length(in px) of line to be detected as a line
    max_line_gap - maximum gap(distance) between consecutive pixels to be detected as a line

    '''
    linesP = cv2.HoughLinesP(img,1,np.pi/180,thr,None,min_line_length,max_line_gap)

    # print(f"linesP : {linesP}")
    # print(linesP[0])
    # print(linesP[0][0])
    # draw_lines(img,linesP)

    # linesP_reshaped = []
    # for l in linesP:
    #     line = l[0]
    #     linesP_reshaped.append(line)

    # draw_lines(img,linesP_reshaped)


    

    if linesP is not None:
        # print(f"detected {len(linesP)} lines")
        
        if only_vertical:
            lines_vertical = []

            # line = [[start_x,start_y,end_x,end_y]]
            for line in linesP:
                line = line[0]

                if line[0]==line[2]:
                    lines_vertical.append(line)
            
            if only_edges:
                left_idx = 0
                right_idx = 0

                for i in range(len(lines_vertical)):
                    if lines_vertical[i][0] < lines_vertical[left_idx][0]:
                        left_idx = i
                    elif lines_vertical[i][0] > lines_vertical[right_idx][0]:
                        right_idx = i

                print(f"left_idx: {left_idx}")
                print(f"right_idx: {right_idx}")

                print(f"left_idx : {left_idx}, | right_idx : {right_idx}")

                return lines_vertical[left_idx],lines_vertical[right_idx]
            
            
            return lines_vertical

        elif only_horizontal:
            lines_horizontal = []

            # print(f"len(linesP): {len(linesP)}")

            # print(f"linesP[0]: {linesP[0]}")
            # print(f"linesP[0][0]: {linesP[0][0]}")
            # print(f"linesP[0][0][0]: {linesP[0][0][0]}")

            for line in linesP:
                line = line[0]
                # print(f"line : {line}")

                if line[1]==line[3]:
                    lines_horizontal.append(line)

            if only_edges:
                top_idx = 0
                bot_idx = 0

                for i in range(len(lines_horizontal)):
                    if lines_horizontal[i][1] < lines_horizontal[bot_idx][1]:
                        bot_idx = i
                    elif lines_horizontal[i][1] > lines_horizontal[top_idx][1]:
                        top_idx = i

                return lines_horizontal[bot_idx], lines_horizontal[bot_idx]

            return lines_horizontal

        else:
            return [line[0] for line in linesP]
    else:
        return [[0,0,1,1],[0,0,1,1]]

def left_right_coords(lines):
    left_coord = lines[0][0]
    right_coord = lines[0][0]

    # find left and right x coords
    for line in lines:
        if line[0] < left_coord:
            left_coord = line[0]
        if line[2] < left_coord:
            left_coord = line[2]

        if line[0] > right_coord:
            right_coord = line[0]
        if line[2] > right_coord:
            right_coord = line[2]

    return left_coord, right_coord

def draw_lines(img,lines, color=(0,0,255)):
    for line in lines:
        cv2.line(img,(line[0],line[1]),(line[2],line[3]),color,4)

    plt.imshow(img)
    plt.show()


    # plt.draw()

def draw_box(img, box_coords):
    # box coords:
    # top-left coord :     (x,y)
    # right-bottom coord:  (x,y)

    x_1,y_1 = box_coords[0], box_coords[1]
    x_2,y_2 = box_coords[2], box_coords[3]


    cv2.line(img, (x_1,y_1), (x_2,y_1), (0,255,0), 2)
    cv2.line(img, (x_2,y_1), (x_2,y_2), (0,255,0), 2)
    cv2.line(img, (x_2,y_2), (x_1,y_2), (0,255,0), 2)
    cv2.line(img, (x_1,y_2), (x_1,y_1), (0,255,0), 2)