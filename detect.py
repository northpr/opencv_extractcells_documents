import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def detect_box(image, config):
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask_2 = mask.copy()
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_scale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Detect the grid boarder
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100000:
            cv2.drawContours(mask, [c], -1, (255,255,255), -1)
            cv2.drawContours(mask_2, [c], -1, (255,255,255), 10)
            
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_BGR2GRAY)
    filled_mask = mask_2.copy()
    mask = cv2.bitwise_and(mask, thresh)


    # Find horizontal line
    kernal_h = cv2.getStructuringElement(cv2.MORPH_RECT, (55,1))
    detect_horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_h, iterations=1)
    detect_horizontal = cv2.dilate(detect_horizontal, kernal_h, iterations = 5)
    lines = cv2.HoughLinesP(detect_horizontal, config['HOUGHLINES']['RHO'], config['HOUGHLINES']['THETA'], 
                            config['HOUGHLINES']['THRESHOLD'], np.array([]),
                        config['HOUGHLINES']['HORIZONTAL_MIN_LINE_LENGTH'], config['HOUGHLINES']['MAX_LINE_GAP'])

    # Draw horizontal lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(filled_mask,(x1,y1),(x2,y2),(128,128,128),3)

    # Find vertical lines
    kernal_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,55))
    detect_vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal_v, iterations=2)
    detect_vertical = cv2.dilate(detect_vertical, kernal_v, iterations = 10)
    lines = cv2.HoughLinesP(detect_vertical, config['HOUGHLINES']['RHO'], config['HOUGHLINES']['THETA'], 
                            config['HOUGHLINES']['THRESHOLD'], np.array([]),
                        config['HOUGHLINES']['VERTICAL_MIN_LINE_LENGTH'], config['HOUGHLINES']['MAX_LINE_GAP'])
    
    # Draw vertical lines
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(filled_mask,(x1,y1),(x2,y2),(128,128,128),3)
            
    img_bin_final = cv2.bitwise_and(mask, filled_mask)
    img_bin_final = cv2.threshold(img_bin_final, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    final_kernel=np.ones((3,3), np.uint8)
    img_bin_final=cv2.dilate(~img_bin_final,final_kernel,iterations=5)

    ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)
    for x,y,w,h,area in stats:
            image_temp = image.copy()
            cv2.rectangle(image_temp,(x,y),(x+w,y+h),(255,0,0), 5)

    stats = stats[2:]
    stats = sorted(stats.tolist(), key=lambda b:b[1], reverse=False)

    #Creating a list of heights for all detected boxes
    heights = [stats[i][3] for i in range(len(stats))]
    mean = np.mean(heights)

    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    #Sorting the boxes to their respective row and column
    box = stats
    for i in range(len(box)):
        if(i==0):
            column.append(box[i])
            previous=box[i]
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]
                if(i==len(box)-1):
                    column.sort(key= lambda row: (row[0]))
                    row.append(column)
            else:
                column.sort(key= lambda row: (row[0]))
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])

    return row,labels


if __name__ == '__main__':
    IMAGE_DIRECTORY = 'tables/0.png'
    image = f'{IMAGE_DIRECTORY}'