import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from detect import detect_box
from extract_cells import TableAnalysis

def check_every_photo(folder_dir, config):
    config = config
    incorrect = []
    png_file = 0
    for file in os.listdir(folder_dir):
        if(file.endswith('.png')):
            png_file += 1
            image_path = f'{folder_dir}/{file}'
            image = cv2.imread(image_path)
            image_width = image.shape[0]
            image_height = image.shape[1]
            stats, labels = detect_box(image, config)
            count_boxes = 0
            for stat in stats:
                for x, y, w, h, area in stat:
                    if image_width*config["PERCENT_BOX_MAX_WIDTH"] > w > image_width*config["PERCENT_BOX_MIN_WIDTH"] \
                    and image_height*config["PERCENT_BOX_MAX_HEIGHT"] > h > image_height*config["PERCENT_BOX_MIN_HEIGHT"]:
                        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0), 5)
                        count_boxes += 1
            print(f'File name: {file}')            
            print(f'Number of boxes: {count_boxes}')
            if count_boxes % 5 == 0:
                print('Correct')
            else:
                print('Incorrect')
                incorrect.append(file)
            print("\n=============\n")
    print(f"Incorrect photos: {incorrect}")
    mistake_percent = len(incorrect)*100/png_file
    print(f"Total Mistakes: {len(incorrect)}/{png_file}")
    print(f"Mistake percentage: {round(mistake_percent,2)}%")
    print(f"Incorrect photos: {incorrect}\nTotal Mistakes: {len(incorrect)}/{png_file}\nMistake percentage: {round(mistake_percent,2)}%", file=open("incorrect.txt", 'w'))
    
if __name__ == '__main__':
    config = TableAnalysis.config
    folder_dir = 'tables'
    check_every_photo(folder_dir, config)