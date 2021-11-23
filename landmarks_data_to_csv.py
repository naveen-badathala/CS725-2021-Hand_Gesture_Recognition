import cv2
import mediapipe
import csv
import os
from pathlib import Path
import shutil
import pandas as pd
delete_count = 0

def images_to_csv(imgpath,custom_gesture=False,custom_label=''):
    global delete_count
    drawingModule = mediapipe.solutions.drawing_utils
    handsModule = mediapipe.solutions.hands

    with handsModule.Hands(static_image_mode=True, max_num_hands=1) as hands:
        #imgpath="C:/Users/HP/Downloads/hand.jpg"
        image = cv2.imread(imgpath)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        #label = imgpath
        if(custom_gesture):
            label = custom_label
        else:
            label= Path(imgpath).parts[-2]
 
        imageHeight, imageWidth, _ = image.shape
        row=[]
        if(custom_gesture):
            #handling delete custom dataset csv for next iterations
            delete_count += 1
            if(delete_count==1):
                os.remove('custom_dataset.csv')

            csv_file_name = "custom_dataset.csv"
        else:
            csv_file_name = 'test.csv'
        with open(csv_file_name,'a',encoding='UTf-8',newline='') as f:
            writer=csv.writer(f)
            #writer.writerow(header)
            if results.multi_hand_landmarks != None:
                
                for handLandmarks in results.multi_hand_landmarks:
                    for point in handsModule.HandLandmark:

                        normalizedLandmark = handLandmarks.landmark[point]
                        pixelCoordinatesLandmark = drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, imageWidth, imageHeight)
                        l=str(normalizedLandmark).split('\n')
                        res=[]
                        for i in l:
                            if (i != ''):
                                res.append(i[3:])
                        #converting coordinate values
                        #for i in range(len(res)):
                        #    res[i] = float(res[i])
                        row.append(res)
                    row.append(label)
                    #row.append(imgpath)
                    writer.writerow(row)

def add_headers(filepath):
    header=[]
    
    for i in range(1,22):
        header.append("feature-"+str(i))

    header.append("output")

    with open(filepath,newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open(filepath,'w',newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)

