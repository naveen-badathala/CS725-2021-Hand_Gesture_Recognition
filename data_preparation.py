import cv2
import mediapipe
import numpy
import os
from pathlib import Path
import shutil
import glob
from landmarks_data_to_csv import *

parent_dir = os.getcwd()
data_dir = parent_dir+'\\dataset'
folders = glob.glob(data_dir+'\\*')


#moving images to another folder based on landmarks field drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

header=[]
#creating directory with name "dataset_used" inside dataset folder
directory = 'dataset_unused'
new_path = os.path.join(parent_dir, directory)
os.mkdir(new_path)


for folder in folders:
    #print("entered")
     directory1 = Path(folder).parts[-1]+'\\'
     #print(directory1)
     new_path1 = os.path.join(new_path, directory1)
     #print(new_path1)
     os.mkdir(new_path1)
           
     for image in glob.glob(folder+'/*.JPG'):
      
        with handsModule.Hands(static_image_mode=True) as hands:
            img = cv2.imread(image)
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks == None:
            
            shutil.move(image, new_path1)



#read_images = []        
#for image in imagenames_list:
#    images_to_csv(image)
