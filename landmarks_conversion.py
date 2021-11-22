from tkinter import image_names
from landmarks_data_to_csv import *
import os
import pandas as pd
import numpy as np
import glob
import pandas as pd

def convert_to_landmarks(image):
    curr_dir = os.getcwd()
    dataset_dir = 'dataset'
    data_dir = os.path.join(curr_dir, dataset_dir)


    #print(image)

    images_to_csv(image)


    #adding headers to csv
    add_headers(curr_dir+"\\test.csv")

    #
    df_initial = pd.read_csv(curr_dir+'\\test.csv')
    # preprocessing for further splitting
    for i in range(df_initial.shape[1]-1):
        j = i+1
        df_initial[f"feature-{j}"] = df_initial[f"feature-{j}"].apply(eval)

    #removing z-coordinate and splitting x and y coordinates
    df_final = pd.DataFrame()
    for i in range(df_initial.shape[1]-1):
        j = i+1
        split_df = pd.DataFrame(df_initial[f"feature-{j}"].tolist(), columns = [f'feature-{j}-x',f'feature-{j}-y',f'feature-{j}-z'])
        split_df.drop(split_df.columns[len(split_df.columns)-1], axis=1, inplace=True)
    #print(split_df)
        df_final= pd.concat([df_final, split_df], axis=1)

    np_final = np.array(df_final)
    #removing created temp test csv file
    os.remove("test.csv")

    return np_final


def convert_to_landmarks_custom_gesture(custom_label):
    header_flag = False
    imagenames_list = []
    curr_dir = os.getcwd()
    custom_dir = 'custom_gestures'
    data_dir = os.path.join(curr_dir, custom_dir)


    #for folder in folders:
        #print(folder)
    for f in glob.glob(data_dir+'/*.JPG'):
        imagenames_list.append(f)

    for image in imagenames_list:
        images_to_csv(image, True, custom_label)


    #if(os.path.exists('custom_dataset.csv')):
    #adding headers to csv
        #add_headers(curr_dir+"\\custom_dataset.csv")

def generate_custom_csv():
    
    curr_dir = os.getcwd()
    #
    df1 = pd.read_csv(curr_dir+"\\dataset.csv")
    df2 = pd.read_csv(curr_dir+"\\custom_dataset.csv")

    frames = [df1, df2]
    #merge
    df = pd.concat(frames)

    #print("starting")
    # Save to a new csv 
    if(os.path.exists('final_dataset.csv')):
        os.remove('final_dataset.csv')

    df.to_csv(curr_dir+"\\final_dataset.csv", index=False)
    #print("ending")

    #deleting custom_dataset.csv
    os.remove("custom_dataset.csv")