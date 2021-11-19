from tkinter import *

from scipy.sparse import coo
import cv2
import os
import numpy as np
import pickle
from landmarks_conversion import *
from beep_sound import *
import textwrap
import tensorflow as tf
from numpy import argmax

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window and other configuration
win.geometry("600x600")
frame = Frame(win, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
win.title('Hand Gesture Recognition')
frame.config(background='light green')
label = Label(frame, text="Hand Gesture Recognition",bg='light green',font=('Times 35 bold'))
label.pack(side=TOP)

#Global variables by default using svm path
modelfile_path = os.getcwd() + "\\model\\SVM_final_model.sav"
NN_flag = False

#to increase brightness of captured frame
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

#predicting using various ML/DL models for the captured frame/gesture
def predict(gesture):
    currdir = os.getcwd()
    #increase brightness and saving
    gesture = increase_brightness(gesture, value=30)

    captured_img_path = currdir+'\\test_img.jpg'
    cv2.imwrite(captured_img_path, gesture)
    coordinates = convert_to_landmarks(captured_img_path)

    #loading saved model pickle file
    global NN_flag
    if(NN_flag):
        loaded_model = tf.keras.models.load_model(modelfile_path)
        loaded_model.build()
    else:
        loaded_model = pickle.load(open(modelfile_path, 'rb'))

    if coordinates.size != 0:
        result = loaded_model.predict(coordinates)
        if(NN_flag):
            coordinates=coordinates.astype(np.float64)
            result = loaded_model.predict(coordinates)
            result = argmax(result, axis=-1).astype('int')
    else:
        print("===========================---No Coordinates--================================----")
        result = ['$']

    #img = cv2.resize(gesture, (50,50))
    #img = img.reshape(1,50,50,1)
    #img = img/255.0
    #prd = model.predict(img)
    #index = prd.argmax()
    #return gestures[index]
    return result[0]

def main_function():
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    old_text = ''
    pred_text = ''
    count_frames = 0
    total_str = ''
    flag = False

    while True:
        delete_action = False
        space_action = False
        if frame is not None: 
            frame = cv2.flip(frame, 1)
            frame = cv2.resize( frame, (600,600) )
        
            #right rectangle
            #cv2.rectangle(frame, (300,0), (600,400), (0,255,0), 2)
            #crop_img = frame[0:400, 300:600]

            #left rectangle
            cv2.rectangle(frame, (0,0), (300,400), (0,255,0), 2)
            crop_img = frame[0:400, 0:300]


            #crop_img_new = cv2.resize(crop_img, (200,200))
            #crop_img_new = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGBA)
        
            #defining initial layout for webcam and textdisplay
            blackboard = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(blackboard, "Predicted character - ", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 179, 255))
        

            cv2.putText(blackboard, "Predicted word - ", (30, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 179, 255))
        

            cv2.putText(blackboard, "Predicted sentence - ", (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 179, 255))

            #checking whether the same gesture was predicted 3 times atleast
            if count_frames > 2 and pred_text != "$":
                beepSound()
                total_str += pred_text       
                count_frames = 0
            
            #handling space and delete gestures
            if pred_text == " ":
                space_action = True
            if pred_text == "#":
                delete_action = True
                count_frames = 0

            
            if flag == True:
                old_text = pred_text
                pred_text = predict(crop_img)
                #pred_text = predict(cv2.flip(crop_img,1))
            
                #handling space and delete predictions
                if(pred_text == 'space'):
                    pred_text = " "
                    #space_action = True
                if(pred_text == 'del'):
                    pred_text = "#"
                    total_str = total_str[:-1]
                    #delete_action = True
                print(pred_text)
        
                if old_text == pred_text:
                    count_frames += 1
                else:
                    count_frames = 0


                #printing current character
                if(len(total_str)>0 and not delete_action and not space_action and pred_text != '$'):
                    cv2.putText(blackboard, total_str[-1], (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))
                
                #printing current word
                if(len(total_str)>0 and not delete_action and not space_action and pred_text != '$'):
                    lis = list(total_str.split(" "))
                    cv2.putText(blackboard, lis[len(lis)-1], (30, 160), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))

                #printing Delete action
                if(delete_action):
                    print("entered")
                    cv2.putText(blackboard, 'DELETE', (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))
                
                #printing Space action
                if(space_action):
                    cv2.putText(blackboard, 'SPACE', (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))
                
                
                #printing and text wrapping to avoid overflow for final sentence
                wrapped_text = textwrap.wrap(total_str, width=25)
                print(wrapped_text)
                index = 0
                for phrase in wrapped_text:
                    cv2.putText(blackboard, phrase, (30, 240+index), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))
                    index += 40

            #layout for frame and blackboard
            res = np.hstack((frame, blackboard))
        
            #window to display capturing webcam
            cv2.imshow("Hand Gesture Recognition and text Extraction", res)
        
        #basic configuration -- Clicking 'c' will start capturing frames and predicting and 'q' to quit the program
        rval, frame = vc.read()
        keypress = cv2.waitKey(1)
        if keypress == ord('c'):
            flag = True
        if keypress == ord('q'):
            break

    #closing all the cv2 frames after program quits
    vc.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    vc.release()

#Set the model file based on selection event
def selected(event):
    model_type = clicked.get() #selecting corresponding model file
    currdir = os.getcwd()
    modeldir = currdir+'\\model'

    
    global modelfile_path
    global NN_flag
    if(model_type=='Logistic Regression'):
        modelfile_path = modeldir+'\\LR_final_model.sav'
    if(model_type=='KNN'):
        modelfile_path = modeldir+'\\KNN_final_model.sav'
    if(model_type=='SVM'):
        modelfile_path = modeldir+'\\SVM_final_model.sav'
    if(model_type=='Neural Networks'):
        modelfile_path = modeldir+'\\NN_final_model.h5'
        NN_flag = True
    

#defining dropdown menu options
options = [
        "SVM",
        "Logistic Regression",
        "KNN",
        "Neural Networks",      
]

clicked = StringVar()
clicked.set(options[0])

drop = OptionMenu(win, clicked, *options, command=selected)
drop.pack(pady=20)

#proceed button action
def selectModel():

    main_function()

#proceed button layout
but1=Button(frame,padx=10,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=selectModel,text='Proceed',font=('helvetica 15 bold'))
but1.place(x=10,y=200)

win.mainloop()