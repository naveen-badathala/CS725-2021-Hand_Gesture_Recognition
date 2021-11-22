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
from main import *
import shutil
import glob
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
        rotation_range = 40,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        brightness_range = (0.5, 1.5))
    

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
modelfile_custom_path = os.getcwd() + "\\custom_model\\SVM_final_model.sav"
NN_flag = False
asl_alphabet_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
dict_asl = dict(list(enumerate(asl_alphabet_list)))
custom_gesture_image_count = 0
multiple_gestures_count = 0
selected_retrain_model = ''

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
def predict(gesture, custom_gesture=False):
    if(custom_gesture):
        print("from custom file path")
        model_path = modelfile_custom_path
    else:
        model_path = modelfile_path

    currdir = os.getcwd()
    #increase brightness and saving
    gesture = increase_brightness(gesture, value=30)

    captured_img_path = currdir+'\\test_img.jpg'
    cv2.imwrite(captured_img_path, gesture)
    coordinates = convert_to_landmarks(captured_img_path)

    #loading saved model pickle file
    global NN_flag
    global dict_asl
    if(NN_flag):
        loaded_model = tf.keras.models.load_model(model_path)
        loaded_model.build()
    else:
        loaded_model = pickle.load(open(model_path, 'rb'))

    if coordinates.size != 0:
        if(NN_flag):
            coordinates=coordinates.astype(np.float64)
            #print(coordinates)
            result = loaded_model.predict(coordinates)
            result = argmax(result, axis=-1).astype('int')
            #print(result)
            result = dict_asl[result[0]]
        else:
            result = loaded_model.predict(coordinates)
            #print(result)
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

#save image for custom gesture
def saveimage(image):
    image_saved = False
    global custom_gesture_image_count
    #custom_gesture_image_count += 1

    currdir = os.getcwd()
    custom_gestures_dir = 'custom_gestures'
    currdir = os.path.join(currdir, custom_gestures_dir)

    #increase brightness and saving
    gesture = image
    #gesture = increase_brightness(image, value=30)
    
    captured_img_path = currdir+f'\\custom_gesture_img-{custom_gesture_image_count}.jpg'
    cv2.imwrite(captured_img_path, gesture)
    
    coordinates = convert_to_landmarks(captured_img_path)
    if(coordinates.size != 0):
        #print("entered")
        cv2.imwrite(captured_img_path, gesture)       
        custom_gesture_image_count += 1
        image_saved = True

    return image_saved
    
def duplicate_images():
    images_list = []
    curr_dir = os.getcwd()
    custom_dir = curr_dir + "\\custom_gestures"
    for f in glob.glob(custom_dir+'/*.JPG'):
        images_list.append(f)

    count = len(images_list)
    for i in range(10):
        for image in images_list: 
            shutil.copy2(image, custom_dir+'\\custom_gesture_img-' + str(count) + '.jpg')
            count += 1
        
def augment_images():
    images_list = []
    curr_dir = os.getcwd()
    custom_dir = curr_dir + "\\custom_gestures"
    for f in glob.glob(custom_dir+'/*.JPG'):
        images_list.append(f)

    for image in images_list:
        # Loading a sample image 
        img = load_img(image) 
        # Converting the input sample image to an array
        x = img_to_array(img)
        # Reshaping the input image
        x = x.reshape((1, ) + x.shape) 
   
        # Generating and saving 5 augmented samples 
        # using the above defined parameters. 
        i = 0
        for batch in datagen.flow(x, batch_size = 1,
                          save_to_dir ='custom_gestures', 
                          save_prefix ='image', save_format ='JPG'):
            i += 1
            if i > 5:
                break


def custom_model_training():
    if(selected_retrain_model=='Logistic Regression'):
        Logistic_Regression(True)
    elif(selected_retrain_model=='KNN'):
        KNN(True)
    elif(selected_retrain_model=='SVM'):
        SVM(True)
    elif(selected_retrain_model=='Neural Networks'):
        NN(True)
    else:
        SVM(True)

def custom_model_predict():
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
            if count_frames > 1 and pred_text != "$":
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
                pred_text = predict(crop_img, True)
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
            if count_frames > 1 and pred_text != "$":
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

def custom_function():
    vc = cv2.VideoCapture(0)
    rval, frame = vc.read()
    flag = False
    gesture_image_count = 0
         
    while True:
        if frame is not None: 
            frame = cv2.flip(frame, 1)
            frame = cv2.resize( frame, (600,600) )
        
            #left rectangle
            cv2.rectangle(frame, (0,0), (300,400), (0,255,0), 2)
            crop_img = frame[0:400, 0:300]

            #resize image to 200 X 200
            crop_img_new = cv2.resize(crop_img, (200,200))


            #defining initial layout for webcam and textdisplay
            blackboard = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(blackboard, "Press 'C' to add your Gesture", (30, 40), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 179, 255))
        

            cv2.putText(blackboard, "Press 'Q' to Quit.............", (30, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 179, 255))
        

            cv2.putText(blackboard, "Current Status:- ", (30, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (102, 179, 255))

            
            
            if flag == True:
                #old_text = pred_text
                if gesture_image_count < 10:
                    saved_flag = saveimage(crop_img_new)
                    
                    if(saved_flag):
                        gesture_image_count += 1
                        beepSound()
                    cv2.putText(blackboard, 'Capturing', (30, 160), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))              
                    
                else:
                    cv2.putText(blackboard, 'Done Capturing', (30, 160), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 128, 255))              
                    global custom_gesture_image_count
                    custom_gesture_image_count = 0


                
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
    global selected_retrain_model
    selected_retrain_model = clicked.get()
    currdir = os.getcwd()
    modeldir = currdir+'\\model'
    custom_model_dir = currdir+"\\custom_model"
    
    global modelfile_path
    global modelfile_custom_path
    global NN_flag
    if(model_type=='Logistic Regression'):
        modelfile_path = modeldir+'\\LR_final_model.sav'
        modelfile_custom_path = custom_model_dir+'\\LR_final_model.sav'
    if(model_type=='KNN'):
        modelfile_path = modeldir+'\\KNN_final_model.sav'
        modelfile_custom_path = custom_model_dir+'\\KNN_final_model.sav'
    if(model_type=='SVM'):
        modelfile_path = modeldir+'\\SVM_final_model.sav'       
        modelfile_custom_path = custom_model_dir+'\\SVM_final_model.sav'
    if(model_type=='Neural Networks'):
        modelfile_path = modeldir+'\\NN_final_model.h5'       
        modelfile_custom_path = custom_model_dir+'\\NN_final_model.h5'
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

#custom gestures action
def selectModel1():
    global multiple_gestures_count
    if(str_var.get() != ""):
        multiple_gestures_count += 1

        phrase=str_var.get()
        print(phrase)
        str_var.set("")
        custom_function()
        
        #duplicate_images()
        #augment_images()
        convert_to_landmarks_custom_gesture(phrase)

        if(multiple_gestures_count < 2):
            #print(multiple_gestures_count)
            #adding headers to new csv
            add_headers(os.getcwd()+"\\custom_dataset.csv")

        
        generate_custom_csv()


def selectModel2():
    custom_model_training()
    custom_model_predict()

#string to take custom gesture phrase
str_var= StringVar()

def submit():
    phrase=str_var.get()
    str_var.set("")

# creating a label for input phrase
name_label = Label(win, text = 'Enter phrase for Gesture: ', font=('helvetica 15 bold'))
# creating a entry for input
name_entry = Entry(win,textvariable = str_var, font=('helvetica 15 bold'))


#submit button
#sub_btn=Button(win,text = 'Submit', command = submit)
  
name_label.place(x=5, y= 100)
name_entry.place(x=280, y=100)
#sub_btn.place(x=5, y= 200)


#Add custom gestures button layout
but0=Button(frame,padx=10,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=selectModel1,text='Add Custom Gesture(s)',font=('helvetica 15 bold'))
but0.place(x=10,y=150)

#proceed button layout
but2=Button(frame,padx=10,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=selectModel2,text='Predict Gestures',font=('helvetica 15 bold'))
but2.place(x=10,y=250)


#proceed button layout
but2=Button(frame,padx=10,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=selectModel,text='Predict Default Gestures',font=('helvetica 15 bold'))
but2.place(x=10,y=300)

win.mainloop()
