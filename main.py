from landmarks_data_to_csv import *
import os
import glob
import pandas as pd
import numpy as np
from numpy import argmax
import pickle
import time
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

curr_dir = os.getcwd()
dataset_dir = 'dataset'
data_dir = os.path.join(curr_dir, dataset_dir)

folders = glob.glob(data_dir+'\\*')
imagenames_list = []

#creating dataset CSV

#for folder in folders:
#    print(folder)
#    for f in glob.glob(folder+'/*.JPG'):
#        imagenames_list.append(f)

#for image in imagenames_list:
#    images_to_csv(image)

#adding headers to csv
#add_headers(curr_dir+"\\dataset.csv")

def preprocessing(custom_gesture):
    # preprocessing for further splitting of dataset
    if(custom_gesture):
        df_initial = pd.read_csv(curr_dir+"\\final_dataset.csv")
    else:
        df_initial = pd.read_csv(curr_dir+"\\dataset.csv")
    
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

    output_df = pd.DataFrame(df_initial['output'])
    return df_final,output_df

def oversampling(custom_gesture):
    df_final,output_df = preprocessing(custom_gesture)
    X_total = np.array(df_final)
    Y_total = output_df[['output']].to_numpy()
    #X_total_scaled = preprocessing.StandardScaler().fit(X_total)

    #splitting training and testing
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20, random_state=0)

    #print("x_train dataset: ", x_train.shape)
    #print("y_train dataset: ", y_train.shape)
    item_list = np.unique(y_train)

    print(item_list)
    print(len(item_list))
    sm = SMOTE(random_state=len(item_list))
    x_train_res, y_train_res = sm.fit_resample(x_train, y_train.ravel())

    return x_train_res, y_train_res

def undersampling(custom_gesture):
    df_final,output_df = preprocessing(custom_gesture)
    X_total = np.array(df_final)
    Y_total = output_df[['output']].to_numpy()
    #X_total_scaled = preprocessing.StandardScaler().fit(X_total)

    #splitting training and testing
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20, random_state=0)

    #print("x_train dataset: ", x_train.shape)
    #print("y_train dataset: ", y_train.shape)
    item_list = np.unique(y_train)

    print(item_list)
    print(len(item_list))
    
    tomek = TomekLinks()
    X_tomek, Y_tomek = tomek.fit_resample(x_train, y_train)
    return X_tomek, Y_tomek

#----------------------Logistic Regression----------------------#
def Logistic_Regression(custom_gesture=False):
    #Preprocessing
    df_final,output_df = preprocessing(custom_gesture)
    X_total = np.array(df_final)
    Y_total = output_df[['output']].to_numpy()
    #X_total_scaled = preprocessing.StandardScaler().fit(X_total)

    #splitting training and testing
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20, random_state=0)

    #if(custom_gesture):
        #applying oversampling to imbalance dataset
        #x_train, y_train = oversampling(custom_gesture)

    # all parameters not specified and are set to their defaults
    logisticRegr = LogisticRegression(max_iter=100000)

    start = time.time()
    #Training
    logisticRegr.fit(x_train, y_train.ravel())
    end = time.time()

    if(custom_gesture):
        #loading the trained model into pickle file
        filename = 'LR_final_model.sav'
        pickle.dump(logisticRegr, open(filename, 'wb'))

        #moving pickle file to custom model folder
        if not os.path.exists('custom_model'):
            os.makedirs('custom_model')
        dest_path = os.path.join(curr_dir, 'custom_model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    else:
        #loading the trained model into pickle file
        filename = 'LR_final_model.sav'
        pickle.dump(logisticRegr, open(filename, 'wb'))

        #moving pickle file to model folder
        if not os.path.exists('model'):
            os.makedirs('model')
        dest_path = os.path.join(curr_dir, 'model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    #loading saved model pickle file
    loaded_model = pickle.load(open(dest_path+'\\'+filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

    #confusion matrix
    predictions = logisticRegr.predict(x_test)
    label = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J',' K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U','V','W','X','Y','Z']
    print(len(label))
    cm = metrics.confusion_matrix(y_test, predictions, labels=label)
    print(cm)

    # Use score method to get accuracy of model
    score = logisticRegr.score(x_test, y_test)
    print("Logistic Regression Accuracy: ",score)
    print("Time taken for training: {} seconds" .format(end-start))

    

    #plotting confusion matrix using seaborn
    plt.figure(figsize=(30,30))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=label, yticklabels=label);
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Logistic Regression Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    #plt.show()

#-----------------------KNN-------------------------------------#
def KNN(custom_gesture=False):
    #Preprocessing
    df_final,output_df = preprocessing(custom_gesture)
    X_total = np.array(df_final)
    Y_total = output_df[['output']].to_numpy()
    #X_total_scaled = preprocessing.StandardScaler().fit(X_total)

    #splitting training and testing
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20, random_state=0)

    #if(custom_gesture):
        #applying oversampling to imbalance dataset
        #x_train, y_train = oversampling(custom_gesture)

    start = time.time()
    #Training
    classifier = KNeighborsClassifier(n_neighbors=28)
    classifier.fit(x_train, y_train)
    end = time.time()
    if(custom_gesture):
        #loading the trained model into pickle file
        filename = 'KNN_final_model.sav'
        pickle.dump(classifier, open(filename, 'wb'))

        #moving pickle file to custom model folder
        if not os.path.exists('custom_model'):
            os.makedirs('custom_model')
        dest_path = os.path.join(curr_dir, 'custom_model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    else:
        #loading the trained model into pickle file
        filename = 'KNN_final_model.sav'
        pickle.dump(classifier, open(filename, 'wb'))

        #moving pickle file to model folder
        if not os.path.exists('model'):
            os.makedirs('model')
        dest_path = os.path.join(curr_dir, 'model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    #loading saved model pickle file
    loaded_model = pickle.load(open(dest_path+'\\'+filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

    #prediction
    y_pred = classifier.predict(x_test)

    
    #confusion matrix
    predictions = classifier.predict(x_test)
    label = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J',' K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U','V','W','X','Y','Z']
    print(len(label))
    cm = metrics.confusion_matrix(y_test, predictions, labels=label)
    print(cm)

    #accuracy score
    score=metrics.accuracy_score(y_pred,y_test)
    print("KNN Accuracy: ", score)
    print("Time taken for training: {} seconds" .format(end-start))


    plt.figure(figsize=(30,30))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=label, yticklabels=label)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'KNN Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    #plt.show()

#-----------------------SVM-------------------------------------#
def SVM(custom_gesture=False):
    #Preprocessing
    df_final,output_df = preprocessing(custom_gesture)
    X_total = np.array(df_final)
    Y_total = output_df[['output']].to_numpy()
    #X_total_scaled = preprocessing.StandardScaler().fit(X_total)

    #splitting training and testing
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20, random_state=0)

    #applying oversampling to imbalance dataset
    #if(custom_gesture):
    #    x_train, y_train = oversampling(custom_gesture)
        #x_train, y_train = undersampling(custom_gesture)


    start = time.time()
    #Training
    clf = svm.SVC(kernel='poly') # Linear Kernel
    clf.fit(x_train, y_train)
    end = time.time()
    if(custom_gesture):
        #loading the trained model into pickle file
        filename = 'SVM_final_model.sav'
        pickle.dump(clf, open(filename, 'wb'))

        #moving pickle file to custom model folder
        if not os.path.exists('custom_model'):
            os.makedirs('custom_model')
        dest_path = os.path.join(curr_dir, 'custom_model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    else:
        #loading the trained model into pickle file
        filename = 'SVM_final_model.sav'
        pickle.dump(clf, open(filename, 'wb'))


        #moving pickle file to model folder
        if not os.path.exists('model'):
            os.makedirs('model')
        dest_path = os.path.join(curr_dir, 'model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    #loading saved model pickle file
    loaded_model = pickle.load(open(dest_path+'\\'+filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

    #prediction
    y_pred = clf.predict(x_test)

    #confusion matrix
    #predictions = clf.predict(x_test)
    #label = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J',' K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U','V','W','X','Y','Z']
    #print(len(label))
    #cm = metrics.confusion_matrix(y_test, predictions, labels=label)
    #print(cm)

    #accuracy score
    score=metrics.accuracy_score(y_pred,y_test)
    print("SVM Accuracy: ",score)
    print("Time taken for training: {} seconds" .format(end-start))

    
    #PRF scores
    eval_metrics = metrics.classification_report(y_test, y_pred,output_dict=True)
    eval_metrics_df = pd.DataFrame(eval_metrics).transpose() 
    print(eval_metrics_df)

    #plt.figure(figsize=(30,30))
    #sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=label, yticklabels=label)
    #plt.ylabel('Actual label')
    #plt.xlabel('Predicted label')
    #all_sample_title = 'KNN Accuracy Score: {0}'.format(score)
    #plt.title(all_sample_title, size = 15)

#-----------------------NN-------------------------------------#
def NN(custom_gesture=False):
    #Preprocessing
    df_final,output_df = preprocessing(custom_gesture)
    X_total = np.array(df_final)
    Y_total = output_df[['output']].to_numpy()
    #X_total_scaled = preprocessing.StandardScaler().fit(X_total)

    
    #splitting training and testing
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20, random_state=0)

    #if(custom_gesture):
        #applying oversampling to imbalance dataset
        #x_train, y_train = oversampling(custom_gesture)


    #further preprocessing
    X_total=X_total.astype(np.float64)
    Y_total = LabelEncoder().fit_transform(Y_total)
    #print((Y_total))
    #print((X_total))
    n_features = x_train.shape[1]
    n_class = len(np.unique(Y_total)) 
    #print (n_features,n_class)
    
    x_train, x_test, y_train, y_test = train_test_split(X_total, Y_total, test_size=0.20)

    start = time.time()
    #Training
    model = Sequential()
    model.add(Dense(100, input_dim=n_features, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))
    model.summary()

    # compile the keras model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=2)
    
    end = time.time()
    if(custom_gesture):
         #saving the model to hdf5
        filename = "NN_final_model.h5"

        #moving pickle file to custom model folder
        if not os.path.exists('custom_model'):
            os.makedirs('custom_model')
        dest_path = os.path.join(curr_dir, 'custom_model')
        shutil.move(os.path.join(curr_dir,filename), os.path.join(dest_path, filename))

    else:
        #saving the model to hdf5
        filename = "NN_final_model.h5"

        if not os.path.exists('model'):
            os.makedirs('model')
        dest_path = os.path.join(curr_dir, 'model')
        model.save(os.path.join(dest_path,filename))


    #prediction
    
    loaded_model = tf.keras.models.load_model(curr_dir+"\\model\\NN_final_model.h5")
    loaded_model.build()

    y_pred_nn = loaded_model.predict(x_test)
    y_pred_nn = argmax(y_pred_nn, axis=-1).astype('int')

    #accuracy score
    score=metrics.accuracy_score(y_pred_nn, y_test)
    print('-----------NN Accuracy on test set-------------------')
    print(score)
    print('Time taken for training: ',end-start)
   


#Logistic_Regression()
#KNN()
#SVM()
#NN()