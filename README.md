# CS725-2021-Hand_Gesture_Recognition
**Problem Statement**

1. Hand Gesture Recognition is very important for human-computer interaction in various applications of AR/VR, Robotics etc.
2. A large portion of previous research requires specialized hardware like depth sensors and are not light-weight enough to run on real-time.
3. We developed a real time application to recognize the hand gesture and convert it into text interactively using Mediapipe library.
4. Our application is able to recognize new hand gestures along with asl alphabet letters.

**Installation**

Clone the repo and and run the following commands inside the folder to open the application
```
$ pip install -r requirements.txt
```

```
$ python final_demo.py
```
**How to operate**

Predicting default ASL gestures

1. Select an ml model from the dropdown list and then click on "Predict Default Gestures" button.
2. Once the webcam feed appears, place your hand in the green bordered box in the top left part.
3. Press 'c' to start predicting the gestures. You can see the results on the right side of the application.
4. User can also press 's' to convert the predicted text to speech.
4. Press 'q' to quit the application.

Predicting custom gestures along with default ASL gestures

1. Enter the phrase corresponding to the custom gesture and click on "Add Custom Gestures(s)" button.
2. Once the webcam feed appears, place your hand in the green bordered box in the top left part, and press 'c' to start capturing the custom gesture.
3. It take a few seconds for the application to capture the gesture. Once done press 'q' to quit.
4. Now press "Press Gestures" button. This will retrain the ml model along with the custom gestures and takes some time to complete.
5. Once the model is trained, a popup of webcam feed appears and you can start predicting the custom getures along with the asl alphabet as described above.

**DEMO**

Click this link to view the demo video https://drive.google.com/file/d/1iqJGpvdR-TV2sI9IWbKrx9mZS9vaur4q/view

**Report**

Project report and ppt can be found in reports folder.

**Team members:**
- Naveen Badathala 213050052
- Abisek R K 21Q050004
- Tejpal Singh 21Q050008
- Mohan Rajasekhar Ajjampudi 213050060
