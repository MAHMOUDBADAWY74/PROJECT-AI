#!/usr/bin/env python
# coding: utf-8

# In[11]:


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array

face_classifier = cv2.CascadeClassifier(
    r'C:\Users\Dell\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\project AI\project AI\finalprojectmodel.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def classify(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    faces = face_classifier.detectMultiScale(gray, 1.3, 7)
    c = 1
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        rof_gray = gray[y:y + h, x:x + w]
        rof_gray = cv2.resize(rof_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([rof_gray]) != 0:
            rof = rof_gray.astype('float') / 255.0
            rof = img_to_array(rof)
            rof = np.expand_dims(rof, axis=0)
            X = np.reshape(rof, (1, -1))

            preds = classifier.predict(X)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y + h + 8)
            facex = str(c)
            cv2.putText(frame, facex, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            c += 1
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def video_classify():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = classify(frame)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def image_classify(frame):
    frame = cv2.imread(frame)
    # frame = cv2.resize(frame, (32, 32))
    X = frame.reshape((-1, 32, 32, 1))
    X = X / 255.0
    print(X.shape)

    # frame = img_to_array(frame)
    # frame = np.expand_dims(frame, axis=0)

    preds = classifier.predict(X)[0]

    label = class_labels[preds.argmax()]
    return label

# In[ ]:
