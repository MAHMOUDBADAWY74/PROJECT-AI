#!/usr/bin/env python
# coding: utf-8

# In[38]:


import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# In[39]:


face_classifier = cv2.CascadeClassifier()
face_classifier = cv2.CascadeClassifier(
    r'C:\Users\Dell\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

# In[40]:


classifier = load_model(r'D:\project AI\project AI\finalprojectmodel.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# In[41]:


click = cv2.VideoCapture(0)

# In[42]:


while True:
    RT, FramE = click.read()
    LabeLs = 0
    colorGRAY = cv2.cvtColor(FramE, cv2.COLOR_BGR2HSV)
    FACE = face_classifier.detectMultiScale(colorGRAY, 1.3, 5)
    for (i, j, k, l) in FACE:
        cv2.rectangle(FramE, (i, j), (i + k, j + l), (255, 0, 0), 2)
        RO_colorGRAY = colorGRAY[j:j + l, i:i + k]
        RO_colorGRAY = cv2.resize(RO_colorGRAY, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([RO_colorGRAY]) != 0:
            RO = RO_colorGRAY.astype('float') / 255.0
            RO = img_to_array(RO)
            RO = np.expand_dims(RO, axis=0)
            X = np.reshape(RO, (1, -1))

            Prediic = classifier.predict(X)[0]
            LabeL = emotion_labels[Prediic.argmax()]
            positionLABEL = (i, j)
            cv2.putText(FramE, LabeL, positionLABEL, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(FramE, 'Where are you?', (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 3)
    cv2.imshow('Recognizing your Expression', FramE)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# In[ ]:
