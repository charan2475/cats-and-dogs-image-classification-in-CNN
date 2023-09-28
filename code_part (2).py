#!/usr/bin/env python
# coding: utf-8

# In[3]:


import warnings
warnings.filterwarnings("ignore")
import os
import shutil #support file copying and removal.
import glob #The glob () function returns an array of filenames or directories matching a specified pattern.


# In[4]:


TRAIN_DIR = ".\DATASET"
ORG_DIR = "\train"
CLASS = ['cat', 'dog']


# In[5]:


for C in CLASS:
    DEST = os.path.join(TRAIN_DIR, C)
    if not os.path.exists(DEST):
        os.makedirs(DEST)
    for img_path in glob.glob(os.path.join(ORG_DIR, C)+"*"):
        SRC = img_path
        shutil.copy(SRC, DEST)


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator, load_img
import keras
from tensorflow.keras.utils import img_to_array


# In[7]:


base_model = InceptionV3(input_shape=(256,256,3),include_top=False)


# In[7]:


for layer in base_model.layers:
    layer.trainable = False


# In[8]:


X = Flatten()(base_model.output)
X = Dense(units=2, activation='sigmoid')(X)
model = Model(base_model.input, X)
model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()


# In[9]:


train_datagen = ImageDataGenerator(featurewise_center=True, rotation_range=0.4, width_shift_range=0.3, horizontal_flip=True, preprocessing_function=preprocess_input, zoom_range=0.4, shear_range=0.4)
train_data = train_datagen.flow_from_directory(directory="DATASET",target_size=(256,256), batch_size=64)


# In[10]:


train_data.class_indices


# In[11]:


t_img, label = train_data.next()


# In[12]:


def plotImages(img_arr, label):
    for idx, img in enumerate(img_arr):
        if idx <= 10:
            plt.figure(figsize=(5,5))
            plt.imshow(img)
            plt.title(img.shape)
            plt.axis = False
            plt.show()


# In[13]:


plotImages(t_img, label)


# In[14]:


from keras.callbacks import ModelCheckpoint , EarlyStopping
mc = ModelCheckpoint(filepath = "./best_model.h5",monitor = "accuracy", verbose = 1, save_best_only = True)
es = EarlyStopping(monitor="accuracy", min_delta=0.01, patience=5, verbose = 1)
cb = [mc,es]


# In[15]:


his = model.fit_generator(train_data, steps_per_epoch=10, epochs=30, callbacks=cb)


# In[17]:


from keras.models import load_model
model = load_model("best_model.h5")


# In[18]:


h = his.history
h.keys()


# In[19]:


plt.plot(h["loss"], '--go')
plt.title("Loss")
plt.show()
plt.plot(h["accuracy"],'--go')

plt.title("Accuracy")
plt.show()


# In[31]:


path = "test1/test1/156.jpg"
img = load_img(path, target_size=(256,256))
i = img_to_array(img)
i = preprocess_input(i)
input_arr = np.array([i])
input_arr.shape
pred = np.argmax(model.predict(input_arr))

if pred==0:
    print("The image is of cat")
else :
    print("The image is of dog")
    
plt.imshow(input_arr[0])
plt.title("input image")
plt.show


# In[ ]:




