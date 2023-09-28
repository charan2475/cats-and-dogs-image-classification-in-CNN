# cats-and-dogs-image-classification-in-CNN

Description of the

Imports: The code starts by importing the necessary Python libraries, including warnings, os, shutil, glob, numpy, matplotlib, pandas, TensorFlow, and Keras.

Data Preparation:
It sets the TRAIN_DIR variable to a directory named "DATASET" and specifies the original dataset directory as ORG_DIR.
Defines a list of classes CLASS, which contains two class labels: 'cat' and 'dog'.
It then iterates through each class and copies image files from the original directory (ORG_DIR) to a new directory (DEST) within the "DATASET" directory. This process organizes the dataset into subdirectories, separating images by class.

Model Creation:
The code initializes the InceptionV3 base model with a specified input shape (256x256x3) and sets include_top to False to exclude the top classification layers.
It then freezes (sets trainable to False) all layers in the base model.

Model Architecture:
The code flattens the output of the base model and adds a dense output layer with two units (binary classification) and a sigmoid activation function.
The model is created by specifying the input and output layers and is compiled using the Adam optimizer and binary cross-entropy loss function.

Data Augmentation:
The code sets up an image data generator (train_datagen) for data augmentation. It applies various transformations to the training images, such as rotation, width shift, horizontal flip, zoom, and shear. It also applies preprocessing using the preprocess_input function from the InceptionV3 model.
Training data is loaded using the data generator, specifying the target size (256x256) and batch size (64).

Training:
The model is trained using the fit_generator method with 10 steps per epoch and 30 epochs. Callbacks are set up, including model checkpointing and early stopping based on accuracy.

Model Evaluation:
After training, the best model is loaded using load_model from Keras, and its training history is stored in the h variable.
The code then plots the training loss and accuracy over epochs using Matplotlib.

Image Prediction:
The code demonstrates how to make predictions on a test image (path) by loading the image, preprocessing it, and passing it through the model.
The predicted class label is determined using argmax.
Finally, it displays the input image along with the predicted class label ('cat' or 'dog').
Please note that there are some issues in the code, such as missing imports (e.g., import matplotlib.pyplot as plt is missing), and there are unnecessary comments (e.g., # In[3]). Additionally, the code saves the best model as "best_model.h5" but doesn't specify the location where the file will be saved.
