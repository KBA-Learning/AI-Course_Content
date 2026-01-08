# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# The modules necessary to perform the analysis are included in the notebook

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x.shape, test_x.shape
# Here we count the number of (how many from each of the classes) unique train labels. 

unique, counts = np.unique(train_y, return_counts=True)
print("For train dataset - labels: \n", dict(zip(unique, counts)))
# Here we count the number of (how many from each of the classes) unique test labels. 

unique, counts = np.unique(test_y, return_counts=True)
print("For test dataset - labels: \n", dict(zip(unique, counts)))
img_array = train_x[7]
x = plt.imshow(img_array, interpolation = "antialiased", alpha = 1, cmap = plt.cm.binary)
plt.colorbar(x)
plt.show()

idxs = np.random.randint(0, train_x.shape[0], size = 49)
images = train_x[idxs]
labels = train_y[idxs]

plt.figure(figsize = (7, 7))
for i in range(len(idxs)):
    plt.subplot(7, 7, i + 1)
    image = images[i]
    plt.imshow(image, cmap='Dark2')
    plt.axis('off')
    
plt.show()
train_x = train_x.reshape((60000, 28, 28, 1))
train_x = train_x.astype("float32") / 255

test_x = test_x.reshape((10000, 28, 28, 1))
test_x = test_x.astype("float32") / 255
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
# we look at encoded train labels
print("Encoded train labels\n", train_y[0:5])
print("\nEncoded test labels\n", test_y[0:5])
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation = "relu", input_shape = (28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation = "relu"))

model.add(layers.Flatten())

model.add(layers.Dense(128, kernel_regularizer = regularizers.l2(0.00001),
                       activation = "relu"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10, kernel_regularizer = regularizers.l2(0.00001),
                       activation = "softmax"))
epoch = 10
batch_size = 64

model.compile(optimizer = "adam",
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])
model.fit(train_x, train_y, epochs = epoch, batch_size = batch_size)
model.summary()
history = model.fit(train_x, train_y, epochs = epoch, batch_size = batch_size,
                    shuffle = True, validation_split = 0.25)
def model_perf(metric, validations_metric):
    plt.plot(history.history[metric], label = str("Training " + metric))
    plt.plot(history.history[validations_metric], label = str("Validation " + metric))
    plt.legend()

model_perf("accuracy", "val_accuracy")
def model_perf(metric, validations_metric):
    plt.plot(history.history[metric], label = str("Training " + metric))
    plt.plot(history.history[validations_metric], label = str("Validation " + metric))
    plt.legend()

model_perf("loss", "val_loss")
test_loss, test_acc = model.evaluate(test_x, test_y)
print("Test accuracy: ", test_acc * 100)
print("Test loss: ", test_loss * 100)
model.save("mnist_cnn_model.h5")
print("Model saved!")
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_digit(image_path):
    # Load image in grayscale
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    
    # Convert to array
    img = img_to_array(img)
    
    # Normalize
    img = img / 255.0
    
    # Reshape for model
    img = img.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    
    print("Predicted Digit:", digit)
    
    return digit
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_digit(image_path):
    # Load image in grayscale
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    
    # Convert to array
    img = img_to_array(img)
    
    # Normalize
    img = img / 255.0
    
    # Reshape for model
    img = img.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    
    print("Predicted Digit:", digit)
    
    return digit
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

def predict_digit(image_path):
    # Load image in grayscale
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    
    # Convert to array
    img = img_to_array(img)
    
    # Normalize
    img = img / 255.0
    
    # Reshape for model
    img = img.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    
    print("Predicted Digit:", digit)
    
    return digit
predict_digit("figtest1.png")
