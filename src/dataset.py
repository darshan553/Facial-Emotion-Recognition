import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']




def load_fer2013(csv_path, image_size=(48,48)):
"""Load FER-2013 CSV from Kaggle. Returns X (N,H,W,1), y (N,)
The CSV has columns: emotion, pixels, Usage
"""
df = pd.read_csv(csv_path)
pixels = df['pixels'].tolist()
emotions = df['emotion'].tolist()
X = []
for pix in pixels:
arr = np.fromstring(pix, sep=' ', dtype=np.uint8)
img = arr.reshape(48,48)
img = cv2.resize(img, image_size)
img = img.astype('float32') / 255.0
img = np.expand_dims(img, -1) # grayscale channel
X.append(img)
X = np.array(X)
y = np.array(emotions)
return X, y




def prepare_data(csv_path, test_size=0.15, val_size=0.1, image_size=(48,48), one_hot=True):
X, y = load_fer2013(csv_path, image_size=image_size)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
val_relative = val_size / (1 - test_size)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_relative, stratify=y_trainval, random_state=42)
if one_hot:
y_train_cat = to_categorical(y_train, num_classes=len(EMOTIONS))
y_val_cat = to_categorical(y_val, num_classes=len(EMOTIONS))
y_test_cat = to_categorical(y_test, num_classes=len(EMOTIONS))
return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, y_train, y_val, y_test
return X_train, X_val, X_test, y_train, y_val, y_test




def get_augmentor():
datagen = ImageDataGenerator(
rotation_range=15,
width_shift_range=0.1,
height_shift_range=0.1,
zoom_range=0.1,
horizontal_flip=True
)
return datagen
