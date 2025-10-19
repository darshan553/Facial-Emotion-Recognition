import numpy as np
import cv2


EMOTIONS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']


def expand_gray_to_3ch(x):
# x shape: (H,W,1) or (batch,H,W,1)
if x.ndim == 3:
return np.concatenate([x, x, x], axis=-1)
elif x.ndim == 4:
return np.concatenate([x, x, x], axis=-1)
else:
raise ValueError('Unexpected input dims')


# small helper to compute class weights
from sklearn.utils.class_weight import compute_class_weight


def get_class_weights(y):
# y: 1D array of labels
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
return {int(c): float(w) for c, w in zip(classes, weights)}
