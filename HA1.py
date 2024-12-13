import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from scipy import stats
import csv

import pywt
from itertools import cycle

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, AvgPool1D, LSTM, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23
# Matplotlib settings
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.color'] = 'g'
# plt.rcParams['axes.grid'] = True

# Denoise function
def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')
    return datarec

# Path and parameters
path = 'mitbih_database'
window_size = 180
classes = ['N', 'L', 'R', 'A', 'V']
n_classes = len(classes)
count_classes = [0] * n_classes

X, y = [], []

# Load data
filenames = next(os.walk(path))[2]
records = [path + "/" + f for f in filenames if f.endswith('.csv')]
annotations = [path + "/" + f for f in filenames if f.endswith('.txt')]

for r, record in enumerate(records):
    signals = []
    with open(record, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip header
        for row in reader:
            signals.append(int(row[1]))

    signals = stats.zscore(signals)

    # Process annotations
    with open(annotations[r], 'r') as file:
        data = file.readlines()[1:]  # Skip header
        for line in data:
            splitted = list(filter(None, line.split(' ')))
            pos = int(splitted[1])
            arrhythmia_type = splitted[2]
            if arrhythmia_type in classes:
                arrhythmia_index = classes.index(arrhythmia_type)
                count_classes[arrhythmia_index] += 1
                if window_size <= pos < (len(signals) - window_size):
                    beat = signals[pos - window_size:pos + window_size]
                    X.append(beat)
                    y.append(arrhythmia_index)

# Convert data to DataFrame
X = np.array(X)
y = np.array(y)
data_df = pd.DataFrame(X)
data_df['label'] = y

# Balancing dataset
balanced_df = pd.concat([
    resample(data_df[data_df['label'] == i], replace=True, n_samples=5000, random_state=42)
    for i in range(n_classes)
])

# Split into train/test
train, test = train_test_split(balanced_df, test_size=0.2)
train_x = train.iloc[:, :-1].values.reshape(len(train), -1, 1)
train_y = to_categorical(train['label'].values, num_classes=n_classes)
test_x = test.iloc[:, :-1].values.reshape(len(test), -1, 1)
test_y = to_categorical(test['label'].values, num_classes=n_classes)

# Model definition
model = Sequential([
    Conv1D(filters=16, kernel_size=13, padding='same', activation='relu', input_shape=(360, 1)),
    AvgPool1D(pool_size=3, strides=2),
    Conv1D(filters=32, kernel_size=15, padding='same', activation='relu'),
    AvgPool1D(pool_size=3, strides=2),
    Conv1D(filters=64, kernel_size=17, padding='same', activation='relu'),
    AvgPool1D(pool_size=3, strides=2),
    Conv1D(filters=128, kernel_size=19, padding='same', activation='relu'),
    AvgPool1D(pool_size=3, strides=2),
    LSTM(units=64, return_sequences=False, activation='tanh'),
    Dense(35, activation='relu', kernel_regularizer=regularizers.l2(0.0001)),
    Dense(n_classes, kernel_regularizer=regularizers.l2(0.0001)),
    Softmax()
])

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Training
history = model.fit(train_x, train_y, batch_size=36, epochs=20, validation_data=(test_x, test_y))

# Plot accuracy and loss
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate model
score = model.evaluate(test_x, test_y)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

# Save model
model.save("arrhythmia_model.h5")

# ROC Curve
y_probs = model.predict(test_x)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
