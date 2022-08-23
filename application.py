__author__ = "Ettore Caputo"
__version__ = "0.1"
__email__ = "ettore.caputo27@gmail.com"


##
## IMPORTS
##
import tensorflow as tf             # version 2.8.2
import keras                        # version 2.8.0
import pandas as pd                 # version 1.3.5
import librosa                      # version 0.8.1
import numpy as np                  # version 1.21.6
import matplotlib.pyplot as plt     # version 3.2.2 (matplotlib)
import json                         # version 2.0.9
from sklearn.utils import shuffle   # version 1.0.2 (sklearn)
from sklearn.model_selection import train_test_split 
import librosa.display as lplt
import os
import math

##
## CONSTANTS
##
DATASET_PATH = '/content/kaggle/gtzan-dataset-music-genre-classification'
JSON_PATH = "./data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 #in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
NUM_SEGMENTS = 10

SELECTED_GENRES = {
    "rock":0, 
    "metal":1
}

##
## KAGGLE INSTALLATION AND DOWNLOAD DATASET
##
! pip install kaggle
! mkdir ~/.kaggle
! cp /kaggle.json ~/.kaggle/ # the file kaggle.json is personal
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download andradaolteanu/gtzan-dataset-music-genre-classification
! mkdir /content/kaggle
! mv /content/gtzan-dataset-music-genre-classification.zip /content/kaggle/gtzan-dataset-music-genre-classification.zip
! unzip /content/kaggle/gtzan-dataset-music-genre-classification.zip -d /content/kaggle/gtzan-dataset-music-genre-classification
! rm /content/kaggle/gtzan-dataset-music-genre-classification.zip

##
## SAVE MFCC TO JSON FILE
##
def save_mfcc_to_json(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    genres = SELECTED_GENRES


    # loop through all genre sub-folder
    for genre in genres:
        data["mapping"].append(genre)
        print("\nProcessing: {}".format(genre))

        for root, dirs, files in os.walk(dataset_path+"/"+genre):
            for filename in files:

                # load audio file
                file_path = os.path.join(root, filename)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(genres[genre])

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


save_mfcc_to_json(DATASET_PATH+"/Data/genres_original", JSON_PATH, num_segments=NUM_SEGMENTS)

##
## LOAD DATA FROM JSON FILE 
## AND SUBDIVISION OF THE DATASET
##
def load_data(data_path):
    print("Data loading\n")
    with open(data_path, "r") as fp:
        data = json.load(fp)
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    print("Original Shape:  x:{} \t y:{} \n".format(x.shape, y.shape))
    return x, y



def prepare_datasets(test_size, val_size):
    #load the data
    x, y = load_data(JSON_PATH)
    x, y = shuffle(x, y) 
    x_train,x_test,y_train,y_test = train_test_split(x , y , test_size = test_size)
    x_train,x_val,y_train,y_val = train_test_split(x_train , y_train , test_size = val_size)
    
    print("Train Shape:  x:{} \t y:{} \n".format(x_train.shape, y_train.shape))
    print("Test Shape:  x:{} \t y:{} \n".format(x_test.shape, y_test.shape))
    print("Validation Shape:  x:{} \t y:{} \n\n\n\n".format(x_val.shape, y_val.shape))
    return x_train, x_val, x_test, y_train, y_val, y_test


x_train, x_val, x_test, y_train, y_val, y_test = prepare_datasets(0.2, 0.15)

##
## COMPILE, TRAIN AND TEST MODEL
##
def build_model(input_shape):
    model = keras.Sequential()
    model.add(tf.keras.layers.LSTM(64, input_shape = input_shape, return_sequences = True))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model


model = build_model([x_train.shape[1],x_train.shape[2]])
model.compile(
    optimizer = 'adam', 
    loss = 'binary_crossentropy', 
    metrics = ['accuracy']
)
model.summary()
history = model.fit(x_train, y_train, 
                    validation_data=(x_val, y_val), 
                    batch_size = 32, epochs=100,
                    callbacks=[
                        EarlyStopping(monitor='val_loss', mode='min', 
                                      verbose=1, patience=6, 
                                      restore_best_weights=True
                                     )
                    ])



def plot_history(history):
    fig,axs = plt.subplots(2, sharex=True, figsize=(10,10))
    axs[0].plot(history.history["accuracy"],label="train accuracy")
    axs[0].plot(history.history["val_accuracy"],label="validation accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc='lower right')
    axs[0].set_title("Accuracy eval")
    
    axs[1].plot(history.history["loss"],label="train error")
    axs[1].plot(history.history["val_loss"],label="validation error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc='upper right')
    plt.show()

plot_history(history)
test_error , test_accuracy = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy on test is {}".format(test_accuracy))
print("Test error is {}".format(test_error))
