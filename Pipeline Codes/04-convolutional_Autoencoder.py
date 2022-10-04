from sklearn.ensemble import RandomForestClassifier

import DataReading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sns as sns
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from keras.callbacks import TensorBoard
from keras.models import Sequential
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.layers import Dense



class Autoencoder(Model):  # consists of encoder and decoder
  def __init__(self):

    super(Autoencoder, self).__init__()
    # Encoder
    auto_input = layers.Input(shape = (32,32,1))
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(auto_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(x)
    auto_output= layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    self.autoencoder = Model(auto_input,auto_output)

  def call(self, x):
    return self.autoencoder(x)


  def ae_summary(self):
    self.ae_model.compile(optimizer= "adam",loss = losses.MeanSquaredError()) #loss = "binary_crossentropy"
    self.ae_model.summary()


class Classifier():
        def __init__(self):
                self.dim = 3
                self.clf = RandomForestClassifier(max_depth=2, random_state=0)

        def classifier_fit(self,x,y):
                self.clf.fit(x,y)

        def predictor(self,x):
                predicted_x = self.clf.predict(x)
                return predicted_x

path_healthy = "D:/UNI/caseStudy/IR_Images/IR_Images/healthy/healthy_csvData/94689"
path_faulty = "D:/UNI/caseStudy/IR_Images/IR_Images/FehlerL1K1K5/FehlerL1K1K5/94689"

train_auto,health_labels_train,valid_auto,auto_labels_valid, train_classifier,train_classifier_labels,valid_data,valid_data_labels,test_data,test_labels  = DataReading.data_preparation(path_healthy, path_faulty)
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss="binary_crossentropy")

autoencoder_train = autoencoder.fit(train_auto,train_auto,
                epochs=20,
                shuffle=True,
                validation_data=(valid_data, valid_data),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# the model is built, now lets make test the model:

#autoencoder(training).numpy()

out_train = autoencoder(train_auto).numpy()
Autoout_train_classifier = autoencoder(train_classifier).numpy()
out_test = autoencoder(test_data).numpy()


train_classifier_errors = []
for i in range(len(train_classifier)):
        error = ((Autoout_train_classifier[i,:,:]-train_classifier[i,:,:])**2).sum()
        train_classifier_errors.append(error)


# labels = np.one(len(training))

classifier = Classifier()
train_classifier_errors = np.array(train_classifier_errors).reshape((-1,1))
classifier.classifier_fit(train_classifier_errors, train_classifier_labels)  # ???? what other parameters are required??
train_predicted_labels = classifier.predictor(train_classifier_errors)
matrix_confsion = confusion_matrix(train_classifier_labels, train_predicted_labels)
# sns.heatmap(matrix_confsion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
print(matrix_confsion)
print(f"accuracy for train data is = {accuracy_score(train_classifier_labels, train_predicted_labels)}")


test_errors = []
for i in range(len(test_data)):
        error = ((out_test[i,:,:]-test_data[i,:,:])**2).sum()
        test_errors.append(error)

test_errors = np.array(test_errors).reshape((-1,1))
test_predicted_labels = classifier.predictor(test_errors)
matrix_confsion = confusion_matrix(test_labels, test_predicted_labels)
# sns.heatmap(matrix_confsion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
print(matrix_confsion)
print(f"accuracy for test data is = {accuracy_score(test_labels, test_predicted_labels)}")