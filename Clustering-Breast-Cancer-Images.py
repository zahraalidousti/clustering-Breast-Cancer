import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras

dataset = pd.read_csv('breast_cancer.csv', header = None)

dataset.head(5)

data = dataset.iloc[:,:-1]

labels = dataset.iloc[:,-1]

data

labels

data = data.replace('?', np.nan)

data.iloc[235,:]

data = data.fillna(0)

data.iloc[235,:]

from sklearn.preprocessing import normalize

data = normalize(data, axis = 0)

data

labels = np.array(labels)

labels = np.where(labels ==2, 0, 1)
labels

from sklearn.model_selection import train_test_split

x_train, x_test, y_train , y_test = train_test_split(data, labels, test_size = 0.20, random_state = 42)

x_train.shape

x_test.shape

y_test

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping


inputs = Input(shape =(9,))
for i in range(0,15):
    x = Dense(6)(inputs)  
    x = Activation('relu')(x)
    x = Dense(4)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
outputs = Activation('sigmoid')(x) 

model = Model(inputs, outputs)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])

model.summary()

callback = EarlyStopping(monitor='val_loss', patience=50)
history = model.fit(x_train, y_train, epochs=300, batch_size=5,
                    validation_data= (x_test, y_test), callbacks =callback, verbose =1)

result = model.predict(x_test)
result

result = np.where(result>=0.5, 1, 0)
result

result.shape

result = result.reshape(result.shape[0])
result

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print('accuracy = ',accuracy_score(y_test, result))

pd.DataFrame(confusion_matrix(y_test, result))

print(classification_report(y_test, result))
