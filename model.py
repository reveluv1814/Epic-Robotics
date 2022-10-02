import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Normalization

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split


def chi_square(predict, actual):
    chi_stats = 0
    for i in range(len(predict)):
        chi_stats += (((actual[i]-predict[i])**2)/(predict[i]))
    
    return chi_stats


dataframe = pd.read_csv('wind.csv') 
data2 = pd.read_csv('DSCORV.csv')
result = pd.read_csv('ion.csv')

del dataframe["EPOCH"]
#del data2["EPOCH1"]
del result["EPOCH2"]
del dataframe["Unnamed: 0"]
del data2["Unnamed: 0"]
del result["Unnamed: 0"]

testData = [dataframe, data2]
testData = pd.concat(testData, axis=1)

number_of_classes = 2
number_of_features = 4 #X_train.shape[1]

X_train, X_test, y_train, y_test = train_test_split(testData, result, test_size=0.20, 
                                                    shuffle=True, random_state=2)

model = Sequential()

model.add(Normalization(input_shape=[14,],axis=None)) 
model.add(Dense(256,activation="sigmoid"))
model.add(Dense(64,activation="sigmoid"))
model.add(Dense(32,activation="sigmoid"))


model.add(Dense(5))
model.compile( tf.keras.optimizers.Adam(learning_rate = 10) , loss = "MeanSquaredError")
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
model.summary()

print(model.predict(X_test[:1]))