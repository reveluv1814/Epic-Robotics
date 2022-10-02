from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

dataframe = pd.read_csv('wind.csv') 
data2 = pd.read_csv('DSCORV.csv')
result = pd.read_csv('ion.csv')

del dataframe["EPOCH"]
del data2["EPOCH1"]
del result["EPOCH2"]
del dataframe["Unnamed: 0"]
del data2["Unnamed: 0"]
del result["Unnamed: 0"]

testData = [dataframe, data2]
testData = pd.concat(testData, axis=1)


X_train, X_test, y_train, y_test = train_test_split(testData, result, test_size=0.20, 
                                                    shuffle=True, random_state=2)


regr = MultiOutputRegressor(Ridge(random_state=123)).fit(X_train, y_train)
print(regr.predict(X_test[:1]))
print(y_test[:1])
