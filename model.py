import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('Salary.csv') # read excel
X = dataset.iloc[:, :-1].values # input
y = dataset.iloc[:, 1].values # values get predicted

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

regression = LinearRegression()
regression.fit(X_train, y_train)

y_pred = regression.predict(X_test)

pickle.dump(regression, open('model.pkl', 'wb'))