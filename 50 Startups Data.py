import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dg=pd.read_csv(r'C:\Users\lenovo\Desktop\50 Startups Data.csv')
X=dg.iloc[:, :-1]
Y=dg.iloc[:,4]
states=pd.get_dummies(X['State'],drop_first=True)
X=X.drop('State',axis=1)
X=pd.concat([X,states],axis=1)
#Now we will split the data set in to test and train break up
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )
#So i will be fitting the multiple Linear Regression in to trainning set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
#Now test set prediction
Y_pred=regressor.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(Y_test,Y_pred)
r2_score
