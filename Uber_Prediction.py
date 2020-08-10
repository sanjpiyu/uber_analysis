import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from flask import Flask, request, jsonify, render_template

ub = pd.read_csv("taxo.csv")

data_x = ub.iloc[:,0:-1]
data_y = ub.iloc[:,-1]



x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size=0.3,random_state=0)



reg = LinearRegression()
reg.fit(x_train,y_train)



print("Train Score:",reg.score(x_train,y_train))


print("Train Score:",reg.score(x_test,y_test))


pickle.dump(reg, open('taxi.pkl','wb'))


model = pickle.load(open('taxi.pkl','rb'))

print(model.predict([[80,1770000,6000,85]]))

