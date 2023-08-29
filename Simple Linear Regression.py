# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 02:00:38 2023

@author: 91879
"""

## Build a simple linear regression model by performing EDA and do necessary transformations and select the best model using R or Python.

    # 1) Delivery_time -> Predict delivery time using sorting time 
    
    
    # 1-Import data
import pandas as pd
df1 = pd.read_csv("D:/Shiva Data Science/ExcelR Assignments/Simple Linear Regression/delivery_time.csv")
df1

Sorting_Time = df1[["Sorting Time"]]
Delivery_Time = df1["Delivery Time"]

    # 2-Exploratory data analysis
import matplotlib.pyplot as plt
plt.scatter(Sorting_Time, Delivery_Time)
plt.show()
df1.corr()

    # 3 Split the variables as X and Y
x = Sorting_Time
y = Delivery_Time


    # 4 model fitting
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(x, y)

LR.intercept_
LR.coef_

y_pred = LR.predict(x)
y_pred

plt.scatter(x, y, color="blue")
plt.scatter(x, y_pred, color="red")
plt.plot(x, y_pred, color="black")
plt.show()



#===========================================================================================>>>>>>>>>>>>>>>>>>>>

# 2) Salary_hike -> Build a prediction model for Salary_hike

    # 1-Import data
import pandas as pd
df2 = pd.read_csv("D:/Shiva Data Science/ExcelR Assignments/Simple Linear Regression/Salary_Data.csv")
df2

YearsExperience = df2[["YearsExperience"]]
Salary = df2["Salary"]

    # 2-Exploratory data analysis
import matplotlib.pyplot as plt
plt.scatter(YearsExperience, Salary)
plt.show()
df2.corr()

    # 3 Split the variables as X and Y
x = YearsExperience
y = Salary


    # 4 model fitting
from sklearn.linear_model import LinearRegression

LR = LinearRegression()
LR.fit(x, y)

LR.intercept_
LR.coef_

y_pred = LR.predict(x)
y_pred

plt.scatter(x, y, color="blue")
plt.scatter(x, y_pred, color="red")
plt.plot(x, y_pred, color="black")
plt.show()





