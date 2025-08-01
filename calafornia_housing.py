import pandas as pd

from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
# print(dataset)

# setting up the columns to feture names
dataset.columns=df.feature_names

# print(dataset)
# diving data into independent features and dependent features
x=dataset
y=df.target

# print(y)

# train_test_split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=30, random_state=42)

# standardizing the dataset{gives you te best output}
from sklearn.preprocessing import StandardScaler
# initializing the stanard scaler
scaler=StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test= scaler.transform(x_test)
# implementing linear regression
from sklearn.linear_model import LinearRegression
# cross-validation
from sklearn.model_selection import cross_val_score 
# regression object
regression=LinearRegression()
regression.fit(x_train,y_train)
mse=cross_val_score(regression,x_train,y_train,scoring='neg_mean_squared_error', cv=5)
np.mean(mse)

# prediction
reg_pred=regression.predict(x_test)
# print(reg_pred)

# lets visualize
import seaborn as sns
sns.displot(reg_pred-y_test, kind='kde')
plt.show()
