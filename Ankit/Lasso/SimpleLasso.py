# -*- coding: utf-8 -*-
import pandas as pd
#from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.linear_model import Lasso 

#read the data
train = pd.read_csv('../../Common/Data/train.csv')
test = pd.read_csv('../../Common/Data/test.csv')

#we'll modify the sample submission file to make our submission
submission = pd.read_csv('../../Common/Data/sample_submission.csv')

#prep the data for sklearn by separating predictors and response
X = train.drop('Hazard', axis = 1)
y = train['Hazard']

#one-hot the categoricals
num_X = pd.get_dummies(X)
num_Xt = pd.get_dummies(test)

#fit the model and predict
model = Lasso().fit(num_X,y)
prediction = model.predict(num_Xt)

#write the submission file
submission['Hazard'] = prediction
submission.to_csv('basic_RF.csv', index = False)
