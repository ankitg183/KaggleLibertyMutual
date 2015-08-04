# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as GR
from sklearn.metrics import  make_scorer

from sklearn import grid_search

def gini_score(y_true, y_pred):
    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true

def main():
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
    #GBR = GR( loss='ls',n_estimators=1000,subsample=0.9)
    
    score = make_scorer(gini_score, greater_is_better=True)
    
    #parameters = {'n_estimators':(300,500,1000),'learning_rate':(0.02,0.05,0.1,0.2), 'max_depth':(3,4)}
    parameters = {'n_estimators':(1000,2000),'learning_rate':(0.02,0.05,0.001), 'max_depth':(2,3,4,5)}
    #parameters = {'n_estimators':(10,20),'learning_rate':(0.02,0.05)}
    #parameters = {'n_estimators':[100],'learning_rate':[0.1], 'max_depth':(1,2,3,4)}
    #parameters = {'n_estimators':[10],'learning_rate':[0.5], 'max_depth':[5,10]}
    GBR = GR()
    clf = grid_search.GridSearchCV(GBR, parameters,cv=4,verbose=10, n_jobs=10,scoring=score)
    
    #clf = grid_search.RandomizedSearchCV(GBR, parameters,cv=2,verbose=10,n_iter=10, n_jobs=1)
    
    
    clf.fit(num_X, y)
    
    print clf.best_params_
    print clf.best_score_
    
   # GBR2= clf.best_estimator_
    #GBR2.set_params(clf.best_params_)
    #GBR.fit(num_X, y)
    #model = GR().fit(num_X,y)
    
    #prediction1 = GBR.predict(num_Xt)
    prediction2 = clf.predict(num_Xt)
    
    #write the submission file
    #submission['Hazard'] = prediction1
    #submission.to_csv('GBRCV1.csv', index = False)
    #print GBR.oob_improvement_
    
    submission['Hazard'] = prediction2
    submission.to_csv('GBRCV2.csv', index = False)


if __name__ == "__main__":
    main()