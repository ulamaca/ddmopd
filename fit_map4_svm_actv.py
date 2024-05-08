'''
    TODO:
        1. check why train_clf having good performance but cross_val_score results are bad...
            >> that is because the shuffling is not random! (so, the generalization is bad)
'''
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, accuracy_score
import joblib
from config import MODEL_PATH
import json

# data prep
def train_clf(clf, X, Y, x, y):
    clf = clf.fit(X, Y)
    predicted_training = clf.predict_proba(X)
    predicted_test = clf.predict_proba(x)

    roc_training = roc_auc_score(Y, predicted_training[:,1])
    roc_test = roc_auc_score(y, predicted_test[:,1])
    
    y_test = clf.predict(x)
    acc_test = accuracy_score(y, y_test)
    precision_test = precision_score(y, y_test)
    recall_test = recall_score(y, y_test)
    f1_test = f1_score(y, y_test)
    mcc_test = matthews_corrcoef(y, y_test)

    result = {
        'roc-auc_training': roc_training,
        'roc-auc_test': roc_test,
        'acc_test': acc_test,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f1_test': f1_test,
        'mcc_test': mcc_test
    }
    print(json.dumps(result, indent=4))
    
    return clf

NORM = True
norm_types = ['std', 'minmax']
NORM_TYPE = norm_types[1]
GRID_SEARCH = False

if __name__ == "__main__":
    df = pd.read_csv('data/dbaasp/fine_tune_activity.csv')    
    df = df.reset_index(drop=True)
        
    # if processed data not exist >> preprop    
    X_actv = np.load('data/dbaasp/processed_map4_fp_actv.npy')
        
    df_train = df.query("Set == 'training'")
    df_test = df.query("Set == 'test'")
    

    ids_train = df_train.index.to_list()
    ids_test = df_test.index.to_list()
    print("num-train, num-test: ", len(ids_train), len(ids_test))
        
    X = X_actv[ids_train]
    x = X_actv[ids_test]
    Y = df_train['activity'].values
    y = df_test['activity'].values

    if NORM:
        if NORM_TYPE == 'minmax':
            scaler = preprocessing.MinMaxScaler()
        elif NORM_TYPE == 'std':
            scaler = preprocessing.StandardScaler()
        else:
            raise ValueError
        X = scaler.fit_transform(X)
        x = scaler.transform(x)
            
    # prep X,Y,x,y
    # training        
    if GRID_SEARCH:
        # SVM optimization for ROC auc
        param_grid = {'C': [0.1,1, 10, 100], \
                'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf']}
        grid = GridSearchCV(SVC(probability=True),param_grid, scoring='roc_auc',\
                        refit=True,verbose=2, n_jobs=3)
        grid.fit(X,Y)
        print(grid.best_params_)
        best_parmas = grid.best_params_
        best_parmas['probability'] = True # needs to turn on this config
        best_config_svm = SVC(**best_parmas)
        SVM = train_clf(best_config_svm, X, Y, x, y)
    else:
        # cv        
        clf = RandomForestClassifier(n_estimators=500, max_depth=10)
        # cross-validation
        shuffle = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_score_dict = {
            'roc_auc': cross_val_score(clf, X, Y, cv=shuffle, scoring='roc_auc'),
            'f1': cross_val_score(clf, X, Y, cv=shuffle, scoring='f1'),
            'accuracy': cross_val_score(clf, X, Y, cv=shuffle, scoring='accuracy'),
            'precision': cross_val_score(clf, X, Y, cv=shuffle, scoring='precision'),
            'recall': cross_val_score(clf, X, Y, cv=shuffle, scoring='recall')
        }
        df_scores = pd.DataFrame(cv_score_dict)
                                         
        clf = train_clf(clf, X, Y, x, y)  
        # TODO >> save the one trained with full data!      
        joblib.dump(clf, os.path.join(MODEL_PATH, 'actv_rf_minmax_norm_map4.joblib'))        

        
    breakpoint()