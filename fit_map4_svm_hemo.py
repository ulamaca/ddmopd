'''
    TODO:
        1. check why train_clf having good performance but cross_val_score results are bad...
            >> that is because the shuffling is not random! (so, the generalization is bad)
'''
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import joblib
from config import MODEL_PATH

# data prep
def train_clf(clf, X, Y, x, y):
    clf = clf.fit(X, Y)
    predicted_training = clf.predict_proba(X)
    predicted_test = clf.predict_proba(x)

    roc_training = roc_auc_score(Y, predicted_training[:,1])
    roc_test = roc_auc_score(y, predicted_test[:,1])

    print("ROC AUC Train: {:.3f}".format(roc_training))
    print("ROC AUC Test: {:.3f}".format(roc_test))
    return clf

NORM = True
norm_types = ['std', 'minmax']
NORM_TYPE = norm_types[1]
GRID_SEARCH = False
if __name__ == "__main__":
    df = pd.read_csv('data/dbaasp/fine_tune_hemolysis.csv')
    df = df.query('isNotHemolytic >= 0') # leaving only labeled data
    df = df.reset_index(drop=True)

    df_train = df.query("Set == 'training'")
    df_test = df.query("Set == 'test'")
    Y = df_train['isNotHemolytic'].values
    y = df_test['isNotHemolytic'].values

    ids_train = df_train.index.to_list()
    ids_test = df_test.index.to_list()
    print("num-train, num-test: ", len(ids_train), len(ids_test))


    X_npy = np.load('data/dbaasp/processed_map4_fp_hemolysis.npy')
    X = X_npy[ids_train]
    x = X_npy[ids_test]

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
        clf = SVC(probability=True, gamma = 1.0, C=0.1)
        # cross-validation
        # shuffle = KFold(n_splits=5, shuffle=True, random_state=42)
        # cv_scores = cross_val_score(clf, X, Y, cv=shuffle, scoring='roc_auc')
        # print('cross-validation auc_roc:', cv_scores)
        SVM = train_clf(clf, X, Y, x, y)
        joblib.dump(SVM, os.path.join(MODEL_PATH, 'hemo_svm_minmax_norm_map4.joblib'))
        joblib.dump(scaler, os.path.join(MODEL_PATH, 'hemo_scaler_minmax_norm_map4.joblib'))

    
    
    breakpoint()