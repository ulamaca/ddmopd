import os 


MODEL_PATH = 'assets/models'
HEMO_SK_SCALER_PATH = os.path.join(MODEL_PATH, 'hemo_scaler_minmax_norm_map4.joblib')
HEMO_SK_MODEL_PATH = os.path.join(MODEL_PATH, 'hemo_rf_minmax_norm_map4.joblib')
ACTV_SK_SCALER_PATH = os.path.join(MODEL_PATH, 'actv_scaler_minmax_norm_map4.joblib')
ACTV_SK_MODEL_PATH = os.path.join(MODEL_PATH, 'actv_rf_minmax_norm_map4.joblib')
ACTV_DATA_PATH = 'data/dbaasp/fine_tune_activity.csv'
HEMO_DATA_PATH = 'data/dbaasp/fine_tune_hemolysis.csv'