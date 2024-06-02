import os 

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(REPO_PATH, 'assets/models')
HEMO_SK_SCALER_PATH = os.path.join(MODEL_PATH, 'hemo_scaler_minmax_norm_map4.joblib')
HEMO_SK_MODEL_PATH = os.path.join(MODEL_PATH, 'hemo_rf_minmax_norm_map4.joblib')
ACTV_SK_SCALER_PATH = os.path.join(MODEL_PATH, 'actv_scaler_minmax_norm_map4.joblib')
ACTV_SK_MODEL_PATH = os.path.join(MODEL_PATH, 'actv_rf_minmax_norm_map4.joblib')

DATA_PATH = os.path.join(REPO_PATH, 'data')
ACTV_DATA_PATH = os.path.join(DATA_PATH, 'dbaasp/fine_tune_activity.csv')
HEMO_DATA_PATH = os.path.join(DATA_PATH, 'dbaasp/fine_tune_hemolysis.csv')


AMP_SCORE_PIPELINE_PATH = os.path.join(REPO_PATH, 'score_pipeline/amp_kmer_rf.joblib')
MIC_SCORE_PIPELNE_PATH = os.path.join(REPO_PATH, 'score_pipeline/mic_ecoli_kmer_rf.joblib')
HEMO_SCORE_PIPELINE_PATH = os.path.join(REPO_PATH, 'score_pipeline/hemolysis_kmer_rf.joblib')


# Mutation Configs
MUTATION_DATA_PATH = os.path.join(DATA_PATH, 'cpp/cpp_predictor_dataset.csv')