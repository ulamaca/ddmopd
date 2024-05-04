import pandas as pd
import joblib
from predictor.scorer import SKModelScorer
from config import *
from feature.map4_fp import *


if __name__ == '__main__':

    model = joblib.load(HEMO_SK_MODEL_PATH)
    scaler = joblib.load(HEMO_SK_SCALER_PATH)
    featurizer = seq_to_map4

    hemo_svm_scorer = SKModelScorer(
        model,
        featurizer,
        scaler
    )

    # single seq
    df = pd.read_csv(ACTV_DATA_PATH)
    test_seq = df.sample(1, random_state=10)['Sequence'].item()
    y = hemo_svm_scorer.score_single_seq(test_seq)

    # multiple seqs, slower
    test_seqs = df.sample(10, random_state=42)['Sequence'].to_list()
    ys = hemo_svm_scorer.score_seqs(test_seqs)
    breakpoint()