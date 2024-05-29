from optimizer import NSGA2
from config import HEMO_SCORE_PIPELINE_PATH, MIC_SCORE_PIPELNE_PATH
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
class KMerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [' '.join(self.extract_kmers(seq)) for seq in X]

    def extract_kmers(self, sequence):
        # if len(sequence) < self.k:
        #     return []
        return [sequence[i:i+self.k] for i in range(len(sequence) - self.k + 1)]

if __name__ == '__main__':
    hemo_scorer = joblib.load(HEMO_SCORE_PIPELINE_PATH)
    mic_scorer = joblib.load(MIC_SCORE_PIPELNE_PATH)
    scorer_dict = {
        'rf_hemo_clf': hemo_scorer,
        'rf_mic_clf': mic_scorer
    }
    ga = NSGA2(scorers=scorer_dict)
    ga.run()
