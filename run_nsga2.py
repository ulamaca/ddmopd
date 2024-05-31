import joblib
from optimizer import NSGA2
from config import HEMO_SCORE_PIPELINE_PATH, MIC_SCORE_PIPELNE_PATH

import datetime
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_trace_of_ga(df_trace, score_1, score_2, n_fronts=5):
    plt.figure(figsize=(10, 6))
    df_trace = df_trace.query(f'rank <= {n_fronts}')
    scatter = sns.scatterplot(data=df_trace, x=score_1, y=score_2, hue='generation', palette='viridis')
    plt.xlabel(f'score: {score_1}')
    plt.ylabel(f'score: {score_2}')
    plt.title('NSGA-II Evolution Trace')
    plt.legend()
    today_str = datetime.date.today().strftime('%y%m%d')    
    gen = df_trace['generation'].max()
    plt.savefig(f'assets/analysis/evo_traces/{today_str}_gen={gen}_rank<={n_fronts}.png')

    # Show plot
    plt.show()

if __name__ == '__main__':
    hemo_scorer = joblib.load(HEMO_SCORE_PIPELINE_PATH)
    mic_scorer = joblib.load(MIC_SCORE_PIPELNE_PATH)
    scorer_dict = {
        'rf_hemo_clf': hemo_scorer,
        'rf_mic_clf': mic_scorer
    }
    ga = NSGA2(scorers=scorer_dict,
               num_generations=15)
    df_trace = ga.run()
    plot_trace_of_ga(df_trace, 'rf_hemo_clf', 'rf_mic_clf')
    
