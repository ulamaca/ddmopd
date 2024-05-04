import pandas as pd
import numpy as np

# abstract class
class Scorer:
    '''
        score a Sequence according to rule/ ML-model
    '''    
    def score_single_seq(self, seq: str):
        pass

    def score_seqs(self, seqs: list[str]):
        pass

    def score_df_seqs(self, df_seqs: pd.DataFrame, seq_col='sequence', parallel=False):
        pass


class SKModelScorer:
    '''
        score a 
    '''
    def __init__(self, model, featurizer, scaler=None):
        self.model = model
        self.featurizer = featurizer
        self.scaler = scaler

    def seq_to_model_input(self, seq):
        x = self.featurizer(seq)      
        return x

    def score_single_seq(self, seq:str):
        '''
            assume dim==1 is the class-of-interest so we output the proba value of it
        '''        
        
        x = self.seq_to_model_input(seq)        
        x = x.reshape(1, -1)  
        if self.scaler is not None:
            x = self.scaler.transform(x)
        y = self.model.predict_proba(x)
        return y[:, 1]

    def score_seqs(self, seqs):
        '''
            assume dim==1 is the class-of-interest so we output the proba value of it
        '''
        Xs = [self.seq_to_model_input(seq) for seq in seqs]
        X = np.array(Xs)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        assert len(X.shape) == 2, f"shape of X is invalid: {X.shape}"
        assert X.shape[0] == len(seqs)
        Y = self.model.predict_proba(X)

        return Y[:, 1]
