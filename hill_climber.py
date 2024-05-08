'''
    240416: addded crossover to increase diversity
    240420: support a new cross_over: cut_inverted_cross_over

    implicit dependency: data/cpp (this is to be deprecated in the future)
'''

import pandas as pd
import random
from manipulate.mutate import Genetic_Mutations
from manipulate.cross_over import naive_cross_over, cut_inverted_cross_over, random_inverted_cross_over

# scorer related
import joblib
from config import *
from predictor.scorer import SKModelScorer

def legalize_seq_for_clf(seq: str):
    '''
        as the title suggests
        TODO: to see why this vocab are not valid for the given vocab
    '''
    seq = seq.replace(' ', '')
    seq = seq.replace('X', '')
    seq = seq.replace('B', '')

    return seq

def get_svm_scorer(task='hemo'):
    from feature.map4_fp import seq_to_map4
    
    if task == 'hemo':
        model = joblib.load(HEMO_SK_MODEL_PATH)
        scaler = joblib.load(HEMO_SK_SCALER_PATH)
    elif task == 'actv':
        model = joblib.load(ACTV_SK_MODEL_PATH)
        scaler = joblib.load(ACTV_SK_SCALER_PATH)
    else:
        raise ValueError
    featurizer = seq_to_map4

    hemo_svm_scorer = SKModelScorer(
        model,
        featurizer,
        scaler
    )

    return hemo_svm_scorer

def create_population_df(population, scores):
    rows = [{'Sequence': seq, 'score': score} for seq, score in zip(population, scores)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # 1 define mutator
    data_path = 'data/cpp/cpp_predictor_dataset.csv'
    mutator = Genetic_Mutations(data_path=data_path)
    
    # 2 define pred model
    scorer = get_svm_scorer(task='actv') 
    
    # 3 hill climber
    n_step = 50
    n_population = 20 # real poputation size = n_population * (1.3) + 1
    # mutate_rate = 0.5
    cross_over_rate = 0.8
    cross_over_type = 'c_icv'
    random.seed(42)

    #seed_seq = 'TKPRPGP' # peptide: Selank
    seed_seq = 'GALFKVASKVL' # from DBAASP, whose NotHemo is 1
    seed_seq = 'GLPALISWSKRKRQQ' # from DBAASP, whose NotHemo is 1, in training

    generations = []
    
    # initial population
    population = [legalize_seq_for_clf(mutator.mutate(seed_seq)) for _ in range(n_population)]    

    for step in range(n_step):                
        # get prediction            
        scores = scorer.score_seqs(population)        
        df = create_population_df(population, scores)
        df['generation'] = step + 1

        breakpoint()
        
        # generation trace
        generations.append(df)
                
        # mutate & legalize
        df_best = df.sort_values('score').tail(1)
        best_step = df_best['Sequence'].item()
        score = df_best['score'].item()        
        population = [legalize_seq_for_clf(mutator.mutate(best_step)) for _ in range(n_population-1)] + [best_step] 

        # TODO: to sample survivors (seq being intact) according to its current score

        # cross over
        mates = [random.sample(population, 2) for _ in range(int(n_population*cross_over_rate))]
        
        if cross_over_type == 'r_icv':
            offsprings = [random_inverted_cross_over(parent[0], parent[1]) for parent in mates]
        elif cross_over_type == 'c_icv':
            offsprings = [cut_inverted_cross_over(parent[0], parent[1]) for parent in mates]
        elif cross_over_type == 'naive':
            offsprings = [naive_cross_over(parent[0], parent[1]) for parent in mates]
        
        offsprings = [legalize_seq_for_clf(c) for c in offsprings]
        population.extend(offsprings)    
        population = list(set(population) ) # make them unique()

        print(f"step={step}, score={score}, current population size= {len(population)}, current seed peptide: {best_step}")    

        
    df = pd.concat(generations).reset_index(drop=True)

    breakpoint()