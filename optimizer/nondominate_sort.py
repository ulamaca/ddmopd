'''
    Non-Dominated Sorting (from NSGA-II) Related Functions

'''
from .chromosome import Chromosome
from collections import defaultdict
import pandas as pd

# maximization
# 1 p<q
# 0 p>q
# -1 neigher
def dominance_check(p: Chromosome, q:Chromosome, score_names=['p_hemo', 'p_actv'], dominance_type='max'):
    check_result = -1        

    # 2-M comparison
    p_dominate_q = all([p.scores[score_name] >= q.scores[score_name] for score_name in score_names])
    q_dominate_p = all([p.scores[score_name] <= q.scores[score_name] for score_name in score_names])            

    if p_dominate_q:
        check_result = 1
    elif q_dominate_p:
        check_result = 0        

    return check_result


def sort_nondominate(chromosomes: list[Chromosome]):    
    dict_dominance = defaultdict(list)
    dict_n_be_dominated = defaultdict(int)
    
    front_dict = defaultdict(list) # ith > chromosomes in the front
    current_rank = 0        
    
    for p in chromosomes:
        for q in chromosomes:            
            #print(p,q)
            check_p_dominates_q = dominance_check(p,q)            
            if check_p_dominates_q == 1:
                dict_dominance[p.id].append(q)
            elif check_p_dominates_q == 0:
                dict_n_be_dominated[p.id] += 1
            elif check_p_dominates_q == -1:
                pass
            else:
                raise ValueError 
        if dict_n_be_dominated[p.id] == 0:
            p.rank = current_rank
            front_dict[current_rank].append(p)
    
    current_front = front_dict[current_rank]
    
    while len(current_front) > 0:
        new_front = []
        for p in current_front:
            for q in dict_dominance[p.id]:
                dict_n_be_dominated[q.id] -= 1
                nq = dict_n_be_dominated[q.id]
                if nq == 0:
                    q.rank = current_rank + 1
                    new_front.append(q)

        current_rank += 1
        current_front = new_front
        front_dict[current_rank] = current_front
    
    return front_dict

def convert_front_dict_to_df(front_dict, score_names=['p_hemo', 'p_actv']):
    rows = []
    for rank, front in front_dict.items():
        for chrm in front:
            rows.append(chrm.__dict__)
    
    df = pd.DataFrame(rows)    
    for score_name in score_names:
        df[score_name] = df['scores'].apply(lambda x:x[score_name])
    
    return df