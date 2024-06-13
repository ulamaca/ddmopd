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


def sort_nondominate(chromosomes: list[Chromosome], score_names: list[str]):    
    dict_dominance = defaultdict(list)
    dict_n_be_dominated = defaultdict(int)
    
    front_dict = defaultdict(list) # ith > chromosomes in the front
    current_rank = 0        
    
    for p in chromosomes:
        for q in chromosomes:            
            #print(p,q)
            check_p_dominates_q = dominance_check(p,q, score_names)            
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
    
    print(' done NDS (non-dominated sorting)')
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

def calc_crowding_distance(chromosomes: list[Chromosome], score_names: list[str]):
    '''
        to check, if CD calc process is doing wrong for retrieving data
    '''    
    INF = 1e8
    
    # init cd value
    for chromosome in chromosomes:
        chromosome.crowding_distance = 0.0
    
    max_rank = max([chr.rank for chr in chromosomes])

    # calculate cd
    for score_name in score_names:
        for rank in range(1, max_rank+1):
            # get only chromosomes with rank == rank
            chromosomes_m = [chr for chr in chromosomes if chr.rank == rank]
            chromosomes_m = sorted(chromosomes_m, key=lambda x:x.scores[score_name] )            
            N_m_r = len(chromosomes_m)                        
            for i, chromosome in enumerate(chromosomes_m):
                if i == 0 or i == N_m_r-1:
                    chromosome.crowding_distance += INF
                else:                                       
                    delta = chromosomes_m[i+1].scores[score_name] - chromosomes_m[i-1].scores[score_name] 
                    
                    if delta < 0:
                        print('delta should >= 0')
                        breakpoint()
                    chromosome.crowding_distance += (delta)

    try:
        assert min([chr.crowding_distance for chr in chromosomes]) >= 0
    except:
        print(chromosomes)
        breakpoint()

    
    # sort according to (rank >> cd)
    cd_sorted_chromosomes = []
    for rank in range(1, max_rank+1):
        chromosomes_r = [chr for chr in chromosomes if chr.rank == rank]
        chromosomes_r = sorted(chromosomes_r, key=lambda x:x.crowding_distance, reverse=True)
        cd_sorted_chromosomes.extend(chromosomes_r)
        
    return cd_sorted_chromosomes