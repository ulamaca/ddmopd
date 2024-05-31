from .chromosome import Chromosome
from .nondominate_sort import sort_nondominate, convert_front_dict_to_df
from manipulate.mutate import Genetic_Mutations
from manipulate.cross_over import cut_inverted_cross_over
import random
import pandas as pd

'''
    NSGA2 for Peptide Optimization
        TODO: to implement 
            (1) non-dominated sorting
            (2) crowding distance (diversify the front in terms of the score)
            (3) to include sklearn
                > AMP
                > MIC
                > Hemo
                predictors as scorers
        
        TODO: bigger features
            (3) constraint handling

'''

def legalize_seq_for_clf(seq: str):
    '''
        as the title suggests
        TODO: to see why this vocab are not valid for the given vocab
    '''
    seq = seq.replace(' ', '')
    seq = seq.replace('X', '')
    seq = seq.replace('B', '')

    return seq

# default mutator
defult_mutator = Genetic_Mutations(data_path='data/cpp/cpp_predictor_dataset.csv')

class NSGA2:
    def __init__(self,                                
                scorers: dict,      
                num_generations=100,                
                population_size=1000,
                mutator = defult_mutator,
                mutate_rate=0.3,      
                crossover_f = None,                          
                crossover_rate=0.8,
                intsertion_f = None,
                retain_rate=0.2,
                insertion_rate=0.3,                                
                constraint_handle=[],   
                random_seed=56,                                        
                ) -> None:        
        
        assert len(constraint_handle) <= 0, 'constraint handle is not yet implemented'

        random.seed(random_seed)
        self.scorer_dict = scorers
        self.population_size = population_size
        self.cross_over_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.mutator = mutator
        self.num_generations = num_generations
        self.scorer_dict = scorers
        self._runs = 0 # how many runs have done
        
    
    # init population
    def init_popluation_with_a_seed(self, seed_seq = 'GLPALISWSKRKRQQ'):                
        population = [legalize_seq_for_clf(self.mutator.mutate(seed_seq)) for _ in range(self.population_size)]  
        return population
    
    def fill_population(self, front_dict: dict, verbose=True) -> list[Chromosome]:
        '''
            fill the poluation given the current sorted front and retained chromosome
        '''  
        filled_population = []
        n_to_fill = self.population_size
        
        # TODO: assume the front_dict is being sorted already!!
        
        n_fronts = len(front_dict)
        for k in range(n_fronts):
            v = front_dict[k]
            if verbose:
                print(f' collecting {k}th front')
            if len(v) <= n_to_fill:
                filled_population.extend(v)
            else:
                filled_population.extend(v[:n_to_fill])
            
            if len(filled_population) == self.population_size:
                break
            else:
                n_to_fill = self.population_size - len(filled_population)
                        
        return filled_population

    def evaluate_population(self, population: list[str]) -> list[Chromosome]:
        '''
            evaluate (multi-) scores of a given population and
            transform the format to Chromosomes
        '''
        population_chromosomes = []
        
        population_scores = {}
        for name, scorer in self.scorer_dict.items():
            population_scores[name] = scorer.predict_proba(population)[:,1]
        
        for i, pep_seq in enumerate(population):
            x = Chromosome(
                id=i,
                sequence=pep_seq,
                scores={k:v[i] for k,v in population_scores.items()}
            )

            population_chromosomes.append(x)

        # verbose
        print(' done evaluation')        
        return population_chromosomes

    def make_new_population(self, population):
        '''
            do evolutionary operations on the current populations
                1 mutation
                2 cross-over + (3 insertion)                
            according to the config
        '''        

        # cross over
        n_offsprings = int(self.population_size * self.cross_over_rate)
        n_parent_to_stay = self.population_size - n_offsprings
        
        mates = [random.sample(population, 2) for _ in range(n_offsprings)]

        new_population = []

        if n_parent_to_stay > 0:
            parents_to_stay = random.sample(population, n_parent_to_stay)            
            new_population.extend(parents_to_stay)

        if n_offsprings > 0:
            offsprings = [cut_inverted_cross_over(parent[0], parent[1]) for parent in mates]
            new_population.extend(offsprings)
        
        if self.mutate_rate > 0:
            new_population = [ self.mutator.mutate(pep_seq) if random.uniform(0,1) <= self.mutate_rate else pep_seq  for pep_seq in new_population ]

        return new_population
    
    def sort_ranked_population(self, population):
        '''
            cd: sort considering crowding distance (TODO)
        '''
        return

    @staticmethod
    def convert_generations_to_df(generations: list):
        '''
            assume generations follows order
        '''        
        rows = []
        for i, generation_i in enumerate(generations):
            for chorosome in generation_i:
                row = chorosome.__dict__
                for score_name, v in row['scores'].items():
                    row[score_name] = v
                del row['scores']
                row['generation'] = i+1
                rows.append(row)
        df = pd.DataFrame(rows)
        
        return df

    def run(self, seed_population=None, use_crowding_distance=False):        
        '''
            *_population: list of str
            
            R_chromosome: list of Chromoses

            TODO:
                to think about the best use of Chromosome representation
        '''
        # TODO: to make the population init more flexible
        parent_population = self.init_popluation_with_a_seed() if seed_population is None else seed_population                
        child_population = []
        
        generations = []
        # initial population
               
        for step in range(self.num_generations):         
            print(f'ga: runing {step+1}th generation')       
            child_population = self.make_new_population(parent_population)
            R = child_population + parent_population# R for R_t in the NSGA-II paper
            # TODO: to avoid re-evaluate evaluated ones (if needed)
            R_chrsms = self.evaluate_population(R)
            front_dict_R = sort_nondominate(R_chrsms, list(self.scorer_dict.keys())) # only rank the chromosoes                                                            
                                                
            ## where the selection happens
            population_new = self.fill_population(front_dict_R) # updated population_new                        
            if use_crowding_distance:
                # TODO: to implement sorting with adding crowding distance
                population_new = self.sort_ranked_population(population_new) 
            population_new = population_new[:self.population_size] # choose the eliltes                        
            generations.append(population_new)
            ## 
            parent_population = [chromosome.sequence for chromosome in population_new]
        
        df_trace = self.convert_generations_to_df(generations)
        return df_trace
                        

