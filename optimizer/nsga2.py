from .chromosome import Chromosome
from .nondominate_sort import sort_nondominate, convert_front_dict_to_df
from manipulate.mutate import Genetic_Mutations
from manipulate.cross_over import cut_inverted_cross_over
import random

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



class NSGA2:
    def __init__(self,                
                crossover_f,
                intsertion_f,
                scorers: dict,      
                num_generations=100,
                seed_population = list[str],
                population_size=1000,
                mutator = Genetic_Mutations(),
                mutate_rate=0.3,                
                crossover_rate=0.8,
                retain_rate=0.2,
                insertion_rate=0.3,                
                constraint_handle=[],   
                random_seed=56,                                        
                ) -> None:        
        
        assert len(constraint_handle) > 0, 'constraint handle is not yet implemented'

        random.seed(random_seed)
        self.scorer_dict = scorers
        self.population_size = population_size
        self.cross_over_rate = crossover_rate
        self.mutate_rate = mutate_rate
        self.mutator = mutator
        self.num_generations = num_generations
        self.scorer_dict = scorers
        
    
    # init population
    def init_popluation_with_a_seed(self, seed_seq = 'GLPALISWSKRKRQQ'):                
        population = [legalize_seq_for_clf(self.mutator.mutate(seed_seq)) for _ in range(self.population_size)]  
        return population
    
    def fill_population(self, front_dict: dict):
        '''
            fill the poluation given the current sorted front and retained chromosome
        '''        
        return

    def evaluate_population(self, population: list[str]) -> list[Chromosome]:
        '''
            evaluate (multi-) scores of a given population and
            transform the format to Chromosomes
        '''
        population_chromosomes = []

        for i, pep_seq in enumerate(population):
            x = Chromosome(
                id=i,
                sequence=pep_seq,
                scores={k:v(x) for k,v in self.scorer_dict.items()}
            )

            population_chromosomes.append(x)

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
            new_population = [ self.mutator.mutate(pep_seq) if random.uniform(0,1) <= self.mutation_rate else pep_seq  for pep_seq in new_population ]

        return new_population
    
    def sort_ranked_population(self, population, cd=False):
        '''
            cd: sort considering crowding distance (TODO)
        '''

    def run(self):        
        '''
            *_population: list of str
            
            R_chromosome: list of Chromoses

            TODO:
                to think about the best use of Chromosome representation
        '''
        # TODO: to make the population init more flexible
        parent_population = self.init_popluation_with_a_seed()
        child_population = []
        
        generations = []
        # initial population
               
        for step in range(self.num_generations):                
            child_population = self.make_new_population(parent_population)
            R = child_population + population_new # R for R_t in the NSGA-II paper
            R_chromosomes = self.evaluate_population(R)
            front_dict_R = sort_nondominate(R_chromosomes) # only rank the chromosoes                                                
            
            # TODO: to complete the following methods
            population_new = self.fill_population(front_dict_R) # updated population_new
            population_new = self.sort_ranked_population(population_new)
            population_new = population_new[:self.population_size] # choose the eliltes            
            parent_population = population_new
                        

