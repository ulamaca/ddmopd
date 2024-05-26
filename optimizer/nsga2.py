from .chromosome import Chromosome
from .nondominate_sort import sort_nondominate, convert_front_dict_to_df
from manipulate.mutate import Genetic_Mutations

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
                seed_population = list[Chromosome],
                population_size=1000,
                mutator = Genetic_Mutations(),
                mutate_rate=0.3,                
                crossover_rate=0.8,
                insertion_rate=0.3,                
                constraint_handle=[],                           
                ) -> None:
        pass
        
        assert len(constraint_handle) > 0, 'constraint handle is not yet implemented'

        self.scorer_dict = scorers
        self.population_size = population_size
        self.num_generations = num_generations

    
    # init population
    def init_popluation_with_a_seed(self, seed_seq = 'GLPALISWSKRKRQQ'):                
        population = [legalize_seq_for_clf(self.mutator.mutate(seed_seq)) for _ in range(self.population_size)]  
        return population
    
    def fill_population(self, front_dict: dict):
        '''
            fill the poluation given the current sorted front and retained chromosome
        '''        
        return

    def make_new_population(self, population):
        '''
            do evolutionary operations on the current populations
                1 mutation
                2 cross-over
                3 insertion
            according to the config
        '''


        return 
    
    def sort_ranked_population(self, population, cd=False):
        '''
            cd: sort considering crowding distance (TODO)
        '''

    def run(self):        
        population_old = self.init_popluation_with_a_seed()
        population_new = self.init_popluation_with_a_seed()
        
        generations = []
        # initial population

        # TODO: to think more about handle population_new and population_old
        for step in range(self.num_generations):                
            R = population_old + population_new # R for R_t in the NSGA-II paper
            front_dict_R = sort_nondominate(R) # only rank the chromosoes
            population_new = self.fill_population(front_dict_R) # updated population_new
            population_new = self.sort_ranked_population(population_new)
            population_new = population_new[:self.population_size] # choose the eliltes
            population_new = self.make_new_population(population_new)
            #population_old = population_new

