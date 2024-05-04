'''
    dependencies
        1. pip install mapchiral
        2. pip install git+https://github.com/reymond-group/map4@v1.0
        3. pip install -i https://test.pypi.org/simple/ tmap==1.0.8
        4. python -Im pip install attrs

'''
import numpy as np
import pandas as pd
from rdkit import Chem
from pandarallel import pandarallel # this is a default!? for pandas?
pandarallel.initialize(progress_bar=False)

from map4 import MAP4Calculator
from mapchiral.mapchiral import encode

hemo_path = '/home/ulamaca/projects/10_peptide/data/dbaasp/fine_tune_hemolysis.csv'

def replace_inf_to_float_inf(str_):
    return str_.replace('inf', 'float("inf")')


# MAP4c fp (from https://github.com/reymond-group/LLM_classifier/blob/main/03_SVM.ipynb)
def jaccard_kernel(X, Y=None):
    if Y is None:
        X = Y
    js_allpairs = np.zeros((len(X),len(Y)))
    for i,fp1 in enumerate(X):
        for j,fp2 in enumerate(Y):
            js_allpairs[i,j] = np.float64(np.count_nonzero(fp1 == fp2)) / np.float64(len(fp1))
    return js_allpairs

seq2map4c_fp = lambda x: encode(Chem.MolFromSequence(x), max_radius=2, n_permutations=2048)

# MAP4 fp (from github/MLpeptide)
def seq_to_smiles(seq):
    mol = Chem.MolFromFASTA(seq, flavor=True, sanitize = True)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    return smiles

MAP4 = MAP4Calculator(dimensions=1024)
def calc_map4(smiles):
    mol = Chem.MolFromSmiles(smiles)
    map4 = MAP4.calculate(mol)
    return np.array(map4)


if __name__ == "__main__":
    ADD_MAP4C = False
    ADD_MAP4 = True


    df_hemo = pd.read_csv(hemo_path, index_col='ID')
    df_hemo['len'] = df_hemo['Sequence'].map(len)    
    df_hemo = df_hemo.query('isNotHemolytic >= 0')                
    df_hemo['map4c_fp'] = df_hemo['Sequence'].apply(seq2map4c_fp) # this took 2.5mins for my laptops; may apply parallel to speedup
    breakpoint()
    # TODO: to save map4c

    df_hemo['smiles'] = df_hemo['Sequence'].parallel_map(seq_to_smiles)
    df_hemo['map4_fp'] = df_hemo['smiles'].parallel_map(calc_map4) # this took ~8mins in my laptop
    arr = np.array(df_hemo['map4_fp'].to_list())
    np.save('dbaasp/processed_map4_fp_hemolysis.npy', arr)
    
    breakpoint()