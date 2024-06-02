'''
    naive_cross_over: 
    inverted_cross_over: 
        cross_over with inversion process involved
            v0: using random_segment, which may take a random contiguous portion of each sequence, to determine genes to pass
            v1: using cut_segment, cut half of the sequence, to determine genes to pass

    TODO: to decouple inversion operation    

'''

import random

# segment utils
def random_segment(seq: list, seg_size):
    '''
        segment out size=seg_size subseq from seq
    '''
    l = len(seq)
    segment_start = random.randint(0, l-seg_size)

    return seq[segment_start:segment_start+seg_size]

def cut_segment(seq, cut_pt, how='to'):
    segmented = None
    if how == 'to':
        segmented = seq[:cut_pt]
    
    elif how == 'from':
        segmented = seq[cut_pt:]
    else:
        raise ValueError

    return segmented

# naive
def naive_cross_over(seq_1, seq_2, short_tolerance=3):
    '''
        seq_1: str
        seq_2: str
        short_tolerance: int
            how much #char shorter could one tolerate after cross over

        a naive way to cross-over two seq
    
    '''
    l_1 = len(seq_1)
    l_2 = len(seq_2)

    L = min(l_1, l_2)
    
    segment_pos = random.randint(1,L-2) # [<start>,x,y,z...,<end>], the seg should not be start nor end
    final_len = random.randint(L-short_tolerance, l_1+l_2)
    

    if random.uniform(0,1) < 0.5:
        seq = cut_segment(seq_1, segment_pos, 'to') + cut_segment(seq_2, segment_pos, 'from')
    else:
        seq = cut_segment(seq_2, segment_pos, 'to')  + cut_segment(seq_1, segment_pos, 'from')

    return seq[:final_len]

# inverted_cross_over

def invert_v1(seq, ratio=0.5, min_rev=2, verbose=False):
    '''
        invert a sequence with at max_len = len(seq)*ratio
        invert its 'pseudo-order'
    '''
    assert ratio > 0.0 and ratio <= 1    
    pseudo_order = [i for i in range(len(seq))]
    
    inversion_size = random.randint(min_rev, int(len(seq)*ratio))    
    inversion_start = random.randint(0, len(seq)-inversion_size)
    
    pseudo_order = pseudo_order[:inversion_start] + pseudo_order[inversion_start:inversion_start+inversion_size][::-1] + pseudo_order[inversion_start+inversion_size:]
    
    if verbose:
        print('inversion_size', inversion_size)
        print('inversion_start', inversion_start)
        print('pseudo_order', pseudo_order)

    return pseudo_order

def random_segment_inverted(seq, pseudo_order, seg_size=None):    
    '''
        segment inverted (pseudo_order is inverted) sequences 
    '''

    if len(seq) <=2:
        f'warning: {seq}-s len leq 3 >> will not be split'
        split_seq = seq
    else:
        '''
            using pseudo-order to split 
            then resort the seq-order after split
        '''
        order2char = {i:k for i,k in enumerate(seq)}
        
        if seg_size is None:
            l = len(seq)        
            seg_size = random.randint(1, l)        

        split_seq_orders = random_segment(pseudo_order, seg_size)

        split_seq_orders = sorted(split_seq_orders)             
        split_seq_chars = [order2char[i] for i in split_seq_orders]                
        split_seq = ''.join(split_seq_chars)

    return split_seq

def cut_segment_inverted(seq, pseudo_order, seg_pos=None, how='to'):    
    '''
        segment inverted (pseudo_order is inverted) sequences 
        using cut_segment
    '''

    if len(seq) <=2:
        f'warning: {seq}-s len leq 3 >> will not be split'
        split_seq = seq
    else:
        '''
            using pseudo-order to split 
            then resort the seq-order after split
        '''
        order2char = {i:k for i,k in enumerate(seq)}
        l = len(seq)

                
        if seg_pos is None:
            seg_pos = random.randint(1,l-2)

        split_seq_orders = cut_segment(pseudo_order, seg_pos, how=how)
        split_seq_orders = sorted(split_seq_orders)             
        split_seq_chars = [order2char[i] for i in split_seq_orders]                
        split_seq = ''.join(split_seq_chars)

    return split_seq

## v1 inverted cross_over
def cut_inverted_cross_over(seq_1, seq_2, seg_size=None, verbose=False, inv_ratio=0.5, min_rev=2, prob_inv=0.3):
    '''
        inverted a pseudo order of the seq and then do cross over,
        this would make some substring easier to be preserved

        using cut_segment, cut half of the sequence, to determine genes to pass

    '''    
    l1 = len(seq_1)
    l2 = len(seq_2)


    if random.uniform(0,1) < prob_inv:
        pseudo_order_1 = invert_v1(seq_1, ratio=inv_ratio, min_rev=min_rev)
        pseudo_order_2 = invert_v1(seq_2, ratio=inv_ratio, min_rev=min_rev)
    
    else:
        pseudo_order_1 = [i for i in range(l1)]
        pseudo_order_2 = [i for i in range(l2)]    

    cut_1 = l1//2
    cut_2 = l2//2

    # succed from half of each parent string
    if random.uniform(0,1) < 0.5:        
        cv_seq_f = cut_segment_inverted(seq_1, pseudo_order_1, seg_pos=cut_1, how='to')
        cv_seq_m = cut_segment_inverted(seq_2, pseudo_order_2, seg_pos=cut_2, how='from')
        
    else:
        cv_seq_f = cut_segment_inverted(seq_2, pseudo_order_2, seg_pos=cut_2, how='to')
        cv_seq_m = cut_segment_inverted(seq_1, pseudo_order_1, seg_pos=cut_1, how='from')                
    
    cv_seq = cv_seq_f + cv_seq_m    

    return cv_seq
    

## v0 inverted cross_over
def random_inverted_cross_over(seq_1, seq_2, seg_size=None, verbose=False, inv_ratio=0.5, min_rev=2, prob_inv=0.3, short_tolerance=3):
    '''
        inverted a pseudo order of the seq and then do cross over,
        this would make some substring easier to be preserved

        using random_segment, which may take a random contiguous portion of each sequence, to determine genes to pass
    '''    
    l1 = len(seq_1)
    l2 = len(seq_2)


    if random.uniform(0,1) < prob_inv:
        pseudo_order_1 = invert_v1(seq_1, ratio=inv_ratio, min_rev=min_rev)
        pseudo_order_2 = invert_v1(seq_2, ratio=inv_ratio, min_rev=min_rev)
    
    else:
        pseudo_order_1 = [i for i in range(l1)]
        pseudo_order_2 = [i for i in range(l2)]

    cv_seq_f = random_segment_inverted(seq_1, pseudo_order_1, seg_size=seg_size)
    cv_seq_m = random_segment_inverted(seq_2, pseudo_order_2, seg_size=seg_size)

    if random.uniform(0,1) < 0.5:
        cv_seq = cv_seq_f + cv_seq_m
    else:
        cv_seq = cv_seq_m + cv_seq_f
    
    # len control
    l0 = min(l1, l2)
    final_len = random.randint(l0-short_tolerance, l1+l2)

    return cv_seq[:final_len]
##

if __name__ == "__main__":
    seed_seq = 'TKPRPGP' # peptide: Selank
    seq_2 = 'KPKYPRIRLSSRG' # a searched peptide

    tmp = invert_v1(seed_seq, verbose=True)
    tmp_2 = random_segment_inverted(seed_seq, tmp)

    print('before: ', seed_seq)
    print('after: ', tmp_2)

    seq_3 = cut_inverted_cross_over(seed_seq, seq_2)
    seq_4 = random_inverted_cross_over(seed_seq, seq_2)
    breakpoint()
