from dataclasses import dataclass

# each peptide as a Chromosome in GA
@dataclass
class Chromosome:
    id: int
    sequence: str
    scores: dict
    rank: int = -1 # default as -1 (non-ranked)
    crowding_distance: float = None
    constraints: dict = None