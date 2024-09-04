from typing import NamedTuple, List


class DataPair(NamedTuple):
    """
    A class that contains all the info about the seed and obfuscated file
    """

    obfuscated_tokens: str
    seed_tokens: str
    dist_string: List[int] # (ed, # of seed, # of obf)
    dist_ast: List[int]    # (ed, # of seed, # of obf)
    dist_token: List[int]  # (ed, # of seed, # of obf)
    result: int
    obfuscated_ast: str = None
    seed_ast: str = None
