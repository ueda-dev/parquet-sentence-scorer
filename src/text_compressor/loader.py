from typing import DefaultDict
from collections import defaultdict
from glob import glob
import json
import os
from tqdm import tqdm

def load_word_dict() -> DefaultDict[str, int]:
    targets = glob('*.json', root_dir=os.path.dirname(__file__) + '/dependencies')
    init_dict = defaultdict(lambda:0)
    for t in tqdm(targets, 'loading word-dict'):
        with open(t, 'r', encoding='utf-8') as f:
            init_dict.update(**json.load(f))

    return init_dict
