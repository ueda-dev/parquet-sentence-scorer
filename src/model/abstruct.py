from abc import ABC
from typing import *

class ModelAbstruct(ABC):
    def __init__(self):
        pass

    def analyze(self, words:str) -> float:
        pass