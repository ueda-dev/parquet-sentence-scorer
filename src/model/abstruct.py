from abc import ABC
from typing import *

class ModelResponse(TypedDict):
    label: str
    score: float

ModelResponses: TypeAlias = List[ModelResponse]

class ModelAbstruct(ABC):
    def __init__(self):
        pass

    def analyze(self, words:str | List[str]) -> ModelResponses:
        pass