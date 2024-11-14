from model.abstruct import ModelAbstruct
from transformers import pipeline

class Model(ModelAbstruct):
    def __init__(self):
        self._classifer = pipeline("sentiment-analysis", model="koheiduck/bert-japanese-finetuned-sentiment")

    def analyze(self, words):
        return self._classifer(words)