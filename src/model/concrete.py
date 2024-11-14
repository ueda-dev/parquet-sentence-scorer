from model.abstruct import ModelAbstruct, ModelResponses
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class Model(ModelAbstruct):
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
        model = AutoModelForSequenceClassification.from_pretrained("koheiduck/bert-japanese-finetuned-sentiment")
        self._classfier = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

    def analyze(self, words) -> ModelResponses:
        return self._classfier(words)