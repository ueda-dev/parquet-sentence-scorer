import json
import os
from typing import List, Tuple
import MeCab
import ipadic
from text_compressor.loader import load_word_dict

class FrequencyBasedCompressor:
    def __init__(self, max_length: int = 512):
        """
        Parameters:
        -----------
        word_dict_path : str
            単語の出現頻度辞書のJSONファイルパス
        max_length : int
            圧縮後の最大文字数
        """
        self.max_length = max_length
        self.tokenizer = MeCab.Tagger(f"{ipadic.MECAB_ARGS} -Owakati")
        
        # 単語出現頻度辞書の読み込み
        self.word_dict = load_word_dict()
    
    def compress(self, text: str) -> str:
        """
        テキストを圧縮します。
        
        Parameters:
        -----------
        text : str
            圧縮対象のテキスト
            
        Returns:
        --------
        str
            圧縮後のテキスト
        """
        # 文字数が上限以下なら圧縮しない
        if len(text) <= self.max_length:
            return text
            
        # テキストを文章単位に分割
        sentences = self._split_text(text)
        
        while True:
            # 各文のスコアを計算
            sentence_scores = []
            for sentence in sentences:
                score = self._calculate_sentence_score(sentence)
                sentence_scores.append((sentence, score))
            
            # スコアの低い順にソート
            sentence_scores.sort(key=lambda x: x[1])
            
            # 最もスコアの低い文を除外
            sentences = [s[0] for s in sentence_scores[1:]]
            
            # 結合してテキストを再構成
            compressed_text = self._join_sentences(sentences)
            
            # 文字数が上限以下になったら終了
            if len(compressed_text) <= self.max_length:
                return compressed_text
    
    def _split_text(self, text: str) -> List[str]:
        """
        テキストを文章単位に分割します。
        """
        # 。と\nで分割し、空文字を除外
        sentences = []
        for line in text.split('\n'):
            sentences.extend([s.strip() for s in line.split('。') if s.strip()])
        return sentences
    
    def _calculate_sentence_score(self, sentence: str) -> float:
        """
        文章のスコアを計算します。
        単語の出現頻度の平均値を返します。
        """
        # 文章をトークナイズ
        tokens = self.tokenizer.parse(sentence).strip().split()
        
        if not tokens:
            return 0
        
        # 各単語のスコアを取得し平均を計算
        scores = [self.word_dict.get(token, 0) for token in tokens]
        return sum(scores) / len(tokens)
    
    def _join_sentences(self, sentences: List[str]) -> str:
        """
        文章のリストを1つのテキストに結合します。
        """
        return '。'.join(sentences) + ('。' if sentences else '')

