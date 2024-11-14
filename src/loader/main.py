from typing import List, Set
import pandas as pd
import ahocorasick
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial

class TextFilter:
    def __init__(self, keywords: List[str], case_sensitive: bool = False):
        """
        キーワードリストでフィルタを初期化
        
        Args:
            keywords: 検索キーワードのリスト
            case_sensitive: 大文字小文字を区別するかどうか
        """
        self.automaton = ahocorasick.Automaton()
        self.case_sensitive = case_sensitive
        
        # キーワードの登録
        for idx, keyword in enumerate(keywords):
            if not case_sensitive:
                keyword = keyword.lower()
            self.automaton.add_word(keyword, (idx, keyword))
        
        self.automaton.make_automaton()
        self.keywords = set(keywords)

    def _contains_any_keyword(self, text: str) -> bool:
        """
        テキストに任意のキーワードが含まれているかチェック
        """
        if pd.isna(text):
            return False
            
        if not self.case_sensitive:
            text = str(text).lower()
        else:
            text = str(text)
            
        try:
            next(self.automaton.iter(text))
            return True
        except StopIteration:
            return False

    def _process_chunk(self, chunk_data: tuple) -> np.ndarray:
        """
        データチャンクを処理するクラスメソッド
        
        Args:
            chunk_data: (chunk_df, text_columns)のタプル
            
        Returns:
            np.ndarray: ブールマスク
        """
        chunk_df, text_columns = chunk_data
        column_masks = []
        
        for col in text_columns:
            if col in chunk_df.columns:
                mask = chunk_df[col].fillna('').apply(self._contains_any_keyword)
                column_masks.append(mask)
        
        if column_masks:
            return np.any(column_masks, axis=0)
        return np.zeros(len(chunk_df), dtype=bool)

    def create_filter_mask(self, 
                         df: pd.DataFrame, 
                         text_columns: List[str],
                         chunk_size: int = 10000,
                         n_jobs: int = -1) -> pd.Series:
        """
        データフレームの指定された列に対してフィルタマスクを作成
        
        Args:
            df: 検索対象のDataFrame
            text_columns: テキストを含む列名のリスト
            chunk_size: 並列処理時のチャンクサイズ
            n_jobs: 並列処理数（-1で全CPU使用）
            
        Returns:
            pd.Series: ブールマスク（キーワードを含む行がTrue）
        """
        # 小さなデータセットまたは単一プロセスの場合
        if len(df) <= chunk_size or n_jobs == 1:
            return pd.Series(self._process_chunk((df, text_columns)), index=df.index)
        
        # データフレームをチャンクに分割
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        chunk_data = [(chunk, text_columns) for chunk in chunks]
        
        # 並列処理の実行
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(self._process_chunk, chunk_data))
        
        # 結果の結合
        return pd.Series(np.concatenate(results), index=df.index)

    def find_matches_in_row(self, 
                          row: pd.Series, 
                          text_columns: List[str]) -> Set[str]:
        """
        行内の指定された列から全てのマッチするキーワードを抽出
        """
        matches = set()
        for col in text_columns:
            if col in row.index and not pd.isna(row[col]):
                text = str(row[col])
                if not self.case_sensitive:
                    text = text.lower()
                matches.update(keyword for _, (_, keyword) in self.automaton.iter(text))
        return matches

def parquet_loader(parquet_path: str,
                       keywords: List[str],
                       text_columns: List[str],
                       case_sensitive: bool = False,
                       chunk_size: int = 10000,
                       n_jobs: int = -1) -> pd.DataFrame:
    """
    Parquetファイルから指定されたキーワードを含む行を抽出
    
    Args:
        parquet_path: Parquetファイルのパス
        keywords: 検索キーワードのリスト
        text_columns: テキスト検索対象の列名リスト
        case_sensitive: 大文字小文字を区別するかどうか
        chunk_size: 並列処理時のチャンクサイズ
        n_jobs: 並列処理数（-1で全CPU使用）
        
    Returns:
        pd.DataFrame: フィルタされたデータフレーム
    """
    # フィルタの初期化
    text_filter = TextFilter(keywords, case_sensitive=case_sensitive)
    
    # Parquetファイルの読み込み
    df = pd.read_parquet(parquet_path)
    
    # フィルタマスクの作成
    mask = text_filter.create_filter_mask(
        df=df,
        text_columns=text_columns,
        chunk_size=chunk_size,
        n_jobs=n_jobs
    )
    
    # フィルタされたデータフレームを取得
    filtered_df = df[mask].copy()
    
    # マッチしたキーワードを追加
    if len(filtered_df) > 0:
        filtered_df['matched_keywords'] = filtered_df.apply(
            lambda row: text_filter.find_matches_in_row(row, text_columns),
            axis=1
        )
    
    return filtered_df

# 使用例
if __name__ == "__main__":
    # パラメータ設定
    parquet_path = "path/to/your/data.parquet"
    keywords = [
        "企業名",
        "製品名",
        "サービス名",
    ]
    text_columns = ['title', 'content', 'description']
    
    # フィルタリングの実行
    filtered_df = filter_parquet_file(
        parquet_path=parquet_path,
        keywords=keywords,
        text_columns=text_columns,
        case_sensitive=False,
        chunk_size=10000,
        n_jobs=-1
    )
    
    # 結果の確認
    print(f"Matched rows: {len(filtered_df)}")
    filtered_df.to_parquet("filtered_results.parquet")