from typing import List, Set
import pandas as pd
import ahocorasick
from concurrent.futures import ProcessPoolExecutor
import numpy as np

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
        
        Args:
            text: 検索対象テキスト
            
        Returns:
            bool: キーワードが含まれている場合True
        """
        if pd.isna(text):  # None や NaN の処理
            return False
            
        if not self.case_sensitive:
            text = str(text).lower()
        else:
            text = str(text)
            
        # イテレータを使って最初のマッチを確認
        try:
            next(self.automaton.iter(text))
            return True
        except StopIteration:
            return False

    def _find_matched_keywords(self, text: str) -> Set[str]:
        """
        テキストに含まれるキーワードを全て抽出
        
        Args:
            text: 検索対象テキスト
            
        Returns:
            Set[str]: マッチしたキーワードのセット
        """
        if pd.isna(text):
            return set()
            
        if not self.case_sensitive:
            text = str(text).lower()
        else:
            text = str(text)
            
        return {keyword for _, (_, keyword) in self.automaton.iter(text)}

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
        def process_chunk(chunk_df: pd.DataFrame) -> np.ndarray:
            # 各テキスト列に対してキーワードチェック
            column_masks = []
            for col in text_columns:
                if col in chunk_df.columns:
                    mask = chunk_df[col].fillna('').apply(self._contains_any_keyword)
                    column_masks.append(mask)
            
            # いずれかの列でマッチした行を特定
            if column_masks:
                return np.any(column_masks, axis=0)
            return np.zeros(len(chunk_df), dtype=bool)

        # データフレームを分割して並列処理
        if len(df) > chunk_size and n_jobs != 1:
            chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(process_chunk, chunks))
            
            return pd.Series(np.concatenate(results), index=df.index)
        else:
            return pd.Series(process_chunk(df), index=df.index)

    def find_matches_in_row(self, 
                          row: pd.Series, 
                          text_columns: List[str]) -> Set[str]:
        """
        行内の指定された列から全てのマッチするキーワードを抽出
        
        Args:
            row: DataFrame の1行
            text_columns: テキストを含む列名のリスト
            
        Returns:
            Set[str]: マッチしたキーワードのセット
        """
        matches = set()
        for col in text_columns:
            if col in row.index and not pd.isna(row[col]):
                matches.update(self._find_matched_keywords(str(row[col])))
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
    mask = text_filter.create_filter_mask(df, text_columns, chunk_size, n_jobs)
    
    # フィルタされたデータフレームを取得
    filtered_df = df[mask].copy()
    
    # マッチしたキーワードを追加（オプション）
    if len(filtered_df) > 0:
        filtered_df['matched_keywords'] = filtered_df.apply(
            lambda row: text_filter.find_matches_in_row(row, text_columns),
            axis=1
        )
    
    return filtered_df

# 使用例
def main():
    # パラメータ設定
    parquet_path = "path/to/your/data.parquet"
    keywords = [
        "企業名",
        "製品名",
        "サービス名",
        # ... 他のキーワード
    ]
    text_columns = ['title', 'content', 'description']  # 検索対象の列
    
    # フィルタリングの実行
    filtered_df = parquet_loader(
        parquet_path=parquet_path,
        keywords=keywords,
        text_columns=text_columns,
        case_sensitive=False,
        chunk_size=10000,
        n_jobs=-1
    )
    
    # 結果の確認
    print(f"Matched rows: {len(filtered_df)}")
    
    # 結果の保存（必要に応じて）
    filtered_df.to_parquet("filtered_results.parquet")