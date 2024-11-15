from tqdm import tqdm
from glob import glob
import os
import csv
import pandas as pd
from tkinter import filedialog
from model.concrete import Model
from loader import parquet_loader
from text_compressor import FrequencyBasedCompressor
from typing import List

#iteration-options------------------------
TEST_MODE = False

#loader-configs---------------------------
TARGET_COLUMNS = [
    'text'
]
CASE_SENSITIVE = False
CHUNK_SIZE = 10000
USE_N_CPU_CORES = 4
#-----------------------------------------

def build(inputFileName: str, outputFileName: str, model:Model, compressor:FrequencyBasedCompressor, keywords:List[str]) -> None:
    input_df = parquet_loader(inputFileName, keywords, TARGET_COLUMNS, CASE_SENSITIVE, CHUNK_SIZE, USE_N_CPU_CORES)

    input_df['text'] = input_df['text'].apply(compressor.compress)
    modelResponses = model.analyze(input_df['text'].tolist())
    output_df = pd.DataFrame({
        'timestamp': input_df['timestamp'].tolist(),
        'label': map(lambda x:x['label'], modelResponses),
        'score': map(lambda x:x['score'], modelResponses),
        'matched_keywords': input_df['matched_keywords'].tolist()
    })
    output_df.to_csv(outputFileName)

def main():
    #インスタンス初期化-------------------------------
    model = Model()
    text_compressor = FrequencyBasedCompressor()

    #データ入力元を選択-------------------------------
    readDirName = filedialog.askdirectory(title='読み込むディレクトリを選択')
    if not readDirName or not os.path.exists(readDirName):
        print('ERROR: 有効なディレクトリが選択されませんでした。')
        input('Press any key to close...')
        return

    #データ出力先を選択--------------------------------
    writeDirName = filedialog.askdirectory(title='書き込むディレクトリを選択')
    if not writeDirName or not os.path.exists(writeDirName):
        print('ERROR: 有効なディレクトリが選択されませんでした。')
        input('Press any key to close...')
        return

    #検索キーワードを格納したCSVを選択
    txtFileName = filedialog.askopenfilename(title='検索キーワードを格納したテキストファイルを選択してください', filetypes=[('datafile', '*.txt')])
    with open(txtFileName, 'r', encoding='utf-8') as f:
        keywords = f.readlines()

    #読み込み対象のファイルを検出 & 出力先を割り当て-----
    targets = [readDirName + '/' + x for x in glob('*.parquet', root_dir=readDirName)]
    exports = [writeDirName + '/cc_semtiment_' + str(i+1).zfill(3) + '.csv' for i in range(len(targets))]
    
    #処理開始前最終確認--------------------------------
    confirminationTexts = [
        '<CONFIRMINATION>',
        f'Test-Mode: {TEST_MODE}',
        f'InputDir : {readDirName}',
        f'OutputDir : {writeDirName}',
        f'detected {len(targets)} files in InputDir',
        'would you lile to continue?'
    ]
    for t in confirminationTexts:
        print(t)

    if not input('(y/n)>') == 'y':
        print('cancelled process')
        input('Press any key to close...')
        return

    #処理開始-----------------------------------------
    for target, export in tqdm(zip(targets, exports), 'データセット構築中', len(targets)):
        build(target, export, model, text_compressor, keywords)

        if TEST_MODE: break

    print('finished process')
    input('Press any key to close...')

if __name__ == '__main__':
    main()