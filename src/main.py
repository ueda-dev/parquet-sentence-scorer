from tqdm import tqdm
from glob import glob
import os
import pandas as pd
from tkinter import filedialog
from logging import Logger
from model.concrete import Model
from loader import parquet_loader

#loader-configs---------------------------
TARGET_WORDS = [
    'spacex',
    'tesla',
    'eron',
    'musk'
]
TARGET_COLUMNS = [
    'text'
]
CASE_SENSITIVE = False
CHUNK_SIZE = 10000
USE_N_CPU_CORES = -1
#-----------------------------------------

def build(inputFileName: str, outputFileName: str, model:Model) -> None:
    input_df = parquet_loader(inputFileName, TARGET_WORDS, TARGET_COLUMNS, CASE_SENSITIVE, CHUNK_SIZE, USE_N_CPU_CORES)

    modelResponses = model.analyze(input_df['text'])
    output_df = pd.DataFrame({
        'timestamp': input_df['timestamp'].tolist(),
        'label': map(lambda x:x['label'], modelResponses),
        'score': map(lambda x:x['score'], modelResponses)
    })
    output_df.to_csv(outputFileName)

def main():
    #インスタンス初期化-------------------------------
    logger = Logger('builder')
    model = Model()

    #データ入力元を選択-------------------------------
    logger.info('読み込むデータ(parquet)を格納したディレクトリを選択してください')
    readDirName = filedialog.askdirectory()
    if not readDirName or not os.path.exists(readDirName):
        logger.error('有効なディレクトリが選択されませんでした。')
        input('Press any key to close...')
        return

    #データ出力先を選択--------------------------------
    logger.log('データを出力するディレクトリを選択してください')
    writeDirName = filedialog.askdirectory()
    if not writeDirName or not os.path.exists(writeDirName):
        logger.error('有効なディレクトリが選択されませんでした。')
        input('Press any key to close...')
        return

    #読み込み対象のファイルを検出 & 出力先を割り当て-----
    targets = glob('*.parquet', root_dir=readDirName)
    exports = [writeDirName + '/cc_semtiment_' + str(i+1).zfill(3) + '.csv' for i in range(len(targets))]
    
    #処理開始前最終確認--------------------------------
    confirminationTexts = [
        f'InputDir : {readDirName}',
        f'OutputDir : {writeDirName}',
        f'detected {len(targets)} files in InputDir',
        'would you lile to continue?'
    ]
    for t in confirminationTexts:
        logger.info(t)

    if not input('(y/n)>') == 'y':
        logger.log('cancelled process')
        input('Press any key to close...')
        return

    #処理開始-----------------------------------------
    for target, export in tqdm(zip(targets, exports), 'データセット構築中', len(targets)):
        build(target, export, model)

    logger.info('finished process')
    input('Press any key to close...')

if __name__ == '__main__':
    main()