import os
import time
import logging

from joblib import Parallel, delayed
from numpy import arange

from preprocess import main as preprocess_main
from model_evaluation import ClassifierFamily, run_cross_validation
from file_io import load_preprocessed_data

log_format = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s'
logging.basicConfig(filename='./logs/example.log', encoding='utf-8', level=logging.DEBUG, format=log_format)
def preprocessing():
    win_sizes = arange(10, 30, 5)
    overlaps = arange(0.1, 1.0, 0.1)
    results = Parallel(n_jobs=7)(delayed(preprocess_main)(win_size, overlap)
                                 for win_size in win_sizes for overlap in overlaps)


def evaluation():
    classifier_names = ClassifierFamily(verbose=True).model_names
    data_dir = r'./py_src/data'


    files_collection = list(f'{data_dir}/{file}' for file in os.listdir(data_dir)
                            if
                            not any(i in file for i in ('_ol0.1', '_ol0.2', '_ol0.3'))
                            and
                            os.path.isfile(f'{data_dir}/{file}'))[::-1]

    data_collection = {file: load_preprocessed_data(file) for file in files_collection}

    results = Parallel(n_jobs=7)(delayed(run_cross_validation)(c_name, data, data_name)
                                 for c_name in classifier_names for data_name, data in data_collection.items())


if __name__ == '__main__':
    print('STARTING PARALLEL COMPUTING')
    res = evaluation()
    print(res)

