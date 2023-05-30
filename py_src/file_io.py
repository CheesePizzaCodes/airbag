import json
from typing import Tuple, List, Dict
import numpy as np


def load_preprocessed_data(fullpath: str = None,
                           window_size_overlap_tuple: Tuple[int, float] = None) -> Tuple[np.ndarray, np.ndarray]:
    if fullpath:
        path = fullpath
    elif window_size_overlap_tuple:
        window_size, overlap = window_size_overlap_tuple
        path = f'./py_src/data/ws{window_size}_ol{overlap}--.npy'

    elif window_size_overlap_tuple and fullpath:
        print('Specify only one of the two')
        return
    else:
        print('Wrong arguments')
        return

    try:
        data = np.load(path)
    except FileNotFoundError:
        print(f'The file {path} does not exist. Try again.')
        return
    data = np.nan_to_num(data, nan=0)

    X = data[:, :-1]
    y = data[:, -1:].flatten().astype('int')

    return X, y


def append_to_json_file(file_path, data):
    """

    @param file_path:
    @param data:
    @return:
    """
    # Read existing JSON data from the file
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        json_data = []

    # Modify the JSON data by appending the new data
    json_data.append(data)

    # Write the modified JSON data back to the file
    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=4)


def load_evaluation_results(iteration) -> List[Dict]:
    path = rf'.\results\results{iteration}.json'
    with open(path, 'r') as f:
        json_data = json.load(f)
    return json_data
