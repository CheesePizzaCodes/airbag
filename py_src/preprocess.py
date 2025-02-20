"""
1. Load file / signal
2. Pad signal if necessary
3. Extract Spectrogram
4. Normalize Spectrogram
5. Save normalized Spectrogram
"""
import os
import re
import copy
from typing import Tuple, Dict, List
import time

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

from normalization_strategies import NormalizationStrategy, MinMax, MeanSdev
from loading_strategies import LoadingStrategy, FromCase, FromPath
from graphics import Visualizer
from util import extract_marks


class Loader:
    """
    Responsible for loading single files
    thin wrapper over pandas
    """

    def __init__(self, strategy: LoadingStrategy = FromCase()) -> None:

        """
        base_path: relative path to the dataset folder
        """
        self._strategy = strategy

    @property
    def strategy(self) -> LoadingStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: LoadingStrategy) -> None:
        self._strategy = strategy

    def load_sensor_data(self, *args, **kwargs) -> pd.DataFrame | None:
        try:
            data, metadata = self._strategy.load_sensor_data(*args, **kwargs)
            return data, metadata
        except TypeError:
            # print('Data collection failed.')
            return None, None

    def load_label_data(self, *args) -> pd.DataFrame | None:
        """
        Loads the label (ground truth, onset and impact or NONE if regular activity) data for a given subject
        """
        try:
            data = self._strategy.load_label_data(*args)
            return data
        except TypeError:
            # print('Data collection failed.')
            return None


class Splitter():
    """
    Handles splitting data into test, train and validation set
    thin wrapper over sklearn
    """

    def __init__(self, train: float, validate: float) -> None:
        self.train = train
        self.validate = train + validate

    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        n = len(data)

        train_df = data[0:int(n * self.train)]
        val_df = data[int(n * self.train):int(n * self.validate)]
        test_df = data[int(n * self.validate):]

        return {'train': train_df, 'validation': val_df, 'test': test_df}


class Normalizer:
    """
    Applies normalization to an array
    """

    def __init__(self, strategy: NormalizationStrategy):
        self._strategy = strategy

    @property
    def strategy(self) -> NormalizationStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: NormalizationStrategy) -> None:
        self._strategy = strategy

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = copy.deepcopy(df)
        norm_array = self._strategy.normalize(df_copy)  # perform on copy to not modify data in the original address
        return norm_array

    # def denormalize(self, norm_array, original_min, original_max):
    #     array = (norm_array - self.min) / (self.max - self.min)
    #     array = array * (original_max - original_min) + original_min
    #     return array


class WindowFeatureExtractor:
    def __init__(self, window_size: int, overlap: float) -> None:
        self.window_size = window_size
        self.overlap = overlap
        self.features = None

    def declare_feature_callables(self) -> Dict[str, callable]:
        n = self.window_size
        attrs = {
            'mean': lambda w: w.mean(),
            'std': lambda w: w.std(),
            'skew': lambda w: w.skew(),
        }
        indices = [0, n // 4, n // 2, (3 * n) // 4, -1]
        for index in indices:
            attrs[f'pos_{index}'] = lambda w, index_=index: w.iloc[index_]

        return attrs

    def extract_feature_vec_from_window(self, window: pd.DataFrame) -> np.ndarray:

        attrs = self.features
        fv = []  # initialize feature vector
        for attr_name, callable_ in attrs.items():
            fv.extend(callable_(window).tolist())
        return np.array(fv).flatten()

    def extract_features(self, time_series: pd.DataFrame) -> np.ndarray:
        time_series = time_series.iloc[:, 3:]  # remove meaningless data (index, time, frame, etc.)
        stride = int(self.window_size * self.overlap)
        r = np.arange(len(time_series))
        s = r[::stride]
        n = len(time_series)
        z = zip(s, s + self.window_size)
        st = time.time()
        feature_matrix = []

        self.features = self.declare_feature_callables()

        for start, end in z:  # loop over every window
            window = time_series[start: end]
            try:
                feature_matrix.append(self.extract_feature_vec_from_window(window))
            except IndexError:
                print('Window is truncated. Finished generating feature matrix')
            print(f'{start / n:.3%}', end='\r')
        print('100.000%')
        print(f'Total time: {time.time() - st:.2f}')
        return np.asarray(feature_matrix)


class PreProcessingPipeline:
    """
    Handles all batch operations
    Processes all files in a directory or a live signal, applying the steps to each file or the signal:
    1. Load file / signal
    2. Pad signal if necessary
    3. Extract Spectrogram
    4. Normalize Spectrogram
    5. Save normalized Spectrogram

    For each input passed, the min and max values need to be stored as attributes, for reverse normalization to work.
    """

    def __init__(self, ):
        self.loader: Loader = Loader(strategy=FromCase())
        self.splitter: Splitter = Splitter(0.7, 0.2)
        self.normalizer: Normalizer = None
        self.wfe: WindowFeatureExtractor = None

    def load_raw_data(self, filename):
        return np.load(filename, allow_pickle=True)

    def batch_load(self) -> Tuple[pd.DataFrame, int]:
        """
        Uses a loader object to load all files in a directory and labels them
        """
        SUBJECT_COUNT = 36  # 6 - 36
        TASK_COUNT = 34  # 34
        RUN_COUNT = 5  # 5
        falls = []
        regular_activities = []

        for subject in range(6, SUBJECT_COUNT + 1):
            print(f'loading subject {subject} data')
            label_data = self.loader.load_label_data(subject)
            if label_data is None:
                print(f'failed to collect data for user {subject}')
                continue
            for task in range(1, TASK_COUNT + 1):
                for run in range(1, RUN_COUNT + 1):
                    output = self.loader.load_sensor_data(subject, task,
                                                          run)  # Load accelerometer and gyro data for a given task and run
                    if output is None:
                        continue
                    sensor_data, metadata = output  # metadata for debug

                    marks = extract_marks(label_data=label_data, task=task,
                                          run=run)  # find out onset and impact frames if exist
                    if marks is None:
                        regular_activities.append(sensor_data)
                    else:
                        onset_frame, impact_frame = marks
                        falls.append(sensor_data[onset_frame: impact_frame])  # add only the falling portion to the data
        falls_df = pd.concat(falls, axis=0)  # join='inner'
        falls_df.reset_index(inplace=True)
        regular_activities_df = pd.concat(regular_activities, axis=0)
        regular_activities_df.reset_index(inplace=True)
        # transform dataframes
        print('finished loading data')
        return pd.concat([falls_df, regular_activities_df]), len(falls_df.index)

    def split_data(self):  # TODO refactor later
        fall_df, normal_df = self.batch_load()
        fall, normal = map(self.splitter.split, (fall_df, normal_df))
        return fall, normal

    def batch_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        self.normalizer = Normalizer(strategy=MinMax(0, 1))
        return self.normalizer.normalize(df)

    def batch_extract_features(self, data: pd.DataFrame, split: int) -> np.ndarray:
        # Falls, i.e. class 1
        print('Beginning Feature Extraction for class 1:')
        data_1 = self.wfe.extract_features(data[:split])
        data_1 = np.hstack([data_1, np.ones(shape=(data_1.shape[0], 1))])
        # Regular activities, class 0
        print('Beginning Feature Extraction for class 0:')
        data_0 = self.wfe.extract_features(data[split:])
        data_0 = np.hstack([data_0, np.zeros(shape=(data_0.shape[0], 1))])

        data = np.vstack([data_1, data_0])
        return data

    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f'Processed file {file_path}')
        self.saver.save_min_max_values(self.original_min_max)

    def _process_file(self, file_path):
        """
        Bulk of this class
        :return:
        """
        signal, samplerate = self.loader.load_file(file_path)
        if self._padding_necessary(signal):
            signal = self._apply_padding(signal)

        feature = self.extractor.extract(signal)
        norm_feature = self.normalizer.normalize(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max(save_path, feature.min(), feature.max())

def read_and_save_raw_data(filename):
    pppl = PreProcessingPipeline()
    pppl.loader = Loader(strategy=FromCase())
    data, split = pppl.batch_load()
    np.save(filename, np.array([data, split], dtype='object'), allow_pickle=True)


def batch_preprocess(window_size: int = 10, overlap: float = 0.5, data_loc=str):
    # Configure objects
    pppl = PreProcessingPipeline()
    pppl.normalizer = Normalizer(strategy=MinMax(0, 1))
    pppl.wfe = WindowFeatureExtractor(window_size=window_size, overlap=overlap)
    # load data

    FILENAME = r'./py_src/preprocessed_data/preprocessed.npy'
    data, split = pppl.load_raw_data(FILENAME)

    # normalize data
    data = pppl.batch_normalize(data)
    # extract features
    data = pppl.batch_extract_features(data, split)
    return data


def main(window_size, overlap):
    print(f'Beginning processing of:\nWindow size: {window_size}\nOverlap: {overlap}')
    data = batch_preprocess(window_size, overlap)
    print(f'Dataframe: {data}')

    custom_text = ''
    # custom_text = input('Add hint to distinguish output: ')
    filename = f'./py_src/data/ws{window_size}_ol{overlap}-{custom_text}-.npy'
    np.save(filename, data)
    print(f'saved to {filename}')


def test_1():
    SUBJECT = 10
    TASK = 23
    RUN = 1

    # Load data
    loader = Loader(strategy=FromCase())

    sensor_data, metadata = loader.load_sensor_data(SUBJECT, TASK, RUN)
    label_data = loader.load_label_data(SUBJECT)

    # Normalize data
    normalizer = Normalizer(strategy=MinMax())

    normalized_data = normalizer.normalize(sensor_data)

    # Visualize data
    visualizer1 = Visualizer(sensor_data, metadata, label_data)
    visualizer1.generate_graphs()

    visualizer2 = Visualizer(normalized_data, metadata, label_data)
    visualizer2.generate_graphs()

    input()


if __name__ == '__main__':
    # test
    FILENAME = r'./py_src/preprocessed_data/preprocessed.npy'
    print(f'Working file: {FILENAME}')
    WINDOW_SIZE = 20
    OVERLAP = 0.5
    from_scratch = False
    if from_scratch:
        read_and_save_raw_data(FILENAME)
        ...
    else:
        for ws in range(5, 30, 5):
            for ol in np.arange(0.25, 1., 0.1):
                main(ws, ol)

