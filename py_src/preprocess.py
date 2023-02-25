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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from normalization_strategies import NormalizationStrategy, MinMax, MeanSdev
from loading_strategies import LoadingStrategy, FromCase, FromPath
from graphics import Visualizer


class Loader:
    """
    Responsible for loading single files
    thin wrapper over pandas
    """
    def __init__(self, strategy: LoadingStrategy=FromCase()) -> None:
    
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


    def load_sensor_data(self, *args, **kwargs) -> pd.DataFrame:
        try: 
            data, metadata = self._strategy.load_sensor_data(*args, **kwargs)
            return data, metadata
        except TypeError:
            print('Data collection failed.')
            return None, None

    def load_label_data(self, *args) -> pd.DataFrame:
        try:
            data = self._strategy.load_label_data(*args)
            return data
        except TypeError:
            print('Data collection failed.')
            return None, None



class Splitter():
    """
    Handles splitting data into test, train and validation set
    thin wrapper over sklearn
    """


    def split(train: float, test: float, validate: float):
        
        ...
        
class Filterer:
    ...

# class Padder:
#     """
#     Responsible for applying padding to an array
#     Thin wrapper around numpy's padding functionalities
#     """

#     def __init__(self, mode='constant'):
#         self.mode = mode

#     def left_pad(self, array, num_missing_items):
#         # [1, 2, 3] -> [0, 0, 0, 1, 2, 3]
#         padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode)
#         return padded_array

#     def right_pad(self, array, num_missing_items):
#         # [1, 2, 3] -> [1, 2, 3, 0, 0, 0]
#         padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
#         return padded_array


# class SpectrogramExtractor:
#     """
#     extracts spectrograms in decibels from a time series signal
#     """

#     def __init__(self, frame_size, hop_length, _type='log_spectrogram'):
#         self.frame_size = frame_size
#         self.hop_length = hop_length
#         self._type = _type

#     def extract(self, signal):
#         # compute short-time fourier transform
#         return 



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



class Windower:
    def __init__(self, window_size, overlap: float, ):

        self.window_size = window_size
        self.overlap = overlap



class Saver:
    """
    Saves features and their minmax values
    """

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir

    def save_feature(self, norm_feature, file_path):
        save_path = self._generate_save_path(file_path)
        np.save(save_path, norm_feature)

    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[-1]
        save_path = os.path.join(self.feature_save_dir, file_name + '.npy')
        return save_path

    def save_min_max_values(self, original_min_max_dict):
        save_path = os.path.join(self.min_max_values_save_dir, 'min_max_values.pkl')
        self._save(original_min_max_dict, save_path)

    @staticmethod
    def _save(data, save_path):
        ...


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
        self.loader: Loader = Loader(strategy=FromPath()) 
        self.normalizer: Normalizer = None


    def batch_load(self):
        """
        Uses a loader object to load all files in a directory and labels them
        """
        
        path = f'{self.loader._strategy.base_path}/sensor_data' 
        dfs = []

        for root, subject_folder_names, _ in os.walk(path):
            for subject_folder_name in subject_folder_names:
                
                subject_folder_path = os.path.join(root, subject_folder_name)
                for root, _, files in os.walk(subject_folder_path):
                    for file_name in files: 
                        file_path = os.path.join(root, file_name)
                        data, metadata = self.loader.load_sensor_data(file_path)
                        dfs.append(data)
        batch_df = pd.concat(dfs)


        ...

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




def batch_preprocess(sample_rate, duration, channels, frame_size, hop_length, spectrograms_save_dir,
                     min_max_values_save_dir, audio_files_dir):
    loader = Loader(sample_rate, duration, channels)


    normalizer = Normalizer(0, 1)
    saver = Saver(spectrograms_save_dir, min_max_values_save_dir)

    pppl = PreProcessingPipeline()  # Pre-Processing Pipeline
    pppl.loader = loader

    pppl.normalizer = normalizer
    pppl.saver = saver

    pppl.process(audio_files_dir)



def test_1():
    SUBJECT = 10
    TASK = 6
    RUN = 3

    # Load data
    loader = Loader()

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

def test_2():
    pppl = PreProcessingPipeline()
    pppl.batch_load()
    ...


if __name__ == '__main__':
    test_1()
    # test_2()
    # FRAME_SIZE = 512
    # HOP_LENGTH = 256
    # DURATION = 0.74
    # SAMPLE_RATE = 44100
    # CHANNELS = 1

    # SPECTROGRAMS_SAVE_DIR = '../data/spectrograms'
    # MIN_MAX_VALUES_SAVE_DIR = '../data/min_max_values'
    # FILES_DIR = '../data/input_files'

    # batch_preprocess(sample_rate=SAMPLE_RATE,
    #                  duration=DURATION,
    #                  channels=CHANNELS,
    #                  frame_size=FRAME_SIZE,
    #                  hop_length=HOP_LENGTH,
    #                  spectrograms_save_dir=SPECTROGRAMS_SAVE_DIR,
    #                  min_max_values_save_dir=MIN_MAX_VALUES_SAVE_DIR,
    #                  audio_files_dir=FILES_DIR)
