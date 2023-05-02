import copy
import pandas as pd
from abc import ABC, abstractmethod


class NormalizationStrategy(ABC):
    @abstractmethod
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        ...
    def denormalize(self, data: pd.DataFrame) -> pd.DataFrame:
        ...        



class MinMax(NormalizationStrategy):
    def __init__(self, target_min = 0, target_max = 1):
        self.min = target_min
        self.max = target_max
    # normalize to scale of [0, 1]
    

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        def _slice(): return df.iloc[:, srt:end]
            
        checkpoint = copy.deepcopy(df)
        for i in range(3):  # normalize similar quantities in batch
            srt = 2 + 3 * i # shift 3 places every time
            end = srt + 3

            df.iloc[:, srt:end] = (_slice() - _slice().values.min()) / (_slice().values.max() - _slice().values.min())

            # normalize to scale of desired max and min
            df.iloc[:, srt:end] = _slice() * (self.max - self.min) + self.min

        return df
    def denormalize(self, data: pd.DataFrame) -> pd.DataFrame:
        ...    

class MeanSdev(NormalizationStrategy):
    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        # for case in ['Acc', 'Gyr', 'Euler']:
        # Indices:
        # 
        checkpoint = copy.deepcopy(df)
        def _slice(): return df.iloc[:, srt:end]
            
        for i in range(3):  # TODO refactor to extract loop as a method, code is replicated
            srt, end = 3 + i - 1, 2 * 3 + i - 1 # shift 3 places every time
            # refer to the relevant slice
            df.iloc[:, srt:end] = (_slice() - _slice().values.mean()) / _slice().values.std() # normalize said slice
            
        return df
    def denormalize(self, data: pd.DataFrame) -> pd.DataFrame:
        ...    

