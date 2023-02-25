from typing import Tuple
import re
import copy
import pandas as pd
from abc import ABC, abstractmethod

DEFAULT_PATH = './datasets/opendataset' 
class LoadingStrategy(ABC):
    def __init__(self, base_path: str=DEFAULT_PATH):
        self.base_path = base_path 
    @abstractmethod
    def load_sensor_data(self, *args, **kwargs) -> Tuple[pd.DataFrame, dict]:
        ...
    @abstractmethod
    def load_label_data(self, *args, **kwargs) -> pd.DataFrame:
        ...



class FromPath(LoadingStrategy):
    def __init__(self, base_path: str=DEFAULT_PATH):
        super().__init__(base_path)

    def load_sensor_data(self, file_name: str) -> Tuple[pd.DataFrame, dict]:
        """
        Loads sensor data for a single specified subject, task and run
        Parameters:
            sub: subject ID
            tsk: task ID
            run: trial ID
        Returns: Dataframe containing the data
        """
        metadata = copy.deepcopy(locals())
        metadata.pop('self')
        try:
            data = pd.read_csv(file_name)
        except FileNotFoundError:
            print("The sensor data file could not be found. Returning None.")
            print('Passed arguments:')
            print(f'path: {file_name}')
            print('-'*80)
            return None
        data = data.astype(float)  # convert all to float. Useful for accessing views of dataframes, instead of copies.
        return data, metadata
    
    def load_label_data(self):
        """
        Loads label data for specified subject
        Parameters:
            sub: subject ID
        Returns: Dataframe containing the label data
        """
        metadata = copy.deepcopy(locals())
        metadata.pop('self')        
        sub, = self._transform_args(sub)
        file_name = f'{self.base_path}/label_data/SA{sub}_label.xlsx'
        try:
            data = pd.read_excel(file_name)
        except FileNotFoundError:
            print("The label data file could not be found. Returning None.")
            print('Passed arguments:')
            print(f'subject: {sub}')
            print('-'*80)
            return None

        data = data.fillna(method='ffill')

        # transform all items in the task ID to integer
        _col = 'Task Code (Task ID)'
        data[_col] = data[_col].apply(lambda s: int(re.findall('\(.*?\)', s)[0][1:-1]))  # regex to find values inside parentheses
        return data        
        ...
    

class FromCase(LoadingStrategy):
    def __init__(self, base_path: str = DEFAULT_PATH):
        super().__init__(base_path)

    @staticmethod
    def _transform_int(num):  # Format string
        return str(num).zfill(2)
    @staticmethod
    def _transform_args(*args):
        return tuple(FromCase._transform_int(arg) for arg in args)
    
    def load_sensor_data(self, sub: int=0, tsk: int=0, run: int=0) -> Tuple[pd.DataFrame, dict]:
        """
        Loads sensor data for a single specified subject, task and run
        Parameters:
            sub: subject ID
            tsk: task ID
            run: trial ID
        Returns: Dataframe containing the data
        """
        metadata = copy.deepcopy(locals())
        metadata.pop('self')
        sub, tsk, run = self._transform_args(sub, tsk, run)

        file_name = f'{self.base_path}/sensor_data/SA{sub}/S{sub}T{tsk}R{run}.csv'
        try:
            data = pd.read_csv(file_name)
        except FileNotFoundError:
            print("The sensor data file could not be found. Returning None.")
            print('Passed arguments:')
            print(f'subject: {sub}, task: {tsk}, run: {run}')
            print('-'*80)
            return None
        data = data.astype(float)  # convert all to float. Useful for accessing views of dataframes, instead of copies.
        return data, metadata
    
    def load_label_data(self, sub: int) -> pd.DataFrame:
        """
        Loads label data for specified subject
        Parameters:
            sub: subject ID
        Returns: Dataframe containing the label data
        """
        metadata = copy.deepcopy(locals())
        metadata.pop('self')        
        sub, = self._transform_args(sub)
        file_name = f'{self.base_path}/label_data/SA{sub}_label.xlsx'
        try:
            data = pd.read_excel(file_name)
        except FileNotFoundError:
            print("The label data file could not be found. Returning None.")
            print('Passed arguments:')
            print(f'subject: {sub}')
            print('-'*80)
            return None

        data = data.fillna(method='ffill')

        # transform all items in the task ID to integer
        _col = 'Task Code (Task ID)'
        data[_col] = data[_col].apply(lambda s: int(re.findall('\(.*?\)', s)[0][1:-1]))  # regex to find values inside parentheses
        return data
