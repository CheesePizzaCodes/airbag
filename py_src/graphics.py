from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
import plotly.express as px

from util import extract_marks


def generate_graph_from_df(df: pd.DataFrame):
    fig, ax = plt.subplots()
    for i in range(3, 11 + 1):
        line, = ax.plot(df.iloc[:, i])
        line.set_label(df.columns[i])
        ax.legend()

    plt.show(block=False)


class Visualizer:
    def __init__(self, sensor_data: pd.DataFrame, metadata: dict, label_data: pd.DataFrame) -> None:
        self.sensor_data = sensor_data
        self.sub = metadata['sub']
        self.tsk = metadata['tsk']
        self.run = metadata['run']
        self.label_data = label_data
        self.presets = {  # 0: Plot title            1: x label          2: legends list
            'accel': ['Acceleration over time', 'acceleration ($g$)', ['AccX', 'AccY', 'AccZ']],
            'gyro': ['Angular velocity over time', r'angular velocity ($\frac{^o}{s}$)', ['GyrX', 'GyrY', 'GyrZ']],
            'euler': ['Euler angles over time', 'angle ($^o$)', ['EulerX', 'EulerY', 'EulerZ']]
        }
        self.fig, self.axs = plt.subplots(len(self.presets), sharex=True)
        for _ in self.axs:
            _.label_outer()
        pass

    def generate_graphs(self):
        self.fig.suptitle(f'Subject: {self.sub}, Task: {self.tsk}, Run: {self.run}')
        # self.axs[-1].set_xlabel('time ($s$)') 
        for index, value in enumerate(self.presets.values()):
            ax = self.axs[index]
            # ax.set_title(value[0])
            ax.set_ylabel(value[1])
            for i in value[2]:
                line, = ax.plot(self.sensor_data['TimeStamp(s)'], self.sensor_data[i], '')
                line.set_label(i)
                ax.legend(loc=1)
                self._add_marks(ax)
        plt.show(block=False)
        return

    def _add_marks(self, ax):
        time = self.sensor_data['TimeStamp(s)']

        marks = extract_marks(label_data=self.label_data, task=self.tsk, run=self.run)
        if marks is None:
            return
        onset_frame, impact_frame = marks
        ax.axvline(x=time[onset_frame], color='r')
        ax.axvline(x=time[impact_frame], color='r')


def plot_superimposed_histograms(data_list: List[np.ndarray]):
    for data in data_list:
        # data[data == -np.inf] = 0
        plt.hist(data, density=True, alpha=0.5, bins=100)
    plt.show()


def plot_histograms_grid(data_list: List[np.ndarray], num_rows: int):
    fig = plt.figure()
    # n_rows, n_cols = data.shape
    for i in range(len(data_list[0].T)):  # number of attributes e.g. subplots
        try:
            ax = fig.add_subplot(num_rows, 9, i+1)
            for data in data_list:  # reverse so that negative cases are in the background
                sample = data.T[i]
                ax.hist(sample, bins=20, density=True, alpha=0.5)
        except ValueError as e:
            print(e)
            return fig, ax
    return fig, ax


def plot_pairwise_scatter(data, labels):
    # Assume `data` is your 2D numpy array
    num_samples = int(data.shape[0] * 0.1)  # Number of samples you want to draw
    # Generate a random sample of indices
    indices = np.random.choice(data.shape[0], size=num_samples, replace=False)
    # Select the corresponding rows
    sampled_data = data[indices, :]
    sampled_labels = labels[indices]

    fig = px.scatter_matrix(sampled_data, color=sampled_labels, opacity=0.1)
    fig.show()
    return fig


if __name__ == '__main__':

    from preprocess import Loader, Normalizer
    from normalization_strategies import MinMax

    print('*-' * 20 + 'Visualización de datos' + '-*' * 20)
    while True:

        sub = int(input('ID del sujeto: '))
        tsk = int(input('ID de la actividad: '))
        run = int(input('ID del intento: '))

        # sub, run, tsk = 8, 2, 30

        loader = Loader()
        normalizer = Normalizer(strategy=MinMax(0, 1))

        sensor_data, sensor_metadata = loader.load_sensor_data(sub, tsk, run)
        # sensor_data = normalizer.normalize(sensor_data)
        label_data = loader.load_label_data(sub)

        visualizer = Visualizer(sensor_data, sensor_metadata, label_data)
        visualizer.generate_graphs()

        if input('Press <enter> to continue or enter <q> to exit... ') == 'q':
            break
