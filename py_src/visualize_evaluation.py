"""

"""
import os
import re
from typing import Union

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objs as go
import plotly.express as px

from file_io import load_evaluation_results


def extract_numerical_data(string):
    mean, sdev = match_mean(string), match_sdev(string)
    ...
    return mean, sdev

def extract_wfe_metadata(string):
    ws, ol = match_windowsize(string), match_overlap(string)
    ...
    return ws, ol

def match_overlap(string):
    return eval(re.search(r'_ol(\d+\.+\d+)', string).group(1))

def match_windowsize(string):
    return eval(re.search(r'ws(\d+)_', string).group(1))

def transform_column(callable_: callable, column: pd.Series) -> pd.Series | None:
    """

    @param callable_: function applied on a single item
    @param column:
    @return:
    """
    try:
        return column.apply(callable_)
    except Exception as e:
        print(e)
        return None


def match_mean(string: str) -> Union[int, float]:
    return eval(re.search(r'^(\d+\.\d+)', string).group(1))



def match_sdev(string: str) -> Union[int, float]:
    return eval(re.search(r"\(\+/-\s(\d+\.\d+)", string).group(1))


def generate_surface_plot(df: pd.DataFrame, name, colorscale, color=None):
    xi = np.linspace(df['Window Size'].min(), df['Window Size'].max(), 100)
    yi = np.linspace(df['Overlap'].min(), df['Overlap'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((df['Window Size'], df['Overlap']), df['test_f1'], (xi, yi), method='cubic')
    # color = np.full(zi.shape, 0.5)

    s = go.Surface(x=xi, y=yi, z=zi, surfacecolor=color, name=name, colorscale=colorscale)
    return s


def main():
    df = pd.DataFrame(load_evaluation_results(6))
    ...

    df = format_data(df)
    surfaces = []
    classifier_column = df['classifier']
    colors = ['Viridis', 'Cividis', 'Inferno', 'Plasma']
    for i, algorithm in enumerate(classifier_column.unique()):
        slice_df = df[classifier_column == algorithm]
        surfaces.append(generate_surface_plot(slice_df, algorithm, colorscale=colors[i]))


    fig = generate_figure(surfaces)
    fig.show()
    ...


def generate_figure(s):
    fig = go.Figure(data=s)
    # Update layout
    fig.update_layout(
        title='3D Surface Plot',
        scene=dict(
            xaxis_title='Window Size',
            yaxis_title='Overlap',
            zaxis_title='F1-Score',
        )
    )
    fig.show()
    return fig


def format_data(df):
    for column_name in df.columns:
        new_col = transform_column(match_mean, df[column_name])
        if new_col is not None:
            df[column_name] = new_col
            del new_col
        new_col = transform_column(match_windowsize, df[column_name])
        if new_col is not None:
            df['Window Size'] = new_col
            del new_col
        new_col = transform_column(match_overlap, df[column_name])
        if new_col is not None:
            df['Overlap'] = new_col
            del new_col
    return df


if __name__ == '__main__':
    main()
