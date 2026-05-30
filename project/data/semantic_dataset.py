import pandas as pd
from datasets import load_dataset
import json
from typing import Union
import os
import numpy as np

carte_classification = ['ramen_ratings', 'zomato', 'roger_ebert', 'spotify', 
      'yelp', 'chocolate_bar_ratings', 'coffee_ratings',
        'michelin', 'whisky', 'us_accidents_severity', 'nba_draft']


def _encode_if_category(column: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    # copied from old OpenML Python adapter to maintain comparison with tabpfn
    if column.dtype.name == "category":
        column = column.cat.codes.astype(np.float32)
        mask_nan = column == -1
        column[mask_nan] = np.nan
    return column

def read_carte_classification_datasets(names=None, max_samples=None):
    if names is None:
        names = carte_classification
    if isinstance(names, str):
        names = [names]
    path = '/Users/code/universal_inference_machine/dataset/carte-benchmark/data_carte/'
    df_dict = {}
    for name in names:
        config_file = os.path.join(path, name, 'config_data.json')
        parquet_file = os.path.join(path, name, 'raw.parquet')
        csv_file = os.path.join(path, name, 'raw.csv')
        with open(config_file, 'r') as f:
            config = json.load(f)
        df = pd.read_parquet(parquet_file)
        # df = df.apply(_encode_if_category)
        target_name = config['target_name']
        num_class = df[target_name].nunique()
        target_column = df[target_name]
        # print(df.columns)
        if max_samples is not None and len(df) > max_samples:
            # sample per class to ensure all classes are represented, if possible
            df = df.groupby(config['target_name'], group_keys=False).apply(
                lambda x: x.sample(min(len(x), max_samples // num_class), random_state=42),
                include_groups=False)
        # y = df[target_name]
        y = target_column.iloc[df.index]
        if target_name in df.columns:
            X = df.drop(columns=[target_name])
        else:
            X = df
        # Convert non-numeric columns to object to avoid dtype warnings in tokenizer
        for col in X.columns:
            if X[col].dtype.name not in ('int64', 'int32', 'float64', 'float32', 'bool'):
                X[col] = X[col].astype(object)
        df_dict[name] = (X, y)
    X.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    return df_dict      


if __name__ == "__main__":
    datasets = read_carte_classification_datasets('zomato', 2000)
    print(datasets)