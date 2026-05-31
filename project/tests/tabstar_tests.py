from importlib.resources import files
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import itertools
from tabstar.tabstar_model import TabSTARClassifier
import os
import sys

root_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(root_dir)

max_samples = [
    128,
    256, 
    512,
    1024, 
    2048,
    4096,
]

seed = [
    0, 1, 2, 3, 5
]

perturbation_methods = [
    # 'header_remove',
    None,
    'header_permutation',
    'header_mask',
    'column_extend',
]

kwargs = {
    'masked_text': 'col'
}

from data import read_carte_classification_datasets
from eval import eval_on_datasets




model_name = 'TabSTARClassifier_baseline'

sample_perturbation_combinations = list(itertools.product(max_samples, perturbation_methods))

for sample, perturbation_method in sample_perturbation_combinations:
    dataset = read_carte_classification_datasets(max_samples=sample)
    model = TabSTARClassifier(device='mps', max_epochs=10, keep_model=False, lora_r=4)
    results = eval_on_datasets(model, dataset, seed, device='cuda', perturbation_method=perturbation_method, **kwargs)
    res_df = pd.DataFrame(results)
    res_df.loc['mean'] = res_df.mean(axis=0)
    model_str = 'baseline'
    if perturbation_method is not None:
        model_str = 'baseline' + '_' + perturbation_method
        if perturbation_method == 'header_mask':
            model_str += '_' + kwargs.get('masked_text', 'feature')
    save_dir = os.path.join(root_dir, 'result', model_name)
    os.makedirs(save_dir, exist_ok=True)
    res_df.to_csv(os.path.join(save_dir, f'{model_str}_{sample}.csv'))
    # print(results)
    break 
print(res_df.loc['mean'].mean())

# csv_path = files("tabstar").joinpath("resources", "imdb.csv")
# x = pd.read_csv(csv_path)
# # Convert string columns to object dtype (TabSTAR doesn't support pandas StringDtype)
# for col in x.columns:
#     if x[col].dtype.name == 'str':
#         x[col] = x[col].astype(object)
# y = x.pop('Genre_is_Drama')

# x.columns = np.random.permutation(x.columns)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# tabstar = TabSTARClassifier(device='mps', max_epochs=10, keep_model=False, lora_r=4)
# tabstar.fit(x_train, y_train)
# # tabstar.save("my_model_path.pkl")
# # tabstar = TabSTARClassifier.load("my_model_path.pkl")
# # y_pred = tabstar.predict(x_test)
# metric = tabstar.score(X=x_test, y=y_test)
# print(f"AUC: {metric:.4f}")