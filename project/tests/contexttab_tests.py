import os
import sys
import pandas as pd
import itertools
import torch

# torch.set_num_threads(1)
# torch.manual_seed(42)

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
    # None
    # 'header_permutation',
    'header_mask',
]

kwargs = {
    'masked_text': 'col'
}

from sap_rpt_oss import SAP_RPT_OSS_Classifier
from data import read_carte_classification_datasets
from eval import eval_on_datasets

model_name = 'SAP_RPT_OSS_Classifier_baseline'

sample_perturbation_combinations = list(itertools.product(max_samples, perturbation_methods))

for sample, perturbation_method in sample_perturbation_combinations:
    dataset = read_carte_classification_datasets(max_samples=sample)
    model = SAP_RPT_OSS_Classifier()
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