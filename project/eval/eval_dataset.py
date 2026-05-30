from tqdm import tqdm
from sklearn.base import BaseEstimator
from functools import partial
import itertools
import time
from joblib import Parallel, delayed
import torch
import numpy as np
import os
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score
from typing import Union
import pandas as pd
import json

def check_file_exists(path):
    """Checks if a pickle file exists. Returns None if not, else returns the unpickled file."""
    if (os.path.isfile(path)):
        # print(f'loading results from {path}')
        with open(path, 'rb') as f:
            return np.load(f, allow_pickle=True).tolist()
    return None


def auc_metric(target, pred, multi_class='ovo', numpy=False):
    lib = np if numpy else torch
    if isinstance(target, pd.Series):
        target = target.values
    if not numpy:
        target = torch.tensor(target) if not torch.is_tensor(target) else target
        pred = torch.tensor(pred) if not torch.is_tensor(pred) else pred
    if len(lib.unique(target)) > 2:
        if not numpy:
            return torch.tensor(roc_auc_score(target, pred, multi_class=multi_class))
        return roc_auc_score(target, pred, multi_class=multi_class)
    else:
        if len(pred.shape) == 2:
            pred = pred[:, 1]
        if not numpy:
            return torch.tensor(roc_auc_score(target, pred))
        return roc_auc_score(target, pred)


# def transformer_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, device='cpu', classifier=None, onehot=False, **kwargs):
#     from sklearn.feature_selection import SelectKBest
#     from sklearn.impute import SimpleImputer
#     from sklearn.compose import ColumnTransformer
#     from sklearn.preprocessing import OneHotEncoder
#     if onehot:
#         ohe = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', max_categories=10,
#                                 sparse_output=False), cat_features)], remainder=SimpleImputer(strategy="constant", fill_value=0))
#         ohe.fit(x)
#         x, test_x = ohe.transform(x), ohe.transform(test_x)
#         if x.shape[1] > 100:
#             if not is_classification(metric_used):
#                 raise ValueError('feature selection is only supported for classification tasks')
#             skb = SelectKBest(k=100).fit(x, y)
#             x, test_x = skb.transform(x), skb.transform(test_x)
#     elif classifier is not None:
#         classifier.cat_features = cat_features

#     if classifier is None:
#         raise ValueError('Classifier is not provided for transformer_metric')
        
#     tick = time.time()
#     classifier.fit(x, y)
#     fit_time = time.time() - tick
#     # print('Train data shape', x.shape, ' Test data shape', test_x.shape)
#     tick = time.time()
#     pred = classifier.predict_proba(test_x)
#     inference_time = time.time() - tick
#     times = {'fit_time': fit_time, 'inference_time': inference_time}
#     metric = metric_used(test_y, pred)

#     return metric, pred, times

# def evaluate_position(
#     eval_xs, 
#     eval_ys, 
#     categorical_feats, 
#     model, 
#     n_samples, 
#     eval_position, 
#     overwrite, 
#     save, 
#     base_path, 
#     path_interfix, 
#     method, 
#     ds_name, 
#     fetch_only=False,
#     max_time=300, 
#     split_number=1,
#     metric_used=None, 
#     device='cpu', 
#     verbose=0, 
#     **kwargs,
# ):

#     path = os.path.join(base_path, f'results/tabular/{path_interfix}/results_{method}_{ds_name}_{eval_position}_{n_samples}_{split_number}_{device}.npy')
#     # log_path =
#     # Load results if on disk
#     if not overwrite:
#         result = check_file_exists(path)
#         if result is not None:
#             print(f'Loaded saved result for {path}')
#             return result
#         elif fetch_only:
#             print(f'Could not load saved result for {path}')
#             return None
        

#     # preserve split number for poetree to ensure same split is used when loading model from checkpoint
#     kwargs['split_number'] = split_number
    
#     if eval_xs is None:
#         print(f"No dataset could be generated {ds_name} {n_samples}")
#         return None

#     eval_ys = (eval_ys > torch.unique(eval_ys).unsqueeze(0)).sum(axis=1).unsqueeze(-1)

#     if isinstance(model, nn.Module):
#         model = model.to(device)
#         eval_xs = eval_xs.to(device)
#         eval_ys = eval_ys.to(device)

#     start_time = time.time()

#     if isinstance(model, nn.Module):  # Two separate predict interfaces for transformer and baselines
#         # max_time does not affect nn models
#         outputs, best_configs = transformer_predict(
#             model, 
#             eval_xs, 
#             eval_ys, 
#             eval_position, 
#             metric_used=metric_used, 
#             categorical_feats=categorical_feats,
#             inference_mode=True, 
#             device=device, 
#             extend_features=True, 
#             verbose=verbose,
#             **kwargs
#         ), None
#     else:
#         _, outputs, best_configs = baseline_predict(
#             model, 
#             eval_xs, 
#             eval_ys, 
#             categorical_feats, 
#             eval_pos=eval_position,
#             device=device, 
#             max_time=max_time, 
#             metric_used=metric_used, 
#             verbose=verbose, 
#             **kwargs
#         )
#     eval_ys = eval_ys[eval_position:]
#     if outputs is None:
#         print('Execution failed', ds_name)
#         return None

#     if torch.is_tensor(outputs):  # Transfers data to cpu for saving
#         outputs = outputs.cpu()
#         eval_ys = eval_ys.cpu()

#     ds_result = None, outputs, eval_ys, best_configs, time.time() - start_time

#     if save:
#         with open(path, 'wb') as f:
#             np.save(f, np.asarray(ds_result, dtype=object))
#             if verbose > 0:
#                 print(f'saved results to {path}')

#     return ds_result


# def evaluate(
#     datasets, 
#     n_samples, 
#     eval_positions, 
#     metric_used, 
#     model, 
#     device='cpu',
#     verbose=False, 
#     return_tensor=False, 
#     pca = False,
#     **kwargs
# ):
#     """
#     Evaluates a list of datasets for a model function.

#     :param datasets: List of datasets
#     :param n_samples: maximum sequence length
#     :param eval_positions: List of positions where to evaluate models
#     :param verbose: If True, is verbose.
#     :param metric_used: Which metric is optimized for.
#     :param return_tensor: Wheater to return results as a pytorch.tensor or numpy, this is only relevant for transformer.
#     :param kwargs:
#     :return:
#     """
#     overall_result = {'metric_used': tabular_metrics.get_scoring_string(metric_used), 'n_samples': n_samples, 'eval_positions': eval_positions}

#     aggregated_metric_datasets, num_datasets = torch.tensor(0.0), 0

#     # For each dataset
#     for [ds_name, X, y, categorical_feats, feature_names, _] in datasets:
#         kwargs['feature_names'] = feature_names
#         kwargs['dataset_name'] = ds_name
#         dataset_n_samples = min(len(X), n_samples)
#         if verbose:
#             print(f'Evaluating {ds_name} with {len(X)} samples')
            

#         aggregated_metric, num = torch.tensor(0.0), 0
#         ds_result = {}

#         for eval_position in (eval_positions if verbose else eval_positions):
#             if eval_position is None or (2 * eval_position > dataset_n_samples):
#                 eval_position_real = int(dataset_n_samples * 0.5)
#             else:
#                 eval_position_real = eval_position
#             eval_position_n_samples = int(eval_position_real * 2.0)

#             # Set as speicified
#             # eval_position_n_samples = dataset_n_samples
#             # eval_position_real = eval_position
                        
#             # r should be 
#             # None, outputs, eval_ys, best_configs, time_used
#             r = evaluate_position(
#                 X, 
#                 y, 
#                 model=model, 
#                 categorical_feats=categorical_feats,
#                 n_samples=eval_position_n_samples, 
#                 ds_name=ds_name, 
#                 eval_position=eval_position_real, 
#                 metric_used=metric_used, 
#                 device=device,
#                 verbose=verbose - 1,
#                 pca = pca,
#                 **kwargs
#             )

#             if r is None:
#                 print('Execution failed', ds_name)
#                 continue

#             _, outputs, ys, best_configs, time_used = r

#             if torch.is_tensor(outputs):
#                 outputs = outputs.to(outputs.device)
#                 ys = ys.to(outputs.device)


#             if not return_tensor:
#                 def make_scalar(x): return float(x.detach().cpu().numpy()) if (torch.is_tensor(x) and (len(x.shape) == 0)) else x
#                 new_metric = make_scalar(new_metric)
#                 ds_result = {k: make_scalar(ds_result[k]) for k in ds_result.keys()}

#             lib = torch if return_tensor else np
#             if not lib.isnan(new_metric).any():
#                 aggregated_metric, num = aggregated_metric + new_metric, num + 1

#         overall_result.update(ds_result)
#         if num > 0:
#             aggregated_metric_datasets, num_datasets = (aggregated_metric_datasets + (aggregated_metric / num)), num_datasets + 1

#     overall_result['mean_metric'] = aggregated_metric_datasets / num_datasets

#     return overall_result



# def _eval_single_dataset_wrapper(**kwargs):
#     max_time = kwargs['max_time']
#     time_string = '_time_'+str(max_time) if max_time else ''
#     metric_used_string = 'AUC'
#     result = evaluate(method=kwargs['model_name']+time_string+metric_used_string, **kwargs)
#     result['model'] = kwargs['model_name']
#     result['dataset'] = kwargs['datasets'][0][0]
#     result['max_time'] = kwargs['max_time']
#     return result

def eval_on_datasets(
    model, 
    datasets_dict, 
    seed,
    device='auto', 
    perturbation_method: Union[None, callable, str]=None,
    **kwargs
):
    # if callable(model):
    #     model_callable = model
    #     if device == 'auto':
    #         device = 'cpu'
    # elif isinstance(model, BaseEstimator):
    #     model_callable = partial(transformer_metric, classifier=model, N_ensemble_configurations=num_ensemble, **kwargs)
    #     device_param = [v for k, v in model.get_params().items() if "device" in k]
    #     if device == "auto":
    #         device = device_param[0] if len(device_param) > 0 else "cpu"
    # else:
    #     raise ValueError(f"Got model {model} of type {type(model)} which is not callable or a BaseEstimator")
    if not isinstance(model, BaseEstimator):
        raise ValueError(f"Got model {model} of type {type(model)} which is not a BaseEstimator")
    
    dataset_names = list(datasets_dict.keys())

    # print(f"evaluating {model_name} on {device}")
    if "cuda" in device:
        results = []
        tqdm_bar = tqdm(list(itertools.product(dataset_names, seed)))
        for _, (ds_name, _seed ) in enumerate(tqdm_bar):
            # if _ > 4:
            #     break
            tqdm_bar.set_description(f"evaluating on {device} {ds_name}")
            X, y = datasets_dict[ds_name]
            X = X.copy()
            if perturbation_method == 'header_remove':
                X = X.values
                y = y.values
            elif perturbation_method == 'header_permutation':
                rng = np.random.default_rng(seed=_seed)
                X.columns = np.random.permutation(X.columns)
            elif perturbation_method == 'header_mask':
                masked_text = kwargs.get('masked_text', 'feature')
                X.columns = [f'{masked_text}_{i}' for i in range(X.shape[1])]
            elif perturbation_method == 'column_extend': 
                json_dict = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'rewrite_column', 'extended_meanings.json'), 'r'))
                if ds_name in json_dict:
                    column_meaning_dict = json_dict[ds_name]
                    new_column_names = [f"{col} ({column_meaning_dict.get(col, 'No extended meaning')})" for col in X.columns]
                    print(new_column_names)
                    X.columns = new_column_names
                
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=_seed, stratify=y)
            
            # if perturbation_method is None:
            model.fit(X_train, y_train)

            prediction_probabilities = model.predict_proba(X_test)
            
            auc_score = auc_metric(y_test, prediction_probabilities, multi_class='ovo').item()
            result = {'dataset_name': ds_name, 'seed': _seed, 'AUC': auc_score}
            results.append(result)
    
    final_results = {}
    for result in results:
        if result['dataset_name'] not in final_results:
            final_results[result['dataset_name']] = []
        final_results[result['dataset_name']].append(result['AUC'])
    
    return final_results