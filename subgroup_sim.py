# %%
import numpy as np
import os
import pandas as pd
import pickle
import torch

from matplotlib import pyplot as plt
from univ.utils import similarity as sim
from univ.utils import load_data as ld
from univ.utils import measures as ms
from univ.utils import model_import as mi
from univ.measures import rep_sim as rs
from univ.utils.minibatch_cka import MinibatchCKA
from univ.utils import plots as pl
from tqdm import tqdm

# %%
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

# %%
MODELS = ['resnet18', 'resnet50', 'wide_resnet50_2', 'wide_resnet50_4', 'densenet161', 'resnext50_32x4d', 'vgg16_bn',
          'tiny_vit_5m']
LAYERS = ['avgpool', 'avgpool', 'avgpool', 'avgpool', 'features.norm5', 'avgpool', 'classifier.5', 'head.drop']

# %%
from typing import Literal
import rtd
from univ.measures import rep_sim as rs


def get_predictions(models: list, dataloaders: list, device: torch.device):
    outputs = sim.get_outputs(models, dataloaders, device)
    preds = ms.get_class_predictions(outputs)
    probs = ms.get_probabilities(outputs)
    return preds, probs


def get_agreement_indices(preds: dict):
    aggr_indices, disag_indices = {}, {}
    for model1 in preds.keys():
        aggr_indices[model1], disag_indices[model1] = {}, {}
        for model2 in preds.keys():
            aggr_indices[model1][model2] = []
            disag_indices[model1][model2] = []
            if model1 != model2:
                for num, i, j in zip(range(len(preds[model1])), preds[model1], preds[model2]):
                    if i == j:
                        aggr_indices[model1][model2].append(num)
                    else:
                        disag_indices[model1][model2].append(num)
    return aggr_indices, disag_indices


def calculate_mean_sim(acts: dict, aggr_indices: dict, disag_indices: dict, device: torch.device,
                       sim: Literal["cka", "cos", "jacc", "proc", "rtd"] = 'cka',
                       k: int = 10, perc_k: float = 0):
    mean_sim_agg, mean_sim_dis = [], []
    for model1 in acts.keys():
        sims_agg, sims_dis = [], []
        for model2 in tqdm(acts.keys()):
            if model1 == model2:
                sims_agg.append(1)
                sims_dis.append(1)
            else:
                agg_acts1 = torch.flatten(torch.vstack(acts[model1])[aggr_indices[model1][model2]], start_dim=1)
                agg_acts2 = torch.flatten(torch.vstack(acts[model2])[aggr_indices[model1][model2]], start_dim=1)
                dis_acts1 = torch.flatten(torch.vstack(acts[model1])[disag_indices[model1][model2]], start_dim=1)
                dis_acts2 = torch.flatten(torch.vstack(acts[model2])[disag_indices[model1][model2]], start_dim=1)
                if sim == 'cos':
                    sims_agg.append(rs.second_order_cos_sim(agg_acts1.to(device), agg_acts2.to(device)).cpu())
                    sims_dis.append(rs.second_order_cos_sim(dis_acts1.to(device), dis_acts2.to(device)).cpu())
                elif sim == 'jacc':
                    k_agg, k_dis = k, k
                    if perc_k > 0:
                        k_agg = int(len(agg_acts1) * perc_k)
                        k_dis = int(len(dis_acts1) * perc_k)
                    sims_agg.append(torch.mean(rs.knn_jaccard(agg_acts1.to(device), agg_acts2.to(device),
                                                              k=k_agg)).cpu())
                    sims_dis.append(torch.mean(rs.knn_jaccard(dis_acts1.to(device), dis_acts2.to(device),
                                                              k=k_dis)).cpu())
                elif sim == 'cka':
                    minibatch_cka = MinibatchCKA(1, 1, across_models=True)
                    minibatch_cka.update_state_across_models([agg_acts1], [agg_acts2])
                    sims_agg.append(minibatch_cka.result().numpy()[0][0])
                    minibatch_cka.update_state_across_models([dis_acts1], [dis_acts2])
                    sims_dis.append(minibatch_cka.result().numpy()[0][0])
                elif sim == "proc":
                    sims_agg.append(rs.orthogonal_procrustes(agg_acts1, agg_acts2, center_columns=True, use_norm=True).cpu())
                    sims_dis.append(rs.orthogonal_procrustes(dis_acts1, dis_acts2, center_columns=True, use_norm=True).cpu())
                elif sim == "rtd":
                    sims_agg.append(rtd.rtd(agg_acts1.to(device), agg_acts2.to(device)))
                    sims_dis.append(rtd.rtd(dis_acts1.to(device), dis_acts2.to(device)))
                else:
                    raise ValueError(f"Unknown similarity measure: {sim}")
        mean_sim_agg.append(sims_agg)
        mean_sim_dis.append(sims_dis)
    return np.array(mean_sim_agg), np.array(mean_sim_dis)

# %%
models = {
    0: [9, 3, 5, 1, 8],
    1: [9, 7, 2, 1, 8]
}

# %%
with open('/root/cache/outputs_imagenet_std.pkl', 'rb') as f:
    outputs_std = pickle.load(f)
with open('/root/cache/outputs_imagenet_rob.pkl', 'rb') as f:
    outputs_rob = pickle.load(f)

# %%
preds_std = outputs_std['preds']
preds_rob = outputs_rob['preds']

# %%
with open('/root/cache/acts_imagenet_std.pkl', 'rb') as f:
    acts_std = pickle.load(f)
with open('/root/cache/acts_imagenet_rob.pkl', 'rb') as f:
    acts_rob = pickle.load(f)

# %%
activations_std = acts_std['activations']
activations_rob = acts_rob['activations']

# %%
densenet_idx = MODELS.index("densenet161") if "densenet161" in MODELS else None
activations_std[densenet_idx] = [batch_acts.mean(dim=(-1,-2)) for batch_acts in activations_std[densenet_idx]]
activations_rob[densenet_idx] = [batch_acts.mean(dim=(-1,-2)) for batch_acts in activations_rob[densenet_idx]]

# %%
aggr_inds_std, disg_inds_std = get_agreement_indices(preds_std)
aggr_inds_rob, disg_inds_rob = get_agreement_indices(preds_rob)

# %% [markdown]
# ## Computing and caching all potentially relevant comparison results

# %%
import itertools
import pandas as pd
from loguru import logger

metrics = [
    {"name": "cka"},
    {"name": "proc"},
    {"name": "rtd"},
    {"name": "jacc", "k": 10},
    {"name": "jacc", "k": 50},
    {"name": "jacc", "k": 100},
    {"name": "jacc", "perc_k": 0.1},
    {"name": "jacc", "perc_k": 0.01},
    {"name": "jacc", "perc_k": 0.005},
]

def create_subdf(scores: np.ndarray, metric: dict, pred_type: str, eps: float) -> pd.DataFrame:
    data = {"metric": [], "score": [], "model1": [], "model2": [], "k": [], "perc_k": [], "pred_type": [], "eps": []}
    for m1_idx, m2_idx in itertools.product(range(len(scores)), repeat=2):
        data["metric"].append(metric["name"])
        data["score"].append(scores[m1_idx, m2_idx])
        data["model1"].append(MODELS[m1_idx])
        data["model2"].append(MODELS[m2_idx])
        data["k"].append(metric.get("k", None))
        data["perc_k"].append(metric.get("perc_k", None))
        data["pred_type"].append(pred_type)
        data["eps"].append(eps)
    return pd.DataFrame.from_dict(data)


def metric_results_in_cache(df: pd.DataFrame, metric: dict, eps: float) -> bool:
    grouped = df.groupby(["metric", "k", "perc_k", "eps"], dropna=False)["score"].count()
    expected_num_of_comparisons = 128
    try:
        if grouped.loc[(metric["name"], metric.get("k", -1), metric.get("perc_k", -1), eps)] != expected_num_of_comparisons:
            return False
        else:
            return True
    except KeyError as e:
        logger.debug(f"Key({e}) gave KeyError.")
        return False

results_path = "/root/cache/repsim_subgroups.csv"
df = pd.read_csv(results_path, index_col=0)
df.loc[df["k"].isna(), "k"] = -1
df.loc[df["perc_k"].isna(), "perc_k"] = -1
for metric in metrics:
    print(f"Starting with {metric['name']}")

    if not metric_results_in_cache(df, metric, 0.0):
        logger.info(f"{metric} with eps=0.0 does not exist yet. Computing.")
        mean_sim_agg_std, mean_sim_dis_std = calculate_mean_sim(activations_std, aggr_inds_std, disg_inds_std, device, metric["name"], k=metric.get("k", 0), perc_k=metric.get("perc_k", 0.0))
        subdf1 = create_subdf(mean_sim_agg_std, metric, "agree", 0.0)
        subdf2 = create_subdf(mean_sim_dis_std, metric, "disagree", 0.0)
        df = pd.concat([df, subdf1, subdf2]).reset_index(drop=True)
        df.to_csv(results_path)
    else:
        logger.info(f"{metric} with eps=0.0 already exists. Skipping.")

    if not metric_results_in_cache(df, metric, 3.0):
        logger.info(f"{metric} with eps=3.0 does not exist yet. Computing.")
        mean_sim_agg_rob, mean_sim_dis_rob = calculate_mean_sim(activations_rob, aggr_inds_rob, disg_inds_rob, device, metric["name"], k=metric.get("k", 0), perc_k=metric.get("perc_k", 0.0))
        subdf1 = create_subdf(mean_sim_agg_rob, metric, "agree", 3.0)
        subdf2 = create_subdf(mean_sim_dis_rob, metric, "disagree", 3.0)
        df = pd.concat([df, subdf1, subdf2]).reset_index(drop=True)
        df.to_csv(results_path)
    else:
        logger.info(f"{metric} with eps=3.0 already exists. Skipping.")
