import math
from collections import defaultdict
import os
import pdb
import re

import numpy as np
import sklearn.metrics.pairwise as smp
import torch
import torch.nn.functional as F
import hdbscan
from scipy.linalg import eigh as largest_eigh
from sklearn.cluster import DBSCAN, KMeans

from FL_Backdoor_CV.shprg.shprg import SHPRG, load_initialization_values
from configs import args


def avg(model_updates):
    global_update = dict()
    for name, data in model_updates.items():
        global_update[name] = 1 / args.participant_sample_size * model_updates[name].sum(dim=0, keepdim=True)
    return global_update


def roseagg(model_updates, current_round=0):
    # === construct a distance matrix ===
    keys = list(model_updates.keys())
    indicative_layer_updates = F.normalize(model_updates[keys[-2]])
    K = len(indicative_layer_updates)
    distance_matrix = smp.cosine_similarity(indicative_layer_updates.cpu().numpy()) - np.eye(K)

    # === clustering using dbscan
    partition = DBSCAN(eps=args.threshold, min_samples=1, metric='cosine').fit(indicative_layer_updates.cpu().numpy())
    clusters = dict()
    for i, clu in enumerate(partition.labels_):
        if clu in clusters:
            clusters[clu] += [i]
        else:
            clusters[clu] = [i]

    # === find updates with similar directional contribution ===
    sim_idxs = list()
    for clu in clusters.values():
        if len(clu) > 1:
            sim_idxs.append(clu)

    # === find the master index and remove the other indices within each cluster ===
    remove_idxs = []
    for idxs in sim_idxs:
        idxs = list(idxs)
        idxs.sort()
        if remove_idxs:
            remove_idxs.extend(idxs[1:])
        else:
            remove_idxs = idxs[1:]
    reserve_idxs = list(set(list(range(K))) - set(remove_idxs))
    print(f'Clients with similar directional contribution: {sim_idxs}, '
          f'# Clusters: {max(partition.labels_) + 1}')

    # === partial aggregation. The following process intends to assign a weight to each update ===
    weights = torch.ones(K)
    sims = dict()
    if sim_idxs:
        for sim_idx in sim_idxs:
            sim_idx = list(sim_idx)
            sim_idx.sort()
            sub_distance_matrix = distance_matrix[sim_idx, :][:, sim_idx]
            sub_weights = sub_distance_matrix.sum(axis=0)
            sub_weights = sub_weights / sub_weights.sum()
            for i, idx in enumerate(sim_idx):
                sims[idx] = sub_weights[i]
        for i, s in sims.items():
            weights[i] = s

    # === calculate global update ===
    global_update = defaultdict()
    for name, layer_updates in model_updates.items():
        if 'num_batches_tracked' in name:
            if args.is_poison:
                global_update[name] = torch.sum(layer_updates[args.number_of_adversaries:]) / len(layer_updates[args.number_of_adversaries])
            else:
                global_update[name] = torch.sum(layer_updates) / len(layer_updates)
        else:
            # === normalization norm ===
            local_norms = np.array([torch.norm(layer_updates[i]).cpu().numpy() for i in range(K)]).reshape(-1, 1)
            if args.is_poison:
                kmeans = KMeans(n_clusters=2, n_init='auto').fit(local_norms)
                clusters = dict()
                for i, clu in enumerate(kmeans.labels_):
                    if clu in clusters:
                        clusters[clu] += [i]
                    else:
                        clusters[clu] = [i]
                local_norms_0 = local_norms[clusters[0]]
                local_norms_1 = local_norms[clusters[1]]
                if np.median(local_norms_0) > np.median(local_norms_1):
                    normalize_norm = np.median(local_norms_1)
                else:
                    normalize_norm = np.median(local_norms_0)
            else:
                normalize_norm = np.median(local_norms)

            # === plausible clean ingredient ===
            origin_directions = F.normalize(layer_updates)
            origin_directions = weights.view(-1, 1).to(args.device) * origin_directions

            first_idxs = []
            if sim_idxs:
                # === targeted aggregation within each cluster ===
                for idxs in sim_idxs:
                    idxs = list(idxs)
                    idxs.sort()
                    first_idx = idxs[0]
                    first_idxs.append(first_idx)
                    for idx in idxs[1:]:
                        origin_directions[first_idx] = origin_directions[first_idx] + origin_directions[idx]
                    origin_directions[first_idx] /= torch.norm(origin_directions[first_idx])
                origin_directions = origin_directions[reserve_idxs]

            # === extract plausible clean ingredient ===
            N = origin_directions.size(0)
            X = torch.matmul(origin_directions, origin_directions.T)
            evals_large, evecs_large = largest_eigh(X.detach().cpu().numpy(), eigvals=(N - N, N - 1))
            evals_large = torch.tensor(evals_large)[-1].to(args.device)
            evecs_large = torch.tensor(evecs_large)[:, -1].to(args.device)
            principal_direction = torch.matmul(evecs_large.view(1, -1), origin_directions).T / torch.sqrt(
                evals_large)

            # === reweight partial aggregated model udpates ===
            new_weights = torch.pow(torch.matmul(principal_direction.view(1, -1), origin_directions.T), 2)
            new_weights = new_weights / new_weights.sum()

            # === aggregation ===
            origin_directions += torch.normal(0, 0.003, origin_directions.size()).to(args.device)
            origin_directions = F.normalize(origin_directions)
            scale = normalize_norm
            principal_direction = torch.matmul(new_weights, origin_directions * scale)
            global_update[name] = principal_direction
    return global_update


def foolsgold(model_updates):
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]
    K = len(last_layer_updates)
    cs = smp.cosine_similarity(last_layer_updates.cpu().numpy()) - np.eye(K)
    maxcs = np.max(cs, axis=1)
    # === pardoning ===
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    alpha = np.max(cs, axis=1)
    wv = 1 - alpha
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # === Rescale so that max value is wv ===
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # === Logit function ===
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    # === calculate global update ===
    global_update = defaultdict()
    for name in keys:
        tmp = None
        for i, j in enumerate(range(len(wv))):
            if i == 0:
                tmp = model_updates[name][j] * wv[j]
            else:
                tmp += model_updates[name][j] * wv[j]
        global_update[name] = 1 / len(wv) * tmp

    return global_update


def flame(trained_params, current_model_param, param_updates):
    # === clustering ===
    trained_params = torch.stack(trained_params).double()
    cluster = hdbscan.HDBSCAN(metric="cosine", algorithm="generic",
                              min_cluster_size=args.participant_sample_size // 2 + 1,
                              min_samples=1, allow_single_cluster=True)
    cluster.fit(trained_params)
    predict_good = []
    for i, j in enumerate(cluster.labels_):
        if j == 0:
            predict_good.append(i)
    k = len(predict_good)

    # === median clipping ===
    model_updates = trained_params[predict_good] - current_model_param
    local_norms = torch.norm(model_updates, dim=1)
    S_t = torch.median(local_norms)
    scale = S_t / local_norms
    scale = torch.where(scale > 1, torch.ones_like(scale), scale)
    model_updates = model_updates * scale.view(-1, 1)

    # === aggregating ===
    trained_params = current_model_param + model_updates
    trained_params = trained_params.sum(dim=0) / k

    # === noising ===
    delta = 1 / (args.participant_sample_size ** 2)
    epsilon = 10000
    lambda_ = 1 / epsilon * (math.sqrt(2 * math.log((1.25 / delta))))
    sigma = lambda_ * S_t.numpy()
    print(f"sigma: {sigma}; #clean models / clean models: {k} / {predict_good}, median norm: {S_t},")
    trained_params.add_(torch.normal(0, sigma, size=trained_params.size()))

    # === bn ===
    global_update = dict()
    for i, (name, param) in enumerate(param_updates.items()):
        if 'num_batches_tracked' in name:
            global_update[name] = 1 / k * \
                                  param_updates[name][predict_good].sum(dim=0, keepdim=True)
        elif 'running_mean' in name or 'running_var' in name:
            local_norms = torch.norm(param_updates[name][predict_good], dim=1)
            S_t = torch.median(local_norms)
            scale = S_t / local_norms
            scale = torch.where(scale > 1, torch.ones_like(scale), scale)
            global_update[name] = param_updates[name][predict_good] * scale.view(-1, 1)
            global_update[name] = 1 / k * global_update[name].sum(dim=0, keepdim=True)

    return trained_params.float().to(args.device), global_update


def fltrust(model_updates, param_updates, clean_param_update):
    cos = torch.nn.CosineSimilarity(dim=0)
    g0_norm = torch.norm(clean_param_update)
    weights = []
    for param_update in param_updates:
        weights.append(F.relu(cos(param_update.view(-1, 1), clean_param_update.view(-1, 1))))
    weights = torch.tensor(weights).to(args.device).view(1, -1)
    weights = weights / weights.sum()
    weights = torch.where(weights[0].isnan(), torch.zeros_like(weights), weights)
    nonzero_weights = torch.count_nonzero(weights.flatten())
    nonzero_indices = torch.nonzero(weights.flatten()).flatten()

    print(f'g0_norm: {g0_norm}, '
          f'weights_sum: {weights.sum()}, '
          f'*** {nonzero_weights} *** model updates are considered to be aggregated !')

    normalize_weights = []
    for param_update in param_updates:
        normalize_weights.append(g0_norm / torch.norm(param_update))

    global_update = dict()
    for name, params in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / nonzero_weights * params[nonzero_indices].sum(dim=0, keepdim=True)
        else:
            global_update[name] = torch.matmul(
                weights,
                params * torch.tensor(normalize_weights).to(args.device).view(-1, 1))
    return global_update


def robust_lr(model_updates):
    global_update = dict()
    for name, param in model_updates.items():
        if 'num_batches_tracked' in name or 'running_mean' in name or 'running_var' in name:
            global_update[name] = 1 / args.participant_sample_size * \
                                  model_updates[name].sum(dim=0, keepdim=True)
        else:
            signs = torch.sign(model_updates[name])
            sm_of_signs = torch.abs(torch.sum(signs, dim=0, keepdim=True))
            sm_of_signs[sm_of_signs < args.robustLR_threshold] = -1
            sm_of_signs[sm_of_signs >= args.robustLR_threshold] = 1
            global_update[name] = 1 / args.participant_sample_size * \
                                  (sm_of_signs * model_updates[name].sum(dim=0, keepdim=True))
    return global_update


def jzx(model_updates):
    # pdb.set_trace()
    malicious_ratio, weight = 0.1, 0.1
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]

    client_num = last_layer_updates.shape[0]
    
    l2_norm = torch.norm(last_layer_updates, dim=1)
    sorted_l2_norm, sorted_indices = torch.sort(l2_norm)
    b = sorted_l2_norm[int((1 - malicious_ratio) * client_num)]
    
    linf_norm = torch.norm(last_layer_updates, p=float('inf'), dim=1)
    linf_norm_abs_mean = torch.mean(torch.abs(linf_norm)).item()

    last_layer_global_update = last_layer_updates.mean(dim=0)
    l2_norm_global = torch.norm(last_layer_global_update).item()

    eta = (b / (l2_norm_global + weight * linf_norm_abs_mean)).item()

    indices_selected = sorted_indices[:int((1 - malicious_ratio) * client_num)]
    
    global_update = dict()
    for name in model_updates.keys():
        global_update[name] = model_updates[name][indices_selected].float().mean(dim=0)

    return global_update, eta, l2_norm_global, linf_norm_abs_mean, b.item()


def jza(model_updates, l2_old, linf_old, b_old, current_round):

    # pdb.set_trace()
    weight = args.weight
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]

    # SHPRG init
    initialization_values_filename = "./FL_Backdoor_CV/shprg/initialization_values"
    m, cnt = last_layer_updates.shape[1], last_layer_updates.shape[0]
    n, _, p, q = load_initialization_values(initialization_values_filename)
    filename = "matrix"+'_'+str(m)
    shprg = SHPRG(n, m, p, q, filename)
    seeds = shprg.generate_seeds(cnt)
    max_mask = weight * linf_old
    
    # SHPRG generation
    vector = torch.tensor([shprg.generate(seed, m, max_mask) for seed in seeds]).cuda()
    masked_updates = last_layer_updates + vector
    server_sum_vector = torch.tensor(shprg.client_sum_hprg(seeds, m, max_mask)).cuda()

    # select
    l2_norm = torch.norm(masked_updates, dim=1)
    sorted_l2_norm, sorted_indices = torch.sort(l2_norm)
    
    if current_round <= 2 or (args.resume and current_round <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        b = sorted_l2_norm[int(0.5 * cnt)]
    else:
        b = l2_old[1] / l2_old[0] * b_old
    
    benign_index = torch.searchsorted(sorted_l2_norm, b).item()
    # benign_index = max(1, min(int(0.8 * cnt), benign_index))
    benign_index = max(int(0.5 * cnt), min(int(0.8 * cnt), benign_index))
    indices_selected = sorted_indices[:benign_index]

    global_update = dict()
    for name in model_updates.keys():
        global_update[name] = model_updates[name][indices_selected].float().mean(dim=0)
    global_update[keys[-2]] = masked_updates[indices_selected].float().mean(dim=0)

    # computation for next round
    last_layer_global_update = global_update[keys[-2]]
    l2_norm_global_new = torch.norm(last_layer_global_update).item()
    if current_round <= 2 or (args.resume and current_round <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        l2_old.append(l2_norm_global_new)
        l2_norm_global = l2_old
    else:
        l2_norm_global = [l2_old[1], l2_norm_global_new]
    linf_norm_global = torch.norm(last_layer_global_update, p=float('inf')).item()
    
    return global_update, l2_norm_global, linf_norm_global, b, benign_index


def zero_mask(model_updates, l2_old, b_old, current_round):

    # pdb.set_trace()
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]
    cnt = last_layer_updates.shape[0]

    # select
    l2_norm = torch.norm(last_layer_updates, dim=1)
    sorted_l2_norm, sorted_indices = torch.sort(l2_norm)
    
    if current_round <= 2 or (args.resume and current_round \
                              <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        b = sorted_l2_norm[int(0.5 * cnt)]
    else:
        b = l2_old[1] / l2_old[0] * b_old
    
    benign_index = torch.searchsorted(sorted_l2_norm, b).item()
    benign_index = max(int(0.5 * cnt), min(int(0.8 * cnt), benign_index))
    indices_selected = sorted_indices[:benign_index]

    global_update = dict()
    for name in model_updates.keys():
        global_update[name] = model_updates[name][indices_selected].float().mean(dim=0)

    # computation for next round
    last_layer_global_update = global_update[keys[-2]]
    l2_norm_global_new = torch.norm(last_layer_global_update).item()
    if current_round <= 2 or (args.resume and current_round \
                              <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        l2_old.append(l2_norm_global_new)
        l2_norm_global = l2_old
    else:
        l2_norm_global = [l2_old[1], l2_norm_global_new]
    
    return global_update, l2_norm_global, b, benign_index


def jzx_no_defense(model_updates, l2_old, linf_old, linf_shprg_old, b_old, current_round):

    # pdb.set_trace()
    weight = args.weight
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]

    # SHPRG init
    initialization_values_filename = "./FL_Backdoor_CV/shprg/initialization_values"
    m, cnt = last_layer_updates.shape[1], last_layer_updates.shape[0]
    n, _, p, q = load_initialization_values(initialization_values_filename)
    filename = "matrix"+'_'+str(m)
    shprg = SHPRG(n, m, p, q, filename)
    seeds = shprg.generate_seeds(cnt)
    max_mask = weight * linf_old
    
    # SHPRG generation
    vector = torch.tensor([shprg.generate(seed, m, max_mask) for seed in seeds]).cuda()
    masked_updates = last_layer_updates + vector
    server_sum_vector = torch.tensor(shprg.client_sum_hprg(seeds, m, max_mask)).cuda()
    # server_sum_vector = torch.tensor(shprg.server_sum_hprg(seeds, m, max_mask)).cuda()
    linf_norm_shprg = torch.norm(server_sum_vector, p=float('inf')).item()

    # select
    l2_norm = torch.norm(masked_updates, dim=1)
    sorted_l2_norm, sorted_indices = torch.sort(l2_norm)
    
    if current_round <= 2 or (args.resume and current_round <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        b = sorted_l2_norm[int(0.2 * cnt)]
    else:
        b = (l2_old[1] + linf_norm_shprg) / (l2_old[0] + linf_shprg_old) * b_old
    
    benign_index = torch.searchsorted(sorted_l2_norm, b).item()
    benign_index = max(int(0.2 * cnt), min(int(0.8 * cnt), benign_index))
    indices_selected = sorted_indices[:benign_index]

    global_update = dict()
    for name in model_updates.keys():
        global_update[name] = model_updates[name].float().mean(dim=0)

    # computation for next round
    last_layer_global_update = global_update[keys[-2]]
    l2_norm_global_new = torch.norm(last_layer_global_update).item()
    if current_round <= 2 or (args.resume and current_round <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        l2_old.append(l2_norm_global_new)
        l2_norm_global = l2_old
    else:
        l2_norm_global = [l2_old[1], l2_norm_global_new]
    linf_norm_global = torch.norm(last_layer_global_update, p=float('inf')).item()
    
    return global_update, l2_norm_global, linf_norm_global, linf_norm_shprg, b, benign_index


def jzx_test(model_updates, l2_old, linf_old, linf_shprg_old, b_old, current_round):

    # pdb.set_trace()
    keys = list(model_updates.keys())
    last_layer_updates = model_updates[keys[-2]]

    # SHPRG init
    initialization_values_filename = "./FL_Backdoor_CV/shprg/initialization_values"
    m, cnt = last_layer_updates.shape[1], last_layer_updates.shape[0]
    n, _, p, q = load_initialization_values(initialization_values_filename)
    filename = "matrix"+'_'+str(m)
    shprg = SHPRG(n, m, p, q, filename)
    seeds = shprg.generate_seeds(cnt)
    max_mask = args.weight * linf_old                                                                                                                           
    
    # SHPRG generation
    vector = torch.tensor([shprg.generate(seed, m, max_mask) for seed in seeds]).cuda()
    masked_updates = last_layer_updates + vector
    server_sum_vector = torch.tensor(shprg.client_sum_hprg(seeds, m, max_mask)).cuda()
    # server_sum_vector = torch.tensor(shprg.server_sum_hprg(seeds, m, max_mask)).cuda()
    linf_norm_shprg = torch.norm(server_sum_vector, p=float('inf')).item()

    # select
    l2_norm = torch.norm(masked_updates, dim=1)
    sorted_l2_norm, sorted_indices = torch.sort(l2_norm)
    
    if current_round <= 3 or (args.resume and current_round <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 3):
        b = sorted_l2_norm[int(args.min_threshold * cnt)]
    else:
        b = (l2_old[1] + linf_norm_shprg) / (l2_old[0] + linf_shprg_old) * b_old
    
    benign_index = torch.searchsorted(sorted_l2_norm, b).item()
    benign_index = max(int(args.min_threshold * cnt), min(int(0.8 * cnt), benign_index))
    indices_selected = sorted_indices[:benign_index]

    global_update = dict()
    for name in model_updates.keys():
        global_update[name] = model_updates[name][indices_selected].float().mean(dim=0)
    # global_update[keys[-2]] = masked_updates[indices_selected].float().mean(dim=0)

    # computation for next round
    last_layer_global_update = global_update[keys[-2]]
    l2_norm_global_new = torch.norm(last_layer_global_update).item()
    if current_round <= 2 or (args.resume and current_round <= int(re.findall(r'\d+\d*', args.resumed_name.split('/')[1])[0]) + 2):
        l2_old.append(l2_norm_global_new)
        l2_norm_global = l2_old
    else:
        l2_norm_global = [l2_old[1], l2_norm_global_new]
    linf_norm_global = torch.norm(last_layer_global_update, p=float('inf')).item()
    
    return global_update, l2_norm_global, linf_norm_global, linf_norm_shprg, b, benign_index