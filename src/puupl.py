import torch
import random

import torch.nn.functional as F

import model as m
import loss as l
from utils import random_seed_context


def training(train, val, config):
    # TODO:
    #  - 1: Define P, U, L on batch level. Then we need a way to point batch level to the global level.
    #  - 2: Try to run training without batches.
    P = torch.tensor([i for i, labels in enumerate(train.dataset.labels) if labels == 1])
    U = torch.tensor([i for i, labels in enumerate(train.dataset.labels) if labels == 0])
    L = torch.tensor([])

    K = 2
    T = 1000
    t_l = 0.05
    t_u = 0.35

    models = [m.PyTorchMLP(config=config, num_features=train.dataset.data.shape[1]) for _ in range(K)]

    seeds = generate_seeds(K)
    weights = []
    for i, model in enumerate(models):
        weight = model.initialise_weights(seed=seeds[i])
        weights.append(weight)

    optimizers = [torch.optim.Adam(model.parameters()) for model in models]
    criterion = l.PseudoLabelLoss()

    converged = False
    val_losses = []
    pseudo_labels = []

    while not converged:
        for i, model in enumerate(models):
            model.load_state_dict(weights[i])
            train_model(models[i], optimizers[i], train, criterion, P, U, L, pseudo_labels)

        val_losses = update_ensemble_weights(models, val, criterion, val_losses, P, U, L)

        unlabelled_data = train.dataset.features[U]
        new_labeled_examples, new_unlabeled_examples, pseudo_labels = pseudo_label(models, unlabelled_data, t_l, t_u, T)

        L = torch.cat((L, new_labeled_examples), dim=0)
        U = new_unlabeled_examples

        if has_converged(val_losses, threshold=0.001):
            # TODO: Check how this progresses
            break

    return models


def batching_labels(data, batch_idx, P, U, L):
    P_batch = []
    U_batch = []
    L_batch = []
    for i in range(len(data)):
        if i * (batch_idx + 1) in P:
            P_batch.append(i)
        elif i * (batch_idx + 1) in U:
            U_batch.append(i)
        elif i * (batch_idx + 1) in L:
            L_batch.append(i)
    P_batch = torch.tensor(P_batch)
    U_batch = torch.tensor(U_batch)
    L_batch = torch.tensor(L_batch)
    return P_batch, U_batch, L_batch


def train_model(model, optimizer, train, criterion, P, U, L, pseudo_labels):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_auroc = 0.0
    total_spearman = 0.0
    num_batches = 0

    for batch_idx, (data, labels) in enumerate(train):
        P_batch, U_batch, L_batch = batching_labels(data, batch_idx, P, U, L)

        logits, probas, bin_preds = model(data)
        loss = criterion(logits, P=P_batch, U=U_batch, L=L_batch, pseudo_labels=pseudo_labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += model.acc(bin_preds, labels)
        total_auroc += model.auroc(bin_preds, labels)
        total_spearman += model.spearman(probas, labels.float())
        num_batches += 1

    loss = total_loss / num_batches
    acc = total_acc / num_batches
    auroc = total_auroc / num_batches
    spearman = total_spearman / num_batches

    return loss, acc, auroc, spearman


def generate_seeds(num_seeds):
    seed_value = 42
    with random_seed_context(seed_value):
        # do this to make sure the seed is only set locally
        seed_list = [random.randint(1, 1000) for _ in range(num_seeds)]
    return seed_list


def has_converged(val_losses, threshold=0.001):
    val_loss = min(val_losses)
    min_idx = val_losses.index(val_loss)
    prev_losses = val_losses[:min_idx] + val_losses[min_idx + 1:]

    loss_change = (val_loss - prev_losses[-1]) / prev_losses[-1]
    if abs(loss_change) <= threshold:
        return True
    else:
        return False


def pseudo_label(models, X_U, t_l, t_u, T):
    logits = [model(X_U) for model in models]
    probs = [F.softmax(logit, dim=1)[:, 1] for logit in logits]

    stacked = torch.stack(probs)
    probs_avg = torch.mean(stacked, dim=0)

    # Compute uncertainties. This is a tensor with an uncertainty across models for every entry in the output
    aleatoric = -1 / len(probs) * torch.sum(torch.sum(stacked * torch.log(stacked), dim=0) +
                                            torch.sum((1 - stacked) * torch.log(1 - stacked), dim=0))
    total = -probs_avg * torch.log(probs_avg) - (1 - probs_avg) * torch.log(1 - probs_avg)
    epistemic = total - aleatoric

    # Rank by epistemic uncertainty
    sorted_indices = torch.argsort(epistemic)

    # Take most confident T examples
    confident_indices = sorted_indices[:T]
    L_new = [i for i in confident_indices if epistemic[i] <= t_l]

    # Balance positive/negative
    L_new_pos = [i for i in L_new if probs_avg[i] > 0.5]
    L_new_neg = [i for i in L_new if probs_avg[i] <= 0.5]

    L_new = L_new_pos + L_new_neg
    L_new = torch.tensor(L_new)

    soft_labels = probs_avg[L_new]

    unreliable_indices = [i for i in L_new if epistemic[i] >= t_u]
    U_new = torch.tensor(unreliable_indices)

    return L_new, U_new, soft_labels


def update_ensemble_weights(models, val, criterion, val_losses, P, U, L):
    for model in models:
        val_loss = evaluate(model, val, criterion, P, U, L)
        val_losses.append(val_loss)

    best_model_idx = val_losses.index(min(val_losses))

    if best_model_idx != len(val_losses) - 1 or best_model_idx != len(val_losses) - 2:
        return val_losses
    else:
        best_model_weights = models[best_model_idx].state_dict()
        torch.save(best_model_weights, 'best_model.ckpt')
        return val_losses


def evaluate(model, val, criterion, P, U, L):
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        total_acc = 0.0
        total_auroc = 0.0
        total_spearman = 0.0
        num_batches = 0

        for batch_idx, (data, labels) in enumerate(val):
            P_batch, U_batch, L_batch = batching_labels(data, batch_idx, P, U, L)
            logits, probas, bin_preds = model(data)

            loss = criterion(logits, P=P_batch, U=U_batch, L=L_batch, pseudo_labels=labels)

            total_loss += loss.item()
            total_loss += loss.item()
            total_acc += model.acc(bin_preds, labels)
            total_auroc += model.auroc(bin_preds, labels)
            total_spearman += model.spearman(probas, labels.float())
            num_batches += 1

    loss = total_loss / num_batches
    acc = total_acc / num_batches
    auroc = total_auroc / num_batches
    spearman = total_spearman / num_batches

    return loss, acc, auroc, spearman
