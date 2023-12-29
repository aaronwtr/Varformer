import torch
import random

import torch.nn.functional as F
import numpy as np

import model as m
import loss as l
from utils import random_seed_context

from matplotlib import pyplot as plt


def training(train, val, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P = torch.tensor([i for i, labels in enumerate(train.dataset.labels) if labels == 1]).to(device)
    U = torch.tensor([i for i, labels in enumerate(train.dataset.labels) if labels == 0]).to(device)
    L = torch.tensor([]).to(device)

    K = 2
    T = 1000
    t_l = 0.05
    t_u = 0.35

    models = [m.PyTorchMLP(config=config, num_features=train.dataset.data.shape[1], model_type="puupl").to(device) for _
              in
              range(K)]

    seeds = generate_seeds(K)
    weights = []
    for i, model in enumerate(models):
        weight = model.initialise_weights(seed=seeds[i])
        weights.append(weight)

    weight_decay = float(config['puupl'].get('weight_decay', 0))
    optimizers = [torch.optim.Adam(model.parameters(), lr=float(config['puupl']['lr_start']),
                                   weight_decay=weight_decay) for model in models]
    criterion = l.PseudoLabelLoss()

    converged = False
    val_losses = []
    pseudo_labels = []
    metrics_data = {
        'train_loss_model1': [],
        'val_loss_model1': [],
        'train_auroc_model1': [],
        'val_auroc_model1': [],
        'train_precision_model1': [],
        'val_precision_model1': [],
    }
    epoch = 1
    while not converged:
        print("Epoch: ", epoch)
        for i, model in enumerate(models):
            model.load_state_dict(weights[i])
            train_loss, _, auroc, precision, _ = train_model(models[i], device, optimizers[i], train, criterion, P, U,
                                                             L, pseudo_labels)
            print(f"Train loss model {i}: ", train_loss)
            print(f"Train auroc model {i}: ", float(auroc))
            print(f"Train precision model {i}: ", float(precision))

            if i == 0:
                metrics_data['train_loss_model1'].append(float(train_loss))
                metrics_data['train_auroc_model1'].append(float(auroc))
                metrics_data['train_precision_model1'].append(float(precision))

        val_losses = update_ensemble_weights(models, val, criterion, val_losses, P, U, L, pseudo_labels)
        metrics_data['val_loss_model1'].append(val_losses[-2])
        metrics_data['val_auroc_model1'].append(val_losses[-2])
        metrics_data['val_precision_model1'].append(val_losses[-2])

        # if epoch % 10 == 0:
        #     for metric_name, metric_values in metrics_data.items():
        #         plt.figure(figsize=(10, 6))
        #         moving_averages = moving_average(metric_values, window=10)
        #         plt.plot(moving_averages)
        #         plt.title(f'{metric_name} Moving Average')
        #         plt.xlabel('Epoch')
        #         plt.ylabel(metric_name)
        #         plt.show()

        unlabelled_data = train.dataset.features[U]
        new_labeled_examples, new_unlabeled_examples, pseudo_labels = pseudo_label(models, epoch, device,
                                                                                   unlabelled_data, t_l, t_u, T)

        L = torch.cat((L, new_labeled_examples), dim=0)
        L = L.to(torch.int64)
        U = torch.cat((U, new_unlabeled_examples), dim=0)
        U = torch.tensor([i for i in U if i not in L])
        U = U.to(torch.int64)

        if has_converged(val_losses, threshold=0.001):
            # TODO: Check how this progresses
            break

        epoch += 1

    return models


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas


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


def train_model(model, device, optimizer, train, criterion, P, U, L, pseudo_labels):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_auroc = 0.0
    total_precision = 0.0
    total_spearman = 0.0
    num_batches = 0

    for batch_idx, (data, labels) in enumerate(train):
        data, labels = data.to(device), labels.to(device)
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
        total_precision += model.precision(bin_preds, labels)
        total_spearman += model.spearman(probas, labels.float())
        num_batches += 1

    # model.eval()
    # with torch.no_grad():
    #     X_train = train.dataset.features
    #     y_train = train.dataset.labels
    #     logits, probas, _ = model(X_train)
    #
    # plt.figure(figsize=(10, 6))
    # plt.scatter(range(len(X_train[:, 0])), X_train[:, 0], c=y_train, cmap='viridis')
    # plt.colorbar(label='Predicted Probability')
    # plt.title(f'Epoch {num_batches + 1}')
    # plt.show()

    loss = total_loss / num_batches
    acc = total_acc / num_batches
    auroc = total_auroc / num_batches
    precision = total_precision / num_batches
    spearman = total_spearman / num_batches

    return loss, acc, auroc, precision, spearman


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


def pseudo_label(models, epoch, device, X_U, t_l, t_u, T):
    X_U = X_U.to(device)
    logits = [model(X_U)[0] for model in models]
    probs = [F.sigmoid(logit) for logit in logits]

    stacked = torch.stack(probs)
    probs_avg = torch.mean(stacked, dim=0)
    # plot the distribution of probs_avg

    probs_avg_np = probs_avg.detach().numpy()  # Convert tensor to numpy array
    fig, ax = plt.subplots(dpi=300)
    ax.hist(probs_avg_np, bins=50, edgecolor='black')
    ax.set_title(f'Distribution of average probabilities across models after epoch {epoch}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    plt.show()

    max_avg = torch.max(probs_avg[0])     # Convert tensor to numpy array

    aleatoric = -1 / len(probs) * (torch.sum(stacked * torch.log(stacked), dim=0) +
                                   torch.sum((1 - stacked) * torch.log(1 - stacked), dim=0))
    total = -probs_avg * torch.log(probs_avg) - (1 - probs_avg) * torch.log(1 - probs_avg)
    epistemic = total - aleatoric

    # Rank by epistemic uncertainty
    sorted_indices = torch.argsort(epistemic)

    # Take most confident T examples
    confident_indices = sorted_indices[:T]
    L_new = [i for i in confident_indices if epistemic[i] <= t_l]

    # Balance positive/negative
    L_new_pos = [i for i in L_new if probs_avg[i] > 0.17]
    # L_new_neg = [i for i in L_new if probs_avg[i] <= 0.5]
    L_new_neg = []
    L_new = L_new_pos + L_new_neg
    L_new = torch.tensor(L_new)

    if len(L_new) == 0:
        soft_labels = torch.tensor([])
    else:
        soft_labels = probs_avg[L_new]

    unreliable_indices = [i for i in L_new if epistemic[i] >= t_u]
    U_new = torch.tensor(unreliable_indices)

    L_new = torch.tensor(L_new).to(device)
    U_new = torch.tensor(U_new).to(device)
    soft_labels = soft_labels.to(device)

    return L_new, U_new, soft_labels


def update_ensemble_weights(models, val, criterion, val_losses, P, U, L, pseudo_labels):
    for i, model in enumerate(models):
        val_loss, _, _, _ = evaluate(model, val, criterion, P, U, L, pseudo_labels)
        print(f'Val loss model {i}: ', val_loss)
        val_losses.append(val_loss)

    best_model_idx = val_losses.index(min(val_losses))

    if best_model_idx != len(val_losses) - 1 or best_model_idx != len(val_losses) - 2:
        return val_losses
    else:
        best_model_weights = models[best_model_idx].state_dict()
        torch.save(best_model_weights, 'best_model.ckpt')
        return val_losses


def evaluate(model, val, criterion, P, U, L, pseudo_labels):
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
