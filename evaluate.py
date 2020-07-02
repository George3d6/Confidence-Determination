import torch
import numpy as np
import random

_t_quantile = lambda t, q: t.view(-1).kthvalue(1 + round(float(q) * (t.numel() - 1))).values.item()

def evaluate(Yh,Y,Yc):
    quantile = 0.8
    avg_quantile = 0.5
    low_quantiles = 0.5
    if Yc is not None:
        Yc = [x[0] for x in Yc]
        quantileth_conf = np.quantile(np.array(Yc), quantile)
        avg_quantileth_conf = np.quantile(np.array(Yc), avg_quantile)
        low_quantileth_conf = np.quantile(np.array(Yc), low_quantiles)

    acc_vector = []
    weighted_acc_vector = []
    high_q_acc_vector = []
    avg_q_acc_vector = []
    low_q_acc_vector = []

    for i in range(len(Yh)):
        if Yh[i].index(max(Yh[i])) == Y[i]:
            acc_vector.append(1)
        else:
            acc_vector.append(-1)

        if Yc is not None:
            weighted_acc_vector.append(acc_vector[-1]*Yc[i])

            if Yc[i] > quantileth_conf:
                high_q_acc_vector.append(acc_vector[-1])

            if Yc[i] > avg_quantileth_conf:
                avg_q_acc_vector.append(acc_vector[-1])

            if Yc[i] > low_quantiles:
                low_q_acc_vector.append(acc_vector[-1])

    raw_acc = np.mean(acc_vector)
    if Yc is not None:
        high_q_acc = np.mean(high_q_acc_vector)
        avg_q_acc = np.mean(avg_q_acc_vector)
        low_q_acc = np.mean(low_q_acc_vector)
        weighted_acc = np.mean(weighted_acc_vector)

        acc_mean = np.mean([x if x > 0 else 0 for x in acc_vector])
        conf_mean = np.mean([x for x in Yc])

        subset_indexes = [i for i in range(len(acc_vector)) if random.randint(0,4) == 2]
        subset_acc_mean = np.mean([acc_vector[i] if acc_vector[i] > 0 else 0 for i in subset_indexes])
        subset_conf_mean = np.mean([Yc[i] for i in subset_indexes])

        sub_acc_conf = subset_acc_mean/subset_conf_mean
        full_acc_conf = acc_mean/conf_mean
    else:
        high_q_acc = None
        avg_q_acc = None
        low_q_acc = None
        sub_acc_conf = None
        full_acc_conf = None
        weighted_acc = raw_acc

    return {
        '1. Accuracy': raw_acc
        ,'2. High confidence (0.8 quantile) accuracy': high_q_acc
        ,'3. Average confidence (0.5 quantile) accuracy': avg_q_acc
        ,'4. Above worst confidence (0.2 quantile) accuracy': low_q_acc
        ,'5. Subset acc/conf': sub_acc_conf
        ,'6. Full dataset acc/conf':full_acc_conf
        ,'7. Confidence weighted accuracy': weighted_acc
    }
