import numpy as np


def row_wise_f1_score(labels, preds):
    scores = []
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label) + len(pred))
        scores.append(score)
    return scores, np.mean(scores)