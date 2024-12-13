from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import confusion_matrix
import numpy as np


def metric(TP, TN, FP, FN, ln, alpha=None, beta=None, cond=False):
    if cond:
        TN /= ln ** 1
        FP /= ln ** alpha
        FN /= ln ** beta

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = sensitivity
    f_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    fdr = FP / (FP + TP) if (FP + TP) > 0 else 0

    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f_measure': f_measure,
        'accuracy': accuracy,
        'mcc': mcc,
        'fpr': fpr,
        'fnr': fnr,
        'npv': npv,
        'fdr': fdr
    }

    metrics1 = [accuracy, precision, sensitivity, specificity, f_measure, mcc, npv, fpr, fnr]
    return metrics1


def multi_confu_matrix(Y_test, Y_pred, *args):
    cm = mcm(Y_test, Y_pred)
    ln = len(cm)
    TN = cm[:, 0, 0].sum()
    FP = cm[:, 0, 1].sum()
    FN = cm[:, 1, 0].sum()
    TP = cm[:, 1, 1].sum()
    return metric(TP, TN, FP, FN, ln, *args)


def confu_matrix(Y_test, Y_pred, *args):
    cm = confusion_matrix(Y_test, Y_pred)
    TN, FP, FN, TP = cm.ravel()
    ln = 2  # Since we have a 2x2 matrix for binary classification
    return metric(TP, TN, FP, FN, ln, *args)
