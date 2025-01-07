import time
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, auc

from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6, 5))

"""
Note: This code is for multi-class classification, not binary classification.
"""


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def plot_roc_curve(fpr_dict, tpr_dict, roc_auc, num_classes, class_names, title, save_path):
    plt.figure(figsize=(6, 5.5), dpi=300)
    macro_average = 0
    for i in range(num_classes):
        plt.plot(fpr_dict[i], tpr_dict[i], lw=2, label=f"{class_names[i]} (AUC = {round(roc_auc[i], 3)})")
        macro_average += roc_auc[i]
    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
    mean_tpr /= num_classes
    fpr_dict["macro"] = all_fpr
    tpr_dict["macro"] = mean_tpr
    # 2 ways of calculating macro-average AUC
    # roc_auc["macro"] = auc(fpr_dict["macro"], tpr_dict["macro"])
    roc_auc["macro"] = macro_average / num_classes
    plt.plot(fpr_dict["macro"], tpr_dict["macro"], lw=2, label=f"Macro-average (AUC = {round(roc_auc['macro'], 3)})",
             color='purple', linestyle='--')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')  # False Positive Rate
    plt.ylabel('Sensitivity')  # True Positive Rate
    plt.title('ROC')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'ROC curve saved at {save_path}')
    return roc_auc['macro']


# patient-level AUC, make sure one name with only one label
def val_auc(model, loader, criterion, class_names, save_path, iter_idx, save_as_pdf=False):
    print(f'Validating: iter {iter_idx} ...')
    tic = time.time()
    num_classes = len(class_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    name_list, y_list, pred_y_prob_list = [], [], []
    result = pd.DataFrame(columns=['name', 'y', 'pred_y', 'final_y'])

    model.eval()
    total_loss = 0
    with torch.no_grad():
        for name, X, y in loader:
            X = X.to(device)
            y = y.to(device)

            output = model(X)

            pred_y_prob = torch.softmax(output, dim=-1)

            loss = criterion(output, y)
            total_loss += loss.item()
            pred_y_prob = pred_y_prob.detach().cpu().numpy()

            y = y.cpu().numpy()

            y_list.append(y)
            pred_y_prob_list.append(pred_y_prob)
            name_list.append(name)

    # Flatten lists
    name_list = np.concatenate(name_list, axis=0)
    y_list = np.concatenate(y_list, axis=0)
    pred_y_prob_list = np.concatenate(pred_y_prob_list, axis=0)

    for i in range(num_classes):
        result[f'pred_y_prob_{i}'] = pred_y_prob_list[:, i]

    result['name'] = name_list
    result['y'] = y_list

    # Per-patient prediction, the result should be patient-level
    pred_prob_list_name = [f'pred_y_prob_{i}' for i in range(num_classes)]
    per_patient_result = result.groupby('name')[pred_prob_list_name].mean()  # Soft voting, use name column
    per_patient_result["y_pred"] = per_patient_result[pred_prob_list_name].idxmax(axis=1).str[-1].astype(int)

    per_patient_result = per_patient_result.merge(result[['name', 'y']], on='name', how='left').drop_duplicates()
    per_patient_result = per_patient_result.reset_index(drop=True)

    # ROC curve
    fpr_dict, tpr_dict, roc_auc = {}, {}, {}
    for i in range(num_classes):
        true_y_i = np.where(per_patient_result['y'] == i, 1, 0)
        proba_y_i = per_patient_result[f"pred_y_prob_{i}"]
        fpr_dict[i], tpr_dict[i], _ = roc_curve(true_y_i, proba_y_i)
        roc_auc[i] = auc(fpr_dict[i], tpr_dict[i])

    macro_roc_path = os.path.join(save_path, f'slide_{iter_idx}.png')
    if save_as_pdf:
        macro_roc_path = os.path.join(save_path, f'slide_{iter_idx}.pdf')
    slide_auc = plot_roc_curve(fpr_dict, tpr_dict, roc_auc, num_classes, class_names, 'Slide-level ROC', macro_roc_path)

    print(f'Validation time: {time.time() - tic:.2f}s')
    return slide_auc
