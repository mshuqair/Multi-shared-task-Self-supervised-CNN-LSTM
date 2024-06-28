import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Analyze the results using the saved pickle file
save_figure = True

# import data
# The baseline model
FileName = 'task_baseline_dual_cnn_lstm'
with open('./results/' + FileName + '/model_predictions.pkl', 'rb') as file:
    [y, y_predicted, pNo, roundNo] = pickle.load(file)
y_rounds = np.array([])
y_predicted_rounds = np.array([])
pNo_rounds = np.array([])
for roundI in (np.unique(roundNo)):
    y_rounds = np.append(y_rounds, np.mean(y[roundNo == roundI]), axis=None)
    y_predicted_temp = y_predicted[roundNo == roundI]
    y_predicted_temp = np.rint(np.mean(y_predicted_temp).clip(min=0))
    y_predicted_rounds = np.append(y_predicted_rounds, y_predicted_temp, axis=None)
    pNo_rounds = np.append(pNo_rounds, np.unique(pNo[roundNo == roundI]))
y_cnn = y_predicted_rounds

UPDRS_scores = np.unique(y_rounds)
MAE_cnn = np.zeros(shape=np.unique(y_rounds).shape[0])
index = 0
for score in np.unique(y_rounds):
    y_true = y_rounds[y_rounds == score]
    y_predicted = y_cnn[y_rounds == score]
    MAE_cnn[index] = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
    index = index + 1

# The SSL model
FileName = 'task_downstream_dual_cnn_lstm'
with open('./results/' + FileName + '/model_predictions.pkl', 'rb') as file:
    [y, y_predicted, pNo, roundNo] = pickle.load(file)
y_rounds = np.array([])
y_predicted_rounds = np.array([])
pNo_rounds = np.array([])
for roundI in (np.unique(roundNo)):
    y_rounds = np.append(y_rounds, np.mean(y[roundNo == roundI]), axis=None)
    y_predicted_temp = y_predicted[roundNo == roundI]
    y_predicted_temp = np.rint(np.mean(y_predicted_temp).clip(min=0))
    y_predicted_rounds = np.append(y_predicted_rounds, y_predicted_temp, axis=None)
    pNo_rounds = np.append(pNo_rounds, np.unique(pNo[roundNo == roundI]))
y_ssl = y_predicted_rounds

MAE_ssl = np.zeros(shape=np.unique(y_rounds).shape[0])
index = 0
for score in np.unique(y_rounds):
    y_true = y_rounds[y_rounds == score]
    y_predicted = y_ssl[y_rounds == score]
    MAE_ssl[index] = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
    index = index + 1


# bar plot
mae_dict = {
    'M-SSL': MAE_ssl,
    'Supervised': MAE_cnn}
width = 0.75  # the width of the bars
multiplier = 0
z_order = 3
c = 0
colors = ['tab:blue', 'tab:olive']
fig, ax = plt.subplots(figsize=(6.5, 3))
for attribute, measurement in mae_dict.items():
    offset = width * multiplier
    rects = ax.bar(UPDRS_scores + offset, measurement, width, label=attribute, color=colors[c], zorder=z_order)
    multiplier += 0.75
    z_order -= 1
    c += 1
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5, zorder=0)
ax.legend(loc='upper left', ncols=1, fontsize='small')
ax.set_ylim(0, 21)
ax.set_yticks(ticks=np.arange(start=0, step=5, stop=25), labels=np.arange(start=0, step=5, stop=25))
ax.set_ylabel('Mean absolute error', fontsize='large')
ax.set_xlim((-1, 64))
ax.set_xticks(ticks=np.arange(start=0, step=10, stop=70), labels=np.arange(start=0, step=10, stop=70))
ax.set_xlabel('UPDRS-III scores', fontsize='large')
plt.tight_layout()
if save_figure:
    plt.savefig('./figures/mae_bars.png')
plt.show()


# box plot
fig, ax = plt.subplots(figsize=(3.5, 3))
box_plot = ax.boxplot([MAE_ssl, MAE_cnn], tick_labels=['M-SSL', 'Supervised'], widths=0.30,
                      patch_artist=True, zorder=3, showfliers=False, showmeans=False, meanline=False)
ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5, zorder=0)
ax.set_ylim((-1, 21))
ax.set_yticks(ticks=np.arange(start=0, stop=25, step=2.5), labels=np.arange(start=0, stop=25, step=2.5))
ax.set_ylabel('Mean absolute error', fontsize='large')
colors = ['tab:blue', 'tab:olive']  # fill with colors
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
plt.tight_layout()
if save_figure:
    plt.savefig('./figures/mae_box.png')
plt.show()
