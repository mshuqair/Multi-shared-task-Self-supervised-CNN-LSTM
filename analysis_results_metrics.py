import pickle
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


# Analyze the results using the saved pickle file
# import data
# For M-SSL CNN-LSTM model
print('Regression metrics for the M-SSL CNN-LSTM:')
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

MAE = mean_absolute_error(y_true=y_rounds, y_pred=y_predicted_rounds)
r_2 = r2_score(y_true=y_rounds, y_pred=y_predicted_rounds)
corr = pearsonr(x=y_rounds, y=y_predicted_rounds)
print('MAE %.2f, R2 score %.2f, Correlation coefficient %.2f (p=%.4f)'
      % (MAE, r_2, corr[0], corr[1]))


# For the supervised CNN-LSTM model
print('\nRegression metrics for the Supervised CNN-LSTM:')
FileName = 'task_baseline_dual_cnn_lstm'
with open('./results/' + FileName + '/model_predictions.pkl', 'rb') as file:
    [y, y_predicted, pNo, roundNo] = pickle.load(file)

y_predicted_rounds = np.array([])
for roundI in (np.unique(roundNo)):
    y_predicted_temp = y_predicted[roundNo == roundI]
    y_predicted_temp = np.rint(np.mean(y_predicted_temp).clip(min=0))
    y_predicted_rounds = np.append(y_predicted_rounds, y_predicted_temp, axis=None)

MAE = mean_absolute_error(y_true=y_rounds, y_pred=y_predicted_rounds)
r_2 = r2_score(y_true=y_rounds, y_pred=y_predicted_rounds)
corr = pearsonr(x=y_rounds, y=y_predicted_rounds)
print('MAE %.2f, R2 score %.2f, Correlation coefficient %.2f (p=%.4f)'
      % (MAE, r_2, corr[0], corr[1]))
