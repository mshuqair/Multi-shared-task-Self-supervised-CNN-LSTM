import pickle
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt
import uncertainties.unumpy as unp
import uncertainties as unc
from sklearn.metrics import r2_score


# The code is based on code from:
# https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics


# === Configuration ===
font_size = 'medium'
save_figure = True
file_name = 'task_downstream_dual_cnn_lstm'
results_path = f'./results/{file_name}/model_predictions.pkl'
save_path = f'./figures/{file_name}_correlation.png'


# === Helper Functions ===

def linear_model(x, a, b):
    """Linear function for regression."""
    return a * x + b


def predband(x, xd, yd, p, func, conf=0.95):
    """Compute prediction band for regression."""
    alpha = 1.0 - conf
    N = xd.size
    var_n = len(p)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    yp = func(x, *p)
    dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


# === Load Data ===

with open(results_path, 'rb') as file:
    y, y_predicted, pNo, roundNo = pickle.load(file)

# === Preprocess Round-Wise Predictions ===

y_rounds, y_predicted_rounds, pNo_rounds = [], [], []

for r in np.unique(roundNo):
    y_rounds.append(np.mean(y[roundNo == r]))
    pred = y_predicted[roundNo == r]
    y_predicted_rounds.append(np.rint(np.mean(pred).clip(min=0)))
    pNo_rounds.append(np.unique(pNo[roundNo == r])[0])

x = np.array(y_rounds)
y = np.array(y_predicted_rounds)

# === Curve Fitting ===

popt, pcov = curve_fit(linear_model, x, y)
a, b = popt
print('Optimal Values')
print(f'a: {a:.4f}')
print(f'b: {b:.4f}')

# === Regression Statistics ===

r2 = r2_score(x, y)
print(f'R²: {r2:.4f}')

# Confidence intervals
a_unc, b_unc = unc.correlated_values(popt, pcov)
print('Uncertainty')
print(f'a: {a_unc}')
print(f'b: {b_unc}')

# === Plotting ===

plt.figure(figsize=(4, 4))
plt.scatter(x, y, s=20, c='tab:blue')

# Fit line and confidence bands
px = np.linspace(0, np.max(x), 100)
py = a_unc * px + b_unc
nom = unp.nominal_values(py)
std = unp.std_devs(py)

lpb, upb = predband(px, x, y, popt, linear_model, conf=0.95)

# Plot regression line
plt.plot(px, nom, c='black', label='y = a·x + b')

# 95% confidence interval
plt.plot(px, nom - 1.96 * std, c='tab:red', label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='tab:red')

# 95% prediction band
plt.plot(px, lpb, 'k--', label='95% Prediction Band')
plt.plot(px, upb, 'k--')

# Formatting
plt.xlim(-1, 61)
plt.ylim(-1, 61)
plt.xlabel('Clinical UPDRS-III', fontsize='large')
plt.ylabel('Estimated UPDRS-III', fontsize='large')
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.legend(loc='upper left', fontsize='small')
plt.tight_layout()

if save_figure:
    plt.savefig(save_path)

plt.show()
