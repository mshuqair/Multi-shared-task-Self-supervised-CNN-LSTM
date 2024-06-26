import pickle
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc
from sklearn.metrics import r2_score


# The code is based on code from:
# https://apmonitor.com/che263/index.php/Main/PythonRegressionStatistics


def f(x, a, b):
    return a * x + b


def predband(x, xd, yd, p, func, conf=0.95):
    # x = requested points
    # xd = x data
    # yd = y data
    # p = parameters
    # func = function name
    alpha = 1.0 - conf  # significance
    N = xd.size  # data sample size
    var_n = len(p)  # number of parameters
    # Quantile of Student's t distribution for p=(1-alpha/2)
    q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
    # Stdev of an individual measurement
    se = np.sqrt(1. / (N - var_n) * np.sum((yd - func(xd, *p)) ** 2))
    # Auxiliary definitions
    sx = (x - xd.mean()) ** 2
    sxd = np.sum((xd - xd.mean()) ** 2)
    # Predicted values (best-fit model)
    yp = func(x, *p)
    # Prediction band
    dy = q * se * np.sqrt(1.0 + (1.0 / N) + (sx / sxd))
    # Upper & lower prediction bands.
    lpb, upb = yp - dy, yp + dy
    return lpb, upb


# Analyze the results using the saved pickle file
# import data
font_size = 'medium'
save_figure = True
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

x = y_rounds
y = y_predicted_rounds
n = len(y)

popt, pcov = curve_fit(f, x, y)

# retrieve parameter values
a = popt[0]
b = popt[1]
print('Optimal Values')
print('a: ' + str(a))
print('b: ' + str(b))

# compute r^2
r2 = r2_score(y_true=x, y_pred=y)
print('R^2: ' + str(r2))

# calculate parameter confidence interval
a, b = unc.correlated_values(popt, pcov)
print('Uncertainty')
print('a: ' + str(a))
print('b: ' + str(b))

# plot data
plt.figure(figsize=(4, 4))
# plt.scatter(x, y, s=25, c='tab:blue', label='Data')
plt.scatter(x, y, s=20, c='tab:blue')

# calculate regression confidence interval
px = np.linspace(start=0, stop=np.max(x), num=100)
py = a * px + b
nom = unp.nominal_values(py)
std = unp.std_devs(py)

lpb, upb = predband(px, x, y, popt, f, conf=0.95)

# plot the regression
plt.plot(px, nom, c='black', label='y=a x + b')
plt.plot(px, nom, c='black')


# uncertainty lines (95% confidence)
plt.plot(px, nom - 1.96 * std, c='tab:red', label='95% Confidence Region')
plt.plot(px, nom + 1.96 * std, c='tab:red')
# prediction band (95% confidence)
plt.plot(px, lpb, 'k--', label='95% Prediction Band')
plt.plot(px, upb, 'k--')
plt.ylim(-1, 61)
plt.yticks(fontsize=font_size)
plt.ylabel('Estimated UPDRS-III', fontsize='large')
plt.xlim(-1, 61)
plt.xticks(fontsize=font_size)
plt.xlabel('Clinical UPDRS-III', fontsize='large')
plt.legend(loc='upper left', fontsize='small')
plt.tight_layout()
if save_figure:
    plt.savefig('./figures/' + FileName + '_correlation.png')
plt.show()
