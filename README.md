# Multi-shared-task Self-supervised (M-SSL) Multichannel CNN-LSTM

This code is for the proposed Multi-shared-task Self-supervised CNN-LSTM network to estimate UPDRS-III scores of PD patients.
The research associated with this code is under revision at the MDPI Bioengineering journal.



## M-SSL Multichannel CNN-LSTM for UPDRS-III Estimation in PD Patients
![](figures/figure_main.png)
**Figure 1.** The main algorithm for estimating UPDRS-III scores.


## General Note
- The code treats the estimation of UPDRS-III scores as a regression problem. If you want to use the model as a classifier, you only need to alter the model's output layer and loss function.
- The code performs a leave-one-out subject-wise testing. You can replace the folds with the desired training and testing data. 


## Code Requirements and Compatability
The code was run and tested using the following:
- Python		3.10.11
- tensorflow	2.10.1
- keras			2.10.0
- h5py			3.10.0
- matplotlib	3.9.0
- numpy			1.26.3
- pandas		2.1.4
- scikit-learn	1.5.0
- scipy			1.13.1
- transforms3d	0.4.1


## 

