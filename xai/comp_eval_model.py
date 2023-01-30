# Step.1 Load Data
# Load the Dataframe with:
#     - n predictors (features)
#     - 1 target


# Step.2 Train Prediction Models

# 2.1 Train Random Forest
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

n_defined_jobs = multiprocessing.cpu_count() - 2
param_grid = {
    'n_estimators': [64, 256, 1024],
    'max_depth': [8, 16, None],
    'criterion': ["gini"]
             }

def trainRFModel():
    clf = RandomForestClassifier()

    rf_model = GridSearchCV(clf, param_grid, cv=5, n_jobs = n_defined_jobs)
    rf_model.fit(X_train, y_train)
    save_model("rf_model", rf_model, X_test, y_test, features)

    y_pred = rf_model.predict_proba(X_test)
    y_pred_classes =  rf_model.predict(X_test)
    model_auc = roc_auc_score(y_test, y_pred[:, 1])
    print(model_auc)
    classificationReport = classification_report(y_test, y_pred_classes)
    print(classificationReport)
    return rf_model

# Step.3 Run Explanation Models
# 3.1 LIME
from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(data_to_be_explained
                                               ,feature_names = features 
                                               ,class_names=['No','Yes']
                                               ,mode='classification'
                                               ,feature_selection= 'lasso_path' 
                                               ,discretize_continuous=True
                                               ,discretizer='quartile'
                                             )

predict_fn = lambda x: rf_model.predict_proba(x).astype(float)

fi_df = pd.DataFrame(0.0, index=np.arange(data_size),columns=features)

import datetime
now = datetime.datetime.now()
print(now)
        
for p in range(0,data_size):
    lime_explanation = lime_explainer.explain_instance(data_to_be_explained[p]
                                          ,predict_fn
                                          ,num_features=10
                                          #,top_labels=1
                                          #,distance_metric='cosine'
                                          #,distance_metric='manhattan'
                                        )
   
    fi_lime = lime_explanation.as_list()
   
    fi_lime = pd.DataFrame(fi_lime, columns=['feature','importance'])
    for i in range(0,len(fi_lime)): 
        splitted = fi_lime.loc[i, 'feature'].split(' ')
        #splitted = fi_lime['feature'][i].split(' ')
        if len(splitted)==3:
            fi_lime.loc[i, 'feature'] = splitted[0]
            #fi_lime['feature'][i] = splitted[0]
        else:
            fi_lime.loc[i, 'feature'] = splitted[2]
            #fi_lime['feature'][i] = splitted[2]
    
    for i in range(0,len(fi_lime)):
        fi_df.loc[p,fi_lime['feature'][i]]= fi_lime.loc[i,'importance']

save_data("LIME", fi_df)

# 3.2 SHAP
import shap 
shap.initjs()
    
# Define a tree explainer for the built model
explainer_shap = shap.TreeExplainer(rf_model.best_estimator_)

fi_shap_df = pd.DataFrame(0.0, index=np.arange(data_size),columns=features)

now = datetime.datetime.now()
print(now)

for p in range(0,data_size):
    # obtain shap values for the chosen row of the test data
    shap_values = explainer_shap.shap_values(data_to_be_explained[p])
   
    #fi_shap = pd.DataFrame(shap_values, columns=['importance'])
    for f in range(0,len(features)): 
        fi_shap_df.loc[p,features[f]]= shap_values[0][f]
               
save_data("SHAP", fi_shap_df)

# Step.4 Evaluate

# 4.1 Evaluation Metrics

from sklearn.metrics import roc_auc_score
from math import sqrt

def roc_auc_ci(y_true, y_score):
    positive=1
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (str(AUC)+ "," +str(lower) +"-"+str(upper))

"""
Created on Tue Nov  6 10:06:52 2018

@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison

updated: Raul Sanchez-Vazquez
"""

import numpy as np
import scipy.stats
from scipy import stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def roc_auc_deLong_ci(y_true,y_pred):
    alpha = .95
    #y_pred = np.array([0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
    #y_true = np.array([0,    1,    0,    0,    1,    1,    0,    1,    0   ])

    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1

    print('AUC:', auc)
    print('AUC COV:', auc_cov)
    print('95% AUC CI:', ci)
    return (str(round(auc, 2)) + " (" + str(round(ci[0], 2))+ "-"+ str(round(ci[1], 2))+ ")")

# 4.2 Datasets

## Same dataset is predicted with RF
X_train_rf3, X_test_rf3, y_train_rf3, y_test_rf3 = train_test_split(X_test[0:data_size], y_test[0:data_size], test_size=0.3, random_state=16)

clf = RandomForestClassifier()
eval_model_rf3 = GridSearchCV(clf, param_grid, cv=5, n_jobs = n_defined_jobs) 
eval_model_rf3.fit(X_train_rf3, y_train_rf3)
y_pred_rf3 = eval_model_rf3.predict_proba(X_test_rf3)
y_pred_classes_rf3 =  eval_model_rf3.predict(X_test_rf3)

# compare with Test Label: y_test_eval1c
model_auc_rf3 = roc_auc_score(y_test_rf3, y_pred_rf3[:, 1])
classificationReport_rf3 = classification_report(y_test_rf3, y_pred_classes_rf3)
print("AUROC:")
print(model_auc_rf3)
print(classificationReport_rf3)

auroc_results_df.loc['PM','Values'] = roc_auc_deLong_ci(np.array(y_test_rf3).astype(int), np.array(y_pred_rf3[:, 1]))



