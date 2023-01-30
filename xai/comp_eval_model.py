# Step.1 Load Data
# Load the Dataframe with:
#     - n predictors (features)
#     - 1 target


# Step.2 Define Prediction Models

# 2.1 Random Forest
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
