import pandas as pd
import os
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
dagshub.init(repo_owner='kumarashutoshbtech2023', repo_name='mlops-final', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kumarashutoshbtech2023/mlops-final.mlflow')
mlflow.set_experiment('model comparison')
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
x_train,x_test,y_train,y_test=train_test_split(x,y,stratify=y,test_size=0.2,random_state=42)
def evaluate_model(y_true, y_pred):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred)

models = {
    "lc": LogisticRegression(max_iter=500),
    "knn": KNeighborsClassifier(),
    "svc": SVC(probability=True),
    "dtc": DecisionTreeClassifier(),
    "rfc": RandomForestClassifier(),
    "abc": AdaBoostClassifier()
}

params = {
    "lc": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "liblinear"]
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"]
    },
    "dtc": {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "rfc": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "abc": {
        "n_estimators": [50, 100],
        "learning_rate": [0.1, 1]
    }
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None
best_score = -np.inf
best_model_name = None
report = {}
for name,model in models.items():
    print(f"training model {name}")
    with mlflow.start_run(run_name=name):
        param_grid=params.get(name,{})
        if param_grid:
            search=RandomizedSearchCV(model,param_grid,n_iter=3,scoring='f1_weighted',cv=skf,verbose=1,random_state=42,n_jobs=-1)
            search.fit(x_train,y_train)
            best_estimator=search.best_estimator_
            mlflow.log_params(search.best_params_)
        else:
            best_estimator=model.fit(x_train,y_train)
        y_pred=best_estimator.predict(x_test)
        acc,f1=evaluate_model(y_test,y_pred)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(best_estimator, artifact_path="model")
        report[name] = {"accuracy": acc, "f1": f1}

        if f1 > best_score:
            best_score = f1
            best_model = best_estimator
            best_model_name = name
            
print('best model', best_model_name)
print('best score',best_score)

with mlflow.start_run(run_name='best model register') as reg:
    y_pred=best_model.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    sns.heatmap(cm,annot=True,fmt='d')
    plt.title('Confusion Matrix')

    plt.savefig("Confusion-matrix-model.png")

    mlflow.log_artifact("Confusion-matrix-model.png")
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(best_model,artifact_path='best_model',registered_model_name='best model registry')
    
