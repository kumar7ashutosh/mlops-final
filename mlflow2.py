import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='kumarashutoshbtech2023', repo_name='mlops-final', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kumarashutoshbtech2023/mlops-final.mlflow')

wine = load_wine()
X = wine.data
y = wine.target

# Train test split
test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5
# Mention your experiment below
mlflow.set_experiment('remote server')
model_name = "WineRandomForestModel"
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('test size', test_size)
    # Creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig("Confusion-matrix.png")

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'ashutosh', "Project": "log and register mlflow models"})
    os.environ["MLFLOW_ENABLE_LOGGED_MODEL_API"] = "false"

    # Log the model
    mlflow.sklearn.log_model(sk_model=rf,
        artifact_path="model",
        registered_model_name=model_name
        )

    print(accuracy)