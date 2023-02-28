import yaml 
import sys
import numpy as np
import pandas as pd
import argparse
from typing import Text
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score
from visuals import plot_confusion_matrix
import json
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
import mlflow
from sklearn.metrics import accuracy_score, matthews_corrcoef
from urllib.parse import urlparse

sys.path.append('/home/bigpenguin/projects/dvc/')

def train_model(config_path: Text) -> None:

    '''
    loading params
    '''
    with open('params.yaml') as conf_file:
        config = yaml.safe_load(conf_file)
    
    train_df = pd.read_csv(config['data_split']['trainset_path'])
    test_df = pd.read_csv(config['data_split']['testset_path'])

    # model = train(train_df)
    y_train = train_df['status']
    X_train = train_df.drop(['status'],axis=1)
    target_column=config['feature_extraction']['target_column']
    y_test = test_df.loc[:, target_column]
    X_test = test_df.drop(target_column, axis=1)


############################   MLFLOW    ##########################

    mlflow_config = config["mlflow_config"]

    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        etc = ExtraTreesClassifier(random_state=config['base']['random_state'])
        etc.fit(X_train,y_train)

        prediction = etc.predict(X_test)
        f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')
        acc = accuracy_score(prediction,y_test)*100
        mcc = matthews_corrcoef(prediction,y_test)

        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Mathews correlation coefficient", mcc)

        tracking_uri_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_uri_type_store != "file":
            mlflow.sklearn.log_model(
            etc, 
            "model", 
            registered_model_name=mlflow_config["registered_model_name"]
            )
        else:
            mlflow.sklearn.load_model(etc, "model")


################################################################

    # labels = load_iris(as_frame=True).target_names.tolist()
    labels = ['covid-19','healthy']
    cm = confusion_matrix(y_test, prediction)

    report = {
        'f1': f1,
        'cm': cm,
        'actual': y_test,
        'predicted': prediction
    }


    # save f1 metrics file
    reports_folder = Path(config['evaluate']['reports_dir'])
    metrics_path = reports_folder / config['evaluate']['metrics_file']

    json.dump(
        obj={'f1_score': report['f1']},
        fp=open(metrics_path, 'w')
    )

    # save confusion_matrix.png
    plt = plot_confusion_matrix(cm=report['cm'],
                                target_names=labels,
                                normalize=False)
    confusion_matrix_png_path = reports_folder / config['evaluate']['confusion_matrix_image']
    plt.savefig(confusion_matrix_png_path)

    models_path = config['train']['model_path']
    joblib.dump(etc, models_path)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)
