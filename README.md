# covid_mlops
A mlops workflow using dvc and mlflow to monitor models, data and results

mlflow commands: 
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234
