import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml
import argparse

def data_split(config_path: Text) -> None:

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    features=pd.read_csv(config['feature_extraction']['features_path'])
    
    train_dataset, test_dataset = train_test_split(
        features,
        test_size=config['data_split']['test_size'],
        random_state=config['base']['random_state']
    )

    train_csv_path = config['data_split']['trainset_path']
    test_csv_path = config['data_split']['testset_path']
    train_dataset.to_csv(train_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)