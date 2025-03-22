import os

import logging
import sys
from argparse import ArgumentParser
from enum import Enum

import pickle

import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

if not os.path.exists('./data'):
    os.makedirs('./data')

logging.basicConfig(
    filename='./data/log_file.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

MODEL_PATH = './model/logistic_model.pkl'

class Mode(Enum):
    TRAIN = 0
    PREDICT = 1

class LogisticRegressionModel:
    def __init__(self):
        self.pipeline = None

    def load_data(self, dataset_path):
        try:
            data = pd.read_csv(dataset_path)
            logging.info(f"Loaded {dataset_path}")
            return data
        except FileNotFoundError:
            logging.error(f'Dataset {dataset_path} not found.')
            sys.exit(1)

    def preprocess(self, data, mode):
        feature_cols = ['HomePlanet','CryoSleep','Destination','Age','VIP','RoomService','FoodCourt','ShoppingMall','Spa','VRDeck']

        age_cols = ['Age']
        addons_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        categorical_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']

        age_imputer = SimpleImputer(strategy='median')
        addons_imputer = SimpleImputer(strategy='constant', fill_value=0)
        categorical_imputer = SimpleImputer(strategy="most_frequent")

        data[categorical_cols] = pd.DataFrame(categorical_imputer.fit_transform(data[categorical_cols]), columns=categorical_cols)
        data[age_cols] = pd.DataFrame(age_imputer.fit_transform(data[age_cols]), columns=age_cols)
        data[addons_cols] = pd.DataFrame(addons_imputer.fit_transform(data[addons_cols]), columns=addons_cols)

        x = data[feature_cols]
        y = None

        if mode == Mode.TRAIN:
            y = data['Transported']

        return x, y

    def build_pipeline(self):
        categorical = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
        numerical = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        categorical_step = Pipeline(
            steps=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))]
        )

        numerical_step = Pipeline(
            steps=[('scaler', StandardScaler())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('categorical', categorical_step, categorical),
                ('numerical', numerical_step, numerical)
            ]
        )

        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=10000))
            ]
        )

        return pipeline

    def train(self, dataset_path):

        data = self.load_data(dataset_path)
        x,y = self.preprocess(data, Mode.TRAIN)
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(x,y)
        logging.info('Training complete.')

        if not os.path.exists('./model'):
            os.makedirs('./model')

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.pipeline, f)
        logging.info(f'Model saved to {MODEL_PATH}.')

    def predict(self, dataset_path):
        data = self.load_data(dataset_path)
        x,y = self.preprocess(data, Mode.PREDICT)
        logging.info(f"Loaded {dataset_path}")


        with open(MODEL_PATH, 'rb') as f:
            self.pipeline = pickle.load(f)

        logging.info('Model loaded.')

        predictions = self.pipeline.predict(x)
        logging.info('Prediction complete.')

        result_data = pd.DataFrame(
            {
                'PassengerId': data['PassengerId'],
                'Transported': predictions
            }
        )

        result_path = './data/result.csv'
        result_data.to_csv(result_path, index=False)
        logging.info(f'Predictions saved to {result_path}.')

        return predictions


if(__name__ == '__main__'):
    parser = ArgumentParser()
    parser.add_argument('command', choices=['train', 'predict'])
    parser.add_argument('--dataset', type=str)

    args = parser.parse_args()

    model = LogisticRegressionModel()

    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)

