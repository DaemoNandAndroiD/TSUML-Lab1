import logging
import os
import pickle
import sys
from argparse import ArgumentParser
from enum import Enum

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

class Mode(Enum):
    TRAIN = 0
    PREDICT = 1

MODEL_PATH = './model/catboost_model.pkl'

if not os.path.exists('./data'):
    os.makedirs('./data')

logging.basicConfig(
    filename='./data/catboost_log_file.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

class CatBoostModel:
    def __init__(self):
        self.study = None
        self.model = None
        self.x_train = None
        self.y_train = None

    def load_data(self, dataset_path):
        try:
            data = pd.read_csv(dataset_path)
            logging.info(f"Loaded {dataset_path}")
            return data
        except FileNotFoundError:
            logging.error(f'Dataset {dataset_path} not found.')
            sys.exit(1)

    def preprocess(self, data, mode):
        feature_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'Age', 'VIP',
                        'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        categorical_cols = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']
        numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

        imputer_categorical = SimpleImputer(strategy="most_frequent")
        imputer_numerical = SimpleImputer(strategy="median")

        data[categorical_cols] = imputer_categorical.fit_transform(data[categorical_cols])
        data[numerical_cols] = imputer_numerical.fit_transform(data[numerical_cols])

        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoded_cats = encoder.fit_transform(data[categorical_cols])
        encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

        data_final = pd.concat([encoded_cats_df, data[numerical_cols]], axis=1)

        if mode == Mode.TRAIN:
            y = data['Transported']
        else:
            y = None

        return data_final, y

    def objective(self,trial):
        param = {
            "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = CatBoostClassifier(**param)

        scores = cross_val_score(model, self.x_train, self.y_train)

        return np.mean(scores)

    def train(self, dataset_path):
        data = self.load_data(dataset_path)
        self.x_train, self.y_train = self.preprocess(data, Mode.TRAIN)

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=10)
        best_params = self.study.best_params

        self.model = CatBoostClassifier(**best_params)
        self.model.fit(self.x_train, self.y_train)

        if not os.path.exists('./model'):
            os.makedirs('./model')

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, dataset_path):
        data = self.load_data(dataset_path)
        x, y = self.preprocess(data, Mode.PREDICT)
        logging.info('Model loaded.')

        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

        toBool = lambda x: bool(x)

        predictions = self.model.predict(x)

        predictions = [toBool(x) for x in predictions]

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

    model = CatBoostModel()

    if args.command == 'train':
        model.train(args.dataset)
    elif args.command == 'predict':
        model.predict(args.dataset)










