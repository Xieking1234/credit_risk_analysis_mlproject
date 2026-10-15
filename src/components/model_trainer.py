import os
import sys
from dataclasses import dataclass
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            model = XGBClassifier(
                n_estimators=500,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_lambda=1,
                objective='multi:softmax',
                random_state=42
            )

            logging.info("Training XGBoost model...")
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            logging.info(f"Test Accuracy: {acc}, Test F1-Score: {f1}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            return acc, f1

        except Exception as e:
            raise CustomException(e, sys)