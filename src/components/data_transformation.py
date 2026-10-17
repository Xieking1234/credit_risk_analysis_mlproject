import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            scaler = MinMaxScaler()
           
            logging.info("MinMaxScaler initialized successfully.")
            return scaler
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully for transformation.")

            target_column_name = "loan_status"
            
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing (MinMaxScaler) on training and testing data.")

            scaler = self.get_data_transformer_object()

            input_feature_train_scaled = scaler.fit_transform(input_feature_train_df)
            input_feature_test_scaled = scaler.transform(input_feature_test_df)

            #removing pca

            train_arr = np.c_[input_feature_train_scaled, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_scaled, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={"scaler": scaler}
            )
            logging.info("Saved preprocessor (scaler) object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)