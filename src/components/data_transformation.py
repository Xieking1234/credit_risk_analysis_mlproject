import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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
            pca = PCA(n_components=25)
            logging.info("MinMaxScaler and PCA initialized successfully.")
            return scaler, pca
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully for transformation.")

            target_column_name = "loan_status"
            
            # FIXED: Removed axis=1 since columns is already specified
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing (MinMaxScaler & PCA) on training and testing data.")

            scaler, pca = self.get_data_transformer_object()

            input_feature_train_scaled = scaler.fit_transform(input_feature_train_df)
            input_feature_test_scaled = scaler.transform(input_feature_test_df)

            input_feature_train_pca = pca.fit_transform(input_feature_train_scaled)
            input_feature_test_pca = pca.transform(input_feature_test_scaled)

            train_arr = np.c_[input_feature_train_pca, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_pca, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={"scaler": scaler, "pca": pca}
            )
            logging.info("Saved preprocessor (scaler & pca) object.")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)