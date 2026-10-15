import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_injestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info("--- Training Pipeline Started ---")
        
        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_data_path, test_data_path = ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        transformation = DataTransformation()
        train_arr, test_arr, _ = transformation.initiate_data_transformation(train_data_path, test_data_path)

        # 3. Model Training
        trainer = ModelTrainer()
        accuracy, f1 = trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info(f"--- Training Pipeline Completed Successfully. Final Accuracy: {accuracy:.4f} ---")
    except Exception as e:
        raise CustomException(e, sys)