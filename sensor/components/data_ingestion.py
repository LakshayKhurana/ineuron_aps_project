from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.exception import SensorException
from sensor.logger import logging
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e,sys)
    
    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info(f'Exporting collection data as dataframe')
            
            # export collection data as pandas dataframe
            df = utils.get_collection_as_dataframe(self.data_ingestion_config.database_name, self.data_ingestion_config.collection_name)

            # replace na with np.NaN
            df.replace('na',np.NaN,inplace=True)

            # save data in feature store
            logging.info(f'Save data in the feature store')
            logging.info(f'Create feature store folder if not available')
            featuer_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(featuer_store_dir,exist_ok=True)
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path,index=False,header=True)
            logging.info(f'Data successfully saved in the feature store') 

            # split data into train & test datasets
            logging.info('Splitting the dataset into train & test sets')
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size,random_state=42)

            # create dataset directory folder if it does not exists
            logging.info('creating dataset folder if it does not exists')
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path,index=False,header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path,index=False,header=True)

            # Prepare Data Ingestion Artifact

            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path, 
                train_file_path = self.data_ingestion_config.train_file_path, 
                test_file_path = self.data_ingestion_config.test_file_path)

            logging.info(f'Data Ingestion Artifact: {data_ingestion_artifact}')
            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)
        
