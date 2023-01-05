import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor import utils
from sensor.entity import config_entity,artifact_entity
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sensor.config import TARGET_COLUMN
from sklearn.preprocessing import LabelEncoder


class DataTransformation:

    def __init__(self,
                 data_transformation_config:config_entity.DataTransformationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        
        try:
            logging.info(f'{">>"*20} Data Transformation {"<<"*20}')
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transform_object(cls) -> Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant',fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps = [
                                         ('Imputer',simple_imputer),
                                         ('RobustScaler',robust_scaler)])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_transformation(self,) -> artifact_entity.DataTransformationArtifact:
        try:
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f'selecting input features for train and test dataframe')
            input_feature_train_df = train_df.drop(TARGET_COLUMN,axis=1)
            input_feature_test_df = test_df.drop(TARGET_COLUMN,axis=1)

            logging.info(f'selecting target feature for train and test dataframe')
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f'transformation of input features')
            transformation_pipeline = DataTransformation.get_data_transform_object()
            transformation_pipeline.fit(input_feature_train_df)
            input_feature_train_arr = transformation_pipeline.transform(input_feature_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_feature_test_df)

            logging.info(f'transformation of the target feature')
            label_encoder = LabelEncoder()
            label_encoder.fit(target_feature_train_df)
            target_feature_train_arr = label_encoder.transform(target_feature_train_df)
            target_feature_test_arr = label_encoder.transform(target_feature_test_df)

            logging.info(f'balancing the dataset')
            smote = SMOTETomek(sampling_strategy='minority')
            logging.info(f'shape of the train data before resampling, Input: {input_feature_train_arr.shape}, Target: {target_feature_train_df.shape}')
            input_feature_train_arr,target_feature_train_arr = smote.fit_resample(input_feature_train_arr,target_feature_train_arr)
            logging.info(f'shape of the test data before resampling, Input: {input_feature_test_arr.shape}, Target: {target_feature_test_df.shape}')
            input_feature_test_arr,target_feature_test_arr = smote.fit_resample(input_feature_test_arr,target_feature_test_arr)

            logging.info(f'concatenating the input and target arr for train data')
            train_arr = np.c_[input_feature_train_arr,target_feature_train_arr]
            logging.info(f'concatenating the input and target arr for test data')
            test_arr = np.c_[input_feature_test_arr,target_feature_test_arr]
            logging.info(f'saving the nunpy array data')
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_array_data(file_path = self.data_transformation_config.transformed_test_path, array=test_arr)

            logging.info(f'saving the pipeline object')
            utils.save_object(file_path = self.data_transformation_config.transform_object_path, obj = transformation_pipeline)
            logging.info(f'saving the label encoder object')
            utils.save_object(file_path = self.data_transformation_config.target_encoder_path, obj = label_encoder)

            logging.info(f'preparing the data transformation artifact')

            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                                                    transform_object_path = self.data_transformation_config.transform_object_path,
                                                    transformed_train_path = self.data_transformation_config.transformed_train_path,
                                                    transformed_test_path = self.data_transformation_config.transformed_test_path,
                                                    target_encoder_path = self.data_transformation_config.target_encoder_path)
            
            logging.info(f'returning the data transformation artifact {data_transformation_artifact}')
            return data_transformation_artifact
        except Exception as e:
            raise SensorException(e,sys)
                
