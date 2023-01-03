from sensor import utils
from sensor.entity import config_entity
from sensor.entity import artifact_entity
from sensor.logger import logging
from sensor.exception import SensorException
import os,sys
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from typing import Optional

## We will use KS Test, to check if the distribution for a given feature is same or not
## Anomolies Detection
# 1. High Null Value
# 2. Missing Columns
# 3. Outlier
# 4. Categorial Vars: Extra Categories in the train data which the model hasn't seen
# 5. Model Drift: Target Drift, Data Drift, Concept Drift

class DataValidation:

    def __init__(self, 
                 data_validation_config:config_entity.DataValidationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f'{">>"*20} Data Validation {"<<"*20}')
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e,sys)

    def drop_missing_values_columns(self,df:pd.DataFrame,report_key_name:str) -> Optional[pd.DataFrame]:
        """
        This function will drop column which contains missing values more than specified threshold

        df: Accepts pandas dataframe
        threshold: Percentage criteria to drop a column
        =================================================================================================================
        returns Pandas DataFrame if atleast a single column is available after missing value columns drop else None
        """
        try:
            
            # drop_col_names = []
            # column_name = (df.isnull().sum()/df.shape[0]).index
            # missing_percentage = (df.isnull().sum()/df.shape[0]).value
            # for col_name,missing_percentage in zip(column_names, missing_percentage):
            #     if missing_percentage > 0.3:
            #         drop_col_names.append(col_name)

            threshold = self.data_validation_config.missing_threshold
            null_report = df.isna().sum()/df.shape[0]
            logging.info(f'selecting column name which contains null values more than {threshold}')
            drop_col_names = null_report[null_report>threshold].index
            logging.info(f'columns to be dropped: {drop_col_names}')
            self.validation_error[report_key_name] = drop_col_names
            df.drop(list(drop_col_names),axis=1,inplace=True)
            # return None if all the columns are getting dropped
            if len(df.columns) == 0:
                return None
            return df
        except Exception as e:
            raise SensorException(e,sys)

    def is_required_columns_exists(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str) -> bool:
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_columns:
                    logging.info(f'Column: {base_column} is not available')
                    missing_columns.append(base_column)

            if len(missing_columns) >0:
                self.validation_error[report_key_name] = missing_columns
                return False
            return True
        except Exception as e:
            raise SensorException(e,sys)

    def data_drift(self,base_df:pd.DataFrame,current_df:pd.DataFrame,report_key_name:str):
        try:
            self.p_value_threshold = self.data_validation_config.p_value_threshold
            drift_report = dict()
            base_columns = base_df.columns
            current_columns = current_df.columns
            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]
                # NULL Hypothesis: Both the columns are from same distribution
                results = ks_2samp(base_data,current_data)
                if results.pvalue > self.p_value_threshold:
                    # Accept NULL Hypothesis
                    drift_report[base_column] = {'p-value':float(results.pvalue),'distribution_status':True}
                else:
                    # Reject NULL Hypothesis
                    drift_report[base_column] = {'p-value':float(results.pvalue),'distribution_status':False}

            self.validation_error[report_key_name] = drift_report
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        try:
            logging.info(f'reading base dataframe')
            base_df = pd.read_csv(self.data_validation_config.base_file_path)
            base_df.replace('na',np.NaN,inplace=True)
            logging.info(f'replaced na values in the base dataframe')
            base_df = self.drop_missing_values_columns(base_df,report_key_name='missing_values_within_base_dataset')
            logging.info(f'dropped null values from base dataframe')

            logging.info(f'reading train dataframe')
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f'reading test dataframe')
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            train_df = self.drop_missing_values_columns(train_df,report_key_name='missing_values_within_train_dataset')
            logging.info(f'dropped null values from train dataframe')
            test_df = self.drop_missing_values_columns(test_df,report_key_name='missing_values_within_test_dataset')
            logging.info(f'dropped null values from test dataframe')

            exclude_columns = ['class']
            utils.convert_columns_to_float(df=base_df, exclude_columns=exclude_columns)
            utils.convert_columns_to_float(df=train_df, exclude_columns=exclude_columns)
            utils.convert_columns_to_float(df=test_df, exclude_columns=exclude_columns)

            logging.info(f'checking if desired columns are present in the train dataframe')
            train_df_columns_status = self.is_required_columns_exists(base_df, train_df,report_key_name='missing_columns_within_train_dataset')
            logging.info(f'checking if desired columns are present in the test dataframe')
            test_df_columns_status = self.is_required_columns_exists(base_df, test_df,report_key_name='missing_columns_within_test_dataset')

            if train_df_columns_status == True :
                logging.info(f'Since all the columns are present in the train dataframe, therefore checking data drift in the train dataframe')
                self.data_drift(base_df, train_df,report_key_name='data_drift_within_train_dataset')

            if test_df_columns_status == True:
                logging.info(f'Since all the columns are present in the test dataframe, therefore checking data drift in the test dataframe')
                self.data_drift(base_df, test_df,report_key_name='data_drift_within_test_dataset')

            # write the data validation yaml report
            logging.info(f'writing report in yaml file')
            utils.write_data_to_yaml_file(file_path=self.data_validation_config.report_file_path, 
                                          data=self.validation_error)

            data_validation_artifact = artifact_entity.DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f'returning the data validation artifact @ {data_validation_artifact}')
            return data_validation_artifact
            
        except Exception as e:
            raise SensorException(e, sys)
