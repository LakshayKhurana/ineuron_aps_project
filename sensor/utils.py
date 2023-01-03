import os,sys
import pandas as pd
from sensor.config import mongo_client
from sensor.logger import logging
from sensor.exception import SensorException
import yaml

def get_collection_as_dataframe(database_name:str,collection_name:str) -> pd.DataFrame:

    """
    Description: This function return collection as dataframe
    =========================================================
    Params:
    database_name: database name
    collection_name: collection name
    =========================================================
    return Pandas dataframe of a collection
    """

    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found Columns: {df.columns}")
        if '_id' in df.columns:
            logging.info(f"Dropping _id column from the DataFrame")
            df.drop('_id',axis=1,inplace=True)
        return df
    except Exception as e:
        raise SensorException(e, sys)


def write_data_to_yaml_file(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        with open (file_path,'w') as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise SensorException(e, sys)


def convert_columns_to_float(df,exclude_columns:list):
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype('float')
        return df
    except Exception as e:
        raise SensorException(e,sys)



