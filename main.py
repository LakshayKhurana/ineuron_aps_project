import os
import sys
from sensor.exception import SensorException
from sensor.utils import get_collection_as_dataframe
from sensor.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig
from sensor.logger import logging


if __name__ == '__main__':
     try:    
          training_pipeline_config = TrainingPipelineConfig()
          data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
          print(data_ingestion_config.to_dict())
     except Exception as e:
          raise SensorException(e, sys)