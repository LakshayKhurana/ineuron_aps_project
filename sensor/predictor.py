import os,sys
from typing import Optional
from sensor.entity import config_entity
from sensor.exception import SensorException
from sensor.logger import logging


class ModelResolver:
    
    def __init__(self,model_registry:str,transformer_dir_name:str,target_encoder_dir_name:str,model_dir_name:str):
        self.model_registry = model_registry
        self.transformer_dir_name = transformer_dir_name
        self.target_encoder_dir_name = target_encoder_dir_name
        self.model_dir_name = model_dir_name
        os.makedirs(self.model_registry,exist_ok=True)


    def get_latest_dir_path(self) -> Optional[str]:
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            dir_names = list(map(int,dir_names))
            latest_dir_names = max(dir_names)
            return os.path.join(self.model_registry,f'{latest_dir_names}')
        except Exception as e:
            raise SensorException(e,sys)

    
    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir == None:
                raise Exception(f'Model is not available')
            return os.path.join(latest_dir,self.model_dir_name,config_entity.MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    
    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir == None:
                raise Exception(f'Transformer is not available')
            return os.path.join(latest_dir,self.transformer_dir_name,config_entity.TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    
    def get_latest_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir == None:
                raise Exception(f'Target Encoder is not available')
            return os.path.join(latest_dir,self.target_encoder_dir_name,config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e,sys)

    
    def get_latest_save_dir_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir == None:
                return os.path.join(self.model_registry,f'{0}')
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry,f'{latest_dir_num+1}')
        except Exception as e:
            raise SensorException(e,sys)

    
    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.model_dir_name,config_entity.MODEL_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)

    
    def get_latest_save_transformer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.transformer_dir_name,config_entity.TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)


    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir,self.target_encoder_dir_name,config_entity.TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)


    

