import sys,os
from sensor.exception import SensorException
from sensor.logger import logging
from sensor.entity import config_entity,artifact_entity
from sensor.predictor import ModelResolver
from sensor import utils

class ModelPusher:

    def __init__(self, model_pusher_config:config_entity.ModelPusherConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact):
        
        try:
            logging.info(f"{'>>'*20}  Model Pusher {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry='saved_models', transformer_dir_name = 'transformer', 
                                                target_encoder_dir_name = 'target_encoder', model_dir_name = 'model')
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_pusher(self)->artifact_entity.ModelPusherArtifact:
        try:
            logging.info('Loading transformer object,model and target encoder')
            transformer = utils.load_object(file_path=self.data_transformation_artifact.transform_object_path)
            model = utils.load_object(file_path=self.model_trainer_artifact.model_path)
            target_encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info('Pushing the transformer object,model & target encoder to model pusher directory')
            utils.save_object(file_path=self.model_pusher_config.model_pusher_saved_transformer_path,obj=transformer)
            utils.save_object(file_path=self.model_pusher_config.model_pusher_saved_model_path, obj=model)
            utils.save_object(file_path=self.model_pusher_config.model_pusher_saved_target_encoder_path, obj=target_encoder)

            logging.info('Loading the latest transformer object,model and target encoder from saved models directory') 
            transformer_path = self.model_resolver.get_latest_save_transformer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            target_encoder_path = self.model_resolver.get_latest_save_target_encoder_path()
            logging.info('Saving transformer object,model and target encoder to saved models directory')
            utils.save_object(file_path=transformer_path, obj=transformer)
            utils.save_object(file_path=model_path, obj=model)
            utils.save_object(file_path=target_encoder_path, obj=target_encoder)

            logging.info('creating model pusher artifact')
            model_pusher_artifact = artifact_entity.ModelPusherArtifact(
                                                        model_pusher_saved_dir = self.model_pusher_config.model_pusher_saved_dir,
                                                        saved_models_dir = self.model_pusher_config.saved_models_dir)
            logging.info(f'Model Pusher Artifact: {model_pusher_artifact}')
            return model_pusher_artifact
        except Exception as e:
            raise SensorException(e, sys)