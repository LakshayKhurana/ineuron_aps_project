import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity,artifact_entity
from sensor.predictor import ModelResolver
from sensor import utils
from sklearn.metrics import f1_score
from sensor.config import TARGET_COLUMN
import pandas as pd


class ModelEvaluation:

    def __init__(self,
                 model_eval_config:config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact:artifact_entity.ModelTrainerArtifact):

        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver(model_registry='saved_models', transformer_dir_name='transformer', 
                                                target_encoder_dir_name='target_encoder', model_dir_name='model')
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_evaluation(self)->artifact_entity.ModelEvaluationArtifact:
        try:
            logging.info('Checking whether existing model in the saved folder is better than the current model')
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,improved_accuracy=None)
                logging.info('Current model is the first model')
                logging.info(f'Model Evaluation Artifact: {model_eval_artifact}')
                return model_eval_artifact

            logging.info('Finding the location of transformer object, model and target encoder pickle files')
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()

            logging.info('Extracting previously trained objects - transformer, model and target encoder')
            transformer = utils.load_object(file_path=transformer_path)
            model = utils.load_object(file_path=model_path)
            target_encoder = utils.load_object(file_path=target_encoder_path)

            logging.info('Extracting currently trained objects - transformer, model and target encoder')
            current_transformer = utils.load_object(filepath=self.data_transformation_artifact.transform_object_path)
            current_model = utils.load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            logging.info('Decoding the target column to string values for test dataframe')
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)            ## how is the transform function associated with the target encoder

            logging.info('Calculating accuracy using the previous model')
            input_feature_name = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            logging.info(f'Prediction using previous model {target_encoder.inverse_transform(y_pred[:5])}')
            previous_model_score = f1_score(y_true=y_true,y_pred=y_pred)
            logging.info(f'F1-Score using previous model: {previous_model_score}')

            logging.info('Calculating accuracy using current model')
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer(test_df[input_feature_name])
            y_true = current_target_encoder.transform(target_df)
            y_pred = current_model.predict(input_arr)
            logging.info(f'Prediction using current model: {current_target_encoder.inverse_transform(y_pred[:5])}')
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f'F1 Score using current model: {current_model_score}')

            if current_model_score <= previous_model_score:
                logging.info('current model is not better than previous model')
                raise Exception('current trained model is not better than previous model')
            
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True, 
                                                                          improved_accuracy=current_model_score-previous_model_score)
            logging.info(f'Model Evaluation Artifact: {model_eval_artifact}')
            return model_eval_artifact
        except Exception as e:
            raise SensorException(e, sys)
                                                               



