import os,sys
from sensor.logger import logging
from sensor.exception import SensorException
from sensor.entity import config_entity,artifact_entity
from sensor import utils
from typing import Optional
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


class ModelTrainer:

    def __init__(self,
                 model_trainer_config:config_entity.ModelTrainerConfig,
                 data_transformation_artifact:artifact_entity.DataTransformationArtifact):
        
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    
    def model_fine_tune(self):
        try:
            pass # write the code for Grid Search CV
        except Exception as e:
            raise SensorException(e, sys)

    
    def train_model(self,x,y):
        try:
            logging.info(f'model getting trained')
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x,y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    
    def initiate_model_trainer(self) -> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f'Loading the train and test data stored in the numpy array')
            train_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_path)

            logging.info(f'Extracting input and target features from both train and test data')
            x_train, y_train = train_arr[:,:-1] , train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1] , test_arr[:,-1]

            logging.info(f'calling the model train method')
            model = self.train_model(x=x_train, y=y_train)

            logging.info(f'calculating the f1 score for train data')
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info(f'calculating the f1 score for test data')
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)
            logging.info(f'f1 score for train data: {f1_train_score} & f1 score for test data: {f1_test_score}')
            
            logging.info(f'checking if the model is underfitting or not')
            if f1_test_score < self.model_trainer_config.expected_score:
                raise Exception(f"Model is not good as it is not able to give the expected accuracy\
                                  of {self.model_trainer_config.expected_score} as we got the actual f1 score of the model as {f1_test_score}")

            logging.info(f'checking if the model is overfitting or not')
            diff = abs(f1_train_score-f1_test_score)
            
            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f'Train and Test Score diff i.e. {diff} is more than the defined overfitting threshold of {self.model_trainer_config.overfitting_threshold}')

            logging.info(f"Saving mode object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            logging.info(f"Prepare the model trainer artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
                                                                          f1_train_score=f1_train_score,
                                                                          f1_test_score=f1_test_score)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise SensorException(e,sys)