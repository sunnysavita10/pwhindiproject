import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import r2_score,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, load_object, save_object, upload_file


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class CustomModel:
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object

        self.trained_model_object = trained_model_object

    def predict(self, X):
        transformed_feature = self.preprocessing_object.transform(X)

        return self.trained_model_object.predict(transformed_feature)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info(f"Splitting training and testing input and target feature")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "K-Neighbors Classifier": KNeighborsClassifier(),
                "XGBClassifier": XGBClassifier(),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }

            logging.info(f"Extracting model config file path")

            model_report: dict = evaluate_models(X=x_train, y=y_train, models=models)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            preprocessing_obj = load_object(file_path=preprocessor_path)

            custom_model = CustomModel(
                preprocessing_object=preprocessing_obj,
                trained_model_object=best_model,
            )

            logging.info(
                f"Saving model at path: {self.model_trainer_config.trained_model_file_path}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=custom_model,
            )

            predicted = best_model.predict(x_test)

            accuracy_score = accuracy_score(y_test, predicted)

            upload_file(
                from_filename=self.model_trainer_config.trained_model_file_path,
                to_filename="model.pkl",
                bucket_name=AWS_S3_BUCKET_NAME,
            )

            return accuracy_score

        except Exception as e:
            raise CustomException(e, sys)
