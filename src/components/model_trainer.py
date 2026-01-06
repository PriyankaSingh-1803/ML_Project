import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
                
            )
            
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "XGB Regressor" : XGBRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(verbose = False)
            }
            params = {

    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
    },

    "Random Forest": {
        'n_estimators': [50, 100, 200]
    },

    "Gradient Boosting": {
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'n_estimators': [50, 100]
    },

    "Linear Regression": {},

    "XGB Regressor": {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100]
    },

    "AdaBoost Regressor": {
        'learning_rate': [0.01, 0.1, 0.5],
        'n_estimators': [50, 100]
    },

    "CatBoosting Regressor": {
        'depth': [6, 8],
        'learning_rate': [0.01, 0.1],
        'iterations': [50, 100]
    }
}

            
            
            model_report, trained_models = evaluate_models(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    models=models,
    params=params
)

            
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = trained_models[best_model_name]

            
            if best_model_score<0.6:
                raise CustomException("No best model found !")
            logging.info("Best found model on training and testing dataset")
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            
            return {
            "best_model_name": best_model_name,
            "best_model_score": best_model_score,
            "r2_score": r2_square
}
            
        except Exception as e:
            raise CustomException(e, sys)        
        