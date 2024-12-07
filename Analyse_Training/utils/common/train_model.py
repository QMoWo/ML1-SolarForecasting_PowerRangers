import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from utils.common.print_evaluation import print_evaluation
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, BaggingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



def train_model(test_df: pd.DataFrame, columns: list, y_column: str) -> tuple:
    """
    columns: list of columns to use for training
    y_column: column to predict
    """

    # get target column and store it in y
    y = test_df.pop(y_column)

    # one hot encode
    test_df = pd.get_dummies(test_df[columns])

    
    # split data into X and y
    X_train, X_test, y_train, y_test = train_test_split(test_df, 
                                                        y, 
                                                        test_size=0.2)


    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    # fit model
    model = Ridge(alpha=0.5)
    model.fit(X_train, y_train)

    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # evaluate model
    # evaluation = print_evaluation(model, 
    #                               X_train, 
    #                               X_test, 
    #                               y_train, 
    #                               y_test, 
    #                               y_train_pred, 
    #                               y_test_pred, 
    #                               0, 
    #                               test_df.columns)
    
    return model, scaler, test_df.columns
    

def test_model(model, scaler, test_df, test_columns):
    # One-Hot-Encoding der Testdaten
    test_df = pd.get_dummies(test_df)

    # Fülle fehlende Dummy-Spalten mit 0, falls welche fehlen
    missing_columns = set(test_columns) - set(test_df.columns)
    for col in missing_columns:
        test_df[col] = 0

    if test_df.isna().sum().sum() > 0:
        test_df = test_df.fillna(0)

    print(test_df.isna().sum()[test_df.isna().sum() > 0])
    print(test_df[test_df.isna().any(axis=1)])
    print(test_df.head())
    # Skaliere nur die Spalten, die in test_columns enthalten sind
    X = scaler.transform(test_df[test_columns])
    
    # Vorhersagen treffen
    y_pred = model.predict(X)

    # Die Vorhersagen als neue Spalte in das DataFrame einfügen
    test_df['Solar_MWh_pred'] = y_pred
    
    return test_df


def train_model_cv(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict) -> tuple:
    """
    Trains a Ridge regression model using GridSearchCV for hyperparameter tuning.
    
    Parameters:
        test_df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to use for training.
        y_column (str): Column to predict.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        
    Returns:
        tuple: Best model, evaluation metrics, scaler, and selected feature columns.
    """
    # Get target column and store it in y
    y = test_df.pop(y_column)

    # One-hot encode
    test_df = pd.get_dummies(test_df[columns])

    # Split data into X and y
    X_train, X_test, y_train, y_test = train_test_split(
        test_df, y, test_size=0.2
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the model
    ridge = Ridge()

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring='neg_root_mean_squared_error',  # Optimize for MSE
        n_jobs=-1  # Use all available CPUs
    )
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert negative MSE back to positive

    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Evaluate the model
    evaluation = print_evaluation(
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        best_score,
        test_df.columns,
    )

    return best_model, evaluation, scaler, test_df.columns, best_params
    

def train_model_stacking(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict) -> tuple:
    """
    Trains a stacking model using Ridge, Random Forest, and XGBoost as base learners and Ridge as the final estimator.
    
    Parameters:
        test_df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to use for training.
        y_column (str): Column to predict.
        param_grid (dict): Hyperparameter grid for GridSearchCV (used for Ridge).
        
    Returns:
        tuple: Best stacked model, evaluation metrics, scaler, and selected feature columns.
    """
    # Get target column and store it in y
    y = test_df.pop(y_column)

    # One-hot encode
    test_df = pd.get_dummies(test_df[columns], drop_first=True)

    # Split data into X and y
    X_train, X_test, y_train, y_test = train_test_split(
        test_df, y, test_size=0.2
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define base models for stacking
    base_learners = [
        ('ridge', Ridge()),  # Ridge regression as a base learner
        ('rf', RandomForestRegressor(n_estimators=5,  n_jobs=-1)),  # Random Forest as a base learner
        ('xgb', xgb.XGBRegressor(n_estimators=10,  enable_categorical= True, n_jobs=-1))  # XGBoost as a base learner
    ]

    # Define final estimator (meta-model)
    final_estimator = Ridge()

    grid_search = GridSearchCV(estimator=final_estimator, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    best_final_estimator = grid_search.best_estimator_


    # Create the Stacking Regressor model
    stacking_model = StackingRegressor(estimators=base_learners, final_estimator=best_final_estimator)

    # Fit the stacking model
    stacking_model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = stacking_model.predict(X_train)
    y_test_pred = stacking_model.predict(X_test)

    # Evaluate the model
    evaluation = print_evaluation(
        stacking_model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        best_final_estimator,  # R^2 score or you can use RMSE if preferred
        test_df.columns,
    )

    return stacking_model, evaluation, scaler, test_df.columns, stacking_model.get_params()


def train_model_boosting(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict) -> tuple:
    """
    Trains a boosting model using XGBoost with hyperparameter tuning.
    
    Parameters:
        test_df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to use for training.
        y_column (str): Column to predict.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        
    Returns:
        tuple: Best boosting model, evaluation metrics, scaler, and selected feature columns.
    """
    # Get target column and store it in y
    y = test_df.pop(y_column)

    # One-hot encode categorical features
    test_df = pd.get_dummies(test_df[columns], drop_first=True)

    # Split data into X and y
    X_train, X_test, y_train, y_test = train_test_split(
        test_df, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the base boosting model
    xgb_model = xgb.XGBRegressor(
        n_jobs=-1, 
        enable_categorical=True,
        random_state=42
    )

    # Perform hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Retrieve the best model from GridSearchCV
    best_boosting_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_boosting_model.predict(X_train)
    y_test_pred = best_boosting_model.predict(X_test)

    # Evaluate the model
    # evaluation = print_evaluation(
    #     best_boosting_model,
    #     X_train,
    #     X_test,
    #     y_train,
    #     y_test,
    #     y_train_pred,
    #     y_test_pred,
    #     grid_search.best_params_,  # Include best hyperparameters in evaluation
    #     test_df.columns,
    # )

    return best_boosting_model, 0, scaler, test_df.columns, grid_search.best_params_

def train_model_bagging(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict) -> tuple:
    """
    Trains a bagging model using BaggingRegressor with hyperparameter tuning.
    
    Parameters:
        test_df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to use for training.
        y_column (str): Column to predict.
        param_grid (dict): Hyperparameter grid for GridSearchCV.
        
    Returns:
        tuple: Best bagging model, evaluation metrics, scaler, and selected feature columns.
    """
    # Get target column and store it in y
    y = test_df.pop(y_column)

    # One-hot encode categorical features
    test_df = pd.get_dummies(test_df[columns], drop_first=True)

    # Split data into X and y
    X_train, X_test, y_train, y_test = train_test_split(
        test_df, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the base estimator for bagging
    base_estimator = DecisionTreeRegressor(random_state=42)

    # Initialize the bagging model
    bagging_model = BaggingRegressor(
        base_estimator=base_estimator,
        n_jobs=-1,
        random_state=42
    )

    # Perform hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(
        estimator=bagging_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Retrieve the best model from GridSearchCV
    best_bagging_model = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_bagging_model.predict(X_train)
    y_test_pred = best_bagging_model.predict(X_test)

    # Evaluate the model
    evaluation = print_evaluation(
        best_bagging_model,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        grid_search.best_params_,
        test_df.columns,
    )

    return best_bagging_model, evaluation, scaler, test_df.columns, grid_search.best_params_
