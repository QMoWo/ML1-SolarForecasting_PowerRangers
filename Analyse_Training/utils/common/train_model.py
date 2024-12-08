import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import TimeSeriesSplit, KFold, GroupShuffleSplit, GroupKFold, train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from utils.common.print_evaluation import (print_evaluation_simple_model,
    print_evaluation_stacking)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class OneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = None

    def fit(self, X, y=None):
        self.columns = X.columns
        return self

    def transform(self, X):
        return pd.get_dummies(X[self.columns])



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
    evaluation = print_evaluation_simple_model(model,
                                  X_train,
                                  X_test,
                                  y_train,
                                  y_test,
                                  y_train_pred,
                                  y_test_pred,
                                  0,
                                  test_df.columns)

    return model, evaluation, scaler, test_df.columns


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
    X = pd.get_dummies(test_df[columns])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create pipeline
    pipeline = Pipeline([
        ('onehot', OneHotEncoder()),
        ('scaler', MinMaxScaler()),
        ('ridge', Ridge())
    ])

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={'ridge__' + key: value for key, value in param_grid.items()},
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert negative MSE back to positive

    # Make predictions on train and test sets
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    y_train_pred = np.maximum(0, y_train_pred)
    y_test_pred = np.maximum(0, y_test_pred)
    # Evaluate the model
    evaluation = print_evaluation_simple_model(
        best_pipeline.named_steps['ridge'],
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        best_score,
        X.columns,
    )

    return best_pipeline, evaluation, X.columns, best_params


def train_model_stacking(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict) -> tuple:
    """
    Trains a stacking model using a pipeline approach with Ridge, Random Forest, and Gradient Boosting as base learners.

    Parameters:
        test_df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to use for training.
        y_column (str): Column to predict.
        param_grid (dict): Hyperparameter grid for GridSearchCV.

    Returns:
        tuple: Best pipeline, evaluation metrics, selected feature columns, best parameters.
    """
    print("Starting stacking model training...")

    test_df = test_df.sort_values('dtm')  # Sortieren nach Datum

    y = test_df.pop(y_column)

    # One-hot encode
    print("One-hot encoding features...")
    X = pd.get_dummies(test_df[columns])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Ridge Regression
    ridge_param_grid = {
        'alpha': [0.1, 10, 10.0]  # Regularization strength
    }
    ridge_search = GridSearchCV(
        estimator=Ridge(),
        param_grid=ridge_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    ridge_search.fit(X_train, y_train)
    best_ridge = ridge_search.best_estimator_

    # Gradient Boosting
    gbr_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }
    gbr_search = GridSearchCV(
        estimator=GradientBoostingRegressor(),
        param_grid=gbr_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    gbr_search.fit(X_train, y_train)
    best_gbr = gbr_search.best_estimator_

    # XGBoost
    xgb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5]
    }
    xgb_search = GridSearchCV(
        estimator=xgb.XGBRegressor(),
        param_grid=xgb_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error'
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    print("Base learners optimized!")

    # Create stacking pipeline
    base_learners = [
        ('ridge', Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(k=20)),
            ('model', best_ridge)
        ])),
        ('gbr', Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(k=20)),
            ('model', best_gbr)
        ])),
        ('xgb', Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(k=20)),
            ('model', best_xgb)
        ])),
        ('lasso', Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(k=20)),
            ('model', Lasso())
        ]))
    ]

    final_estimator = RandomForestRegressor()

    pipeline = Pipeline([
        # ('onehot', OneHotEncoder()),
        ('feature_selection', SelectKBest(score_func=f_regression, k=20)),
        ('scaler', StandardScaler()),
        ('stacking', StackingRegressor(
            estimators=base_learners,
            final_estimator=final_estimator
        ))
    ])

    full_param_grid_xgb = {
        'feature_selection__k': [10, 20, 'all'],
        'stacking__final_estimator__n_estimators': [5, 10],
        'stacking__final_estimator__learning_rate': [0.01, 0.1]
    }

    full_param_grid_rf = {
        'feature_selection__k': [10, 20, 'all'],
        'stacking__final_estimator__n_estimators': [100, 150],
        'stacking__final_estimator__max_depth': [3, 5]
    }

    full_param_grid = {
        'stacking__final_estimator__alpha': param_grid['alpha']
    }

    full_param_grid_gb = {
        'stacking__final_estimator__n_estimators': [100, 200],  # Anzahl der Bäume
        'stacking__final_estimator__learning_rate': [0.01, 0.05, 0.1],  # Learning Rate
    }

    tscv = TimeSeriesSplit(n_splits=3)

    # Perform GridSearchCV
    print("Optimizing stacking regressor...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=full_param_grid_rf,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Make predictions
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    y_train_pred = np.maximum(0, y_train_pred)
    y_test_pred = np.maximum(0, y_test_pred)

    # Evaluate the model
    evaluation = print_evaluation_stacking(
        best_pipeline.named_steps['stacking'],
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        best_score,
        X.columns,
    )

    return best_pipeline, evaluation, X.columns, best_params


def train_model_stacking_bagging(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict) -> tuple:
    """
    Trains a stacking model using a pipeline approach with Ridge, Random Forest, and Gradient Boosting as base learners.

    Parameters:
        test_df (pd.DataFrame): Input DataFrame.
        columns (list): List of columns to use for training.
        y_column (str): Column to predict.
        param_grid (dict): Hyperparameter grid for GridSearchCV.

    Returns:
        tuple: Best pipeline, evaluation metrics, selected feature columns, best parameters.
    """
    print("Starting stacking model training...")

    test_df = test_df.sort_values('dtm')  # Sortieren nach Datum

    y = test_df.pop(y_column)

    # One-hot encode
    print("One-hot encoding features...")
    X = pd.get_dummies(test_df[columns])

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ridge Regression
    ridge_param_grid = {
        'alpha': [0.1, 10, 10.0]  # Regularization strength
    }
    ridge_search = GridSearchCV(
        estimator=Ridge(),
        param_grid=ridge_param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error'
    )
    ridge_search.fit(X_train, y_train)
    best_ridge = ridge_search.best_estimator_

    # Gradient Boosting
    gbr_param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5]
    }
    gbr_search = GridSearchCV(
        estimator=GradientBoostingRegressor(),
        param_grid=gbr_param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error'
    )
    gbr_search.fit(X_train, y_train)
    best_gbr = gbr_search.best_estimator_

    # XGBoost
    xgb_param_grid = {
        'n_estimators': [100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5]
    }
    xgb_search = GridSearchCV(
        estimator=xgb.XGBRegressor(),
        param_grid=xgb_param_grid,
        cv=3,
        scoring='neg_root_mean_squared_error'
    )
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_

    print("Base learners optimized!")

    # Create stacking pipeline
    base_learners = [
        ('ridge', Pipeline([
            ('scaler', StandardScaler()),
            ('model', best_ridge)
        ])),
        ('gbr', Pipeline([
            ('scaler', StandardScaler()),
            ('model', best_gbr)
        ])),
        ('xgb', Pipeline([
            ('scaler', StandardScaler()),
            ('model', best_xgb)
        ]))
    ]


    final_estimator = DecisionTreeRegressor()

    bagging_model = BaggingRegressor(
        estimator=final_estimator,
        random_state=42,
        n_jobs=-1
    )
    pipeline = Pipeline([
        # ('onehot', OneHotEncoder()),
        ('scaler', StandardScaler()),
        ('stacking', StackingRegressor(
            estimators=base_learners,
            final_estimator=bagging_model
        ))
    ])

    full_param_grid_xgb = {
        'feature_selection__k': [10, 20, 'all'],
        'stacking__final_estimator__n_estimators': [5, 10],
        'stacking__final_estimator__learning_rate': [0.01, 0.1]
    }

    full_param_grid_rf = {
        'stacking__final_estimator__n_estimators': [100, 150],
        'stacking__final_estimator__max_depth': [3, 5]
    }

    full_param_grid = {
        'stacking__final_estimator__alpha': param_grid['alpha']
    }

    full_param_grid_gb = {
        'stacking__final_estimator__n_estimators': [100, 200],  # Anzahl der Bäume
        'stacking__final_estimator__learning_rate': [0.01, 0.05, 0.1],  # Learning Rate
    }

    full_param_grid_bagging = {
        'stacking__final_estimator__n_estimators': [100],      # Anzahl der Bags
        'stacking__final_estimator__max_samples': [0.7],     # Anteil der Samples pro Bag
        'stacking__final_estimator__max_features': [0.7],
    }

    tscv = TimeSeriesSplit(n_splits=3)

    # Perform GridSearchCV
    print("Optimizing stacking regressor...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=full_param_grid_bagging,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_

    # Make predictions
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    y_train_pred = np.maximum(0, y_train_pred)
    y_test_pred = np.maximum(0, y_test_pred)

    # Evaluate the model
    evaluation = print_evaluation_stacking(
        best_pipeline.named_steps['stacking'],
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
        y_test_pred,
        best_score,
        X.columns,
    )

    return best_pipeline, evaluation, X.columns, best_params

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
        estimator=base_estimator,
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
