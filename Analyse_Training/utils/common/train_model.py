import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from utils.common.print_evaluation import (print_evaluation_simple_model, 
                                           print_evaluation_DTR,
                                           print_evaluation_stacking)
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt


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


def train_model_cv(test_df: pd.DataFrame, columns: list, y_column: str, param_grid: dict, model: str) -> tuple:
    # Get target column and store it in y
    y = test_df.pop(y_column)

    X_train, X_test, y_train, y_test = train_test_split(test_df[columns], y, test_size=0.2)

    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    if model == 'ridge':
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('ridge', Ridge())
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'ridge__' + key: value for key, value in param_grid.items()},
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
    elif model == 'lasso':
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('lasso', Lasso())
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'lasso__' + key: value for key, value in param_grid.items()},
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
    elif model == 'DTR':
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('DTR', DecisionTreeRegressor())
        ])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid={'DTR__' + key: value for key, value in param_grid.items()},
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

    
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  

    preprocessor = best_pipeline.named_steps['preprocessing']
    feature_names = preprocessor.get_feature_names_out()
    

    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    # y_train_pred = np.maximum(0, y_train_pred)
    # y_test_pred = np.maximum(0, y_test_pred)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Oben links: Train - Tatsächlich vs Vorhergesagt
    axes[0,0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0,0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Tatsächliche Werte')
    axes[0,0].set_ylabel('Vorhergesagte Werte')
    axes[0,0].set_title('Train: Tatsächliche vs. Vorhergesagte Werte')
    axes[0,0].grid(True)

    # Oben rechts: Train - Residuenplot
    residuals_train = y_train - y_train_pred
    axes[0,1].scatter(y_train_pred, residuals_train, alpha=0.5)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Vorhergesagte Werte')
    axes[0,1].set_ylabel('Residuen')
    axes[0,1].set_title('Train: Residuenplot')
    axes[0,1].grid(True)

    # Unten links: Test - Tatsächlich vs Vorhergesagt
    axes[1,0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Tatsächliche Werte')
    axes[1,0].set_ylabel('Vorhergesagte Werte')
    axes[1,0].set_title('Test: Tatsächliche vs. Vorhergesagte Werte')
    axes[1,0].grid(True)

    # Unten rechts: Test - Residuenplot
    residuals_test = y_test - y_test_pred
    axes[1,1].scatter(y_test_pred, residuals_test, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Vorhergesagte Werte')
    axes[1,1].set_ylabel('Residuen')
    axes[1,1].set_title('Test: Residuenplot')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals_test, kde=True)
    plt.title('Verteilung der Residuen')
    plt.xlabel('Residuen')
    plt.show()

    if model == 'ridge' or model == 'lasso':
        evaluation = print_evaluation_simple_model(
            best_pipeline.named_steps[model],
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_pred,
            y_test_pred,
            best_score,
            feature_names,
        )
    elif model == 'DTR':
        evaluation = print_evaluation_DTR(
            best_pipeline.named_steps[model],
            X_train,
            X_test,
            y_train,
            y_test,
            y_train_pred,
            y_test_pred,
            best_score,
            feature_names,
        )

    return best_pipeline, evaluation, feature_names, best_params
    


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

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
            test_df[columns], y, test_size=0.3
        )

    # Separate numeric and categorical columns
    numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    print("Setting up preprocessing pipeline...")

    # Preprocessing for numeric features: scaling
    numeric_transformer = StandardScaler()
    # Preprocessing for categorical features: one-hot encoding
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    # Combine preprocessors in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    print("Applying transformations...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    print(feature_names)

    X_train = pd.DataFrame(
        X_train_transformed.toarray(),
        columns=preprocessor.get_feature_names_out()
    )
    X_test = pd.DataFrame(
        X_test_transformed.toarray(),
        columns=preprocessor.get_feature_names_out()
    )

    print("Feature preprocessing complete.")

    # Ridge Regression
    ridge_param_grid = {
        'alpha': np.linspace(0, 20, 20)  
    }

    ridge_search = GridSearchCV(
        estimator=Ridge(),
        param_grid=ridge_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    ridge_search.fit(X_train, y_train)
    best_ridge = ridge_search.best_estimator_
    print(f"Best params Ridge: {ridge_search.best_params_}")

    # Gradient Boosting
    gbr_param_grid = {
        'n_estimators': [150],
        'learning_rate': [0.1],
        'max_depth': [5]
    }
    gbr_search = GridSearchCV(
        estimator=GradientBoostingRegressor(),
        param_grid=gbr_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    gbr_search.fit(X_train, y_train)
    best_gbr = gbr_search.best_estimator_
    print(f"Best params GradientBoosting: {gbr_search.best_params_}")


    dt_param_grid = {
        'max_depth': [5],          # Maximale Tiefe des Baums
        'min_samples_split': [5],      # Minimale Anzahl Samples für Split
        'min_samples_leaf': [5]        # Minimale Anzahl Samples in Blättern
    }
    dt_search = GridSearchCV(
        estimator=DecisionTreeRegressor(),
        param_grid=dt_param_grid,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    dt_search.fit(X_train, y_train)
    best_dt = dt_search.best_estimator_
    print(f"Best params DTR: {dt_search.best_params_}")


    print("Base learners optimized!")

    # Create stacking pipeline
    base_learners = [
        ('ridge', Pipeline([
            #('select', SelectKBest(k=20)),
            ('model', best_ridge)
        ])),
        (('gbr', Pipeline([
            #('select', SelectKBest(k=20)),
            ('model', best_gbr)
        ]))),
        ('dct', Pipeline([
            # ('select', SelectKBest(k=20)),
            ('model', best_dt)
        ]))
    ]

    final_estimator = RandomForestRegressor()
    #final_estimator = xgb.XGBRegressor()

    pipeline = Pipeline([
        #('feature_selection', SelectKBest()),
        ('stacking', StackingRegressor(
            estimators=base_learners,
            final_estimator=final_estimator
        ))
    ])

    full_param_grid_rf = {
            'stacking__final_estimator__n_estimators': [100],       # Anzahl der Bäume im Wald
            'stacking__final_estimator__max_depth': [3],         # Maximale Tiefe der Bäume
            'stacking__final_estimator__min_samples_split': [2],       # Mindestanzahl von Proben, um einen Knoten zu teilen
            'stacking__final_estimator__min_samples_leaf': [1],         # Mindestanzahl von Proben pro Blattknoten
        }



    # Perform GridSearchCV
    print("Optimizing stacking regressor...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=full_param_grid_rf,
        cv=5,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f"Best params: {best_params}")
    best_score = -grid_search.best_score_
    print(f"Best pipe: {best_pipeline}")


    # Make predictions
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # Oben links: Train - Tatsächlich vs Vorhergesagt
    axes[0,0].scatter(y_train, y_train_pred, alpha=0.5)
    axes[0,0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('Tatsächliche Werte')
    axes[0,0].set_ylabel('Vorhergesagte Werte')
    axes[0,0].set_title('Train: Tatsächliche vs. Vorhergesagte Werte')
    axes[0,0].grid(True)

    # Oben rechts: Train - Residuenplot
    residuals_train = y_train - y_train_pred
    axes[0,1].scatter(y_train_pred, residuals_train, alpha=0.5)
    axes[0,1].axhline(y=0, color='r', linestyle='--')
    axes[0,1].set_xlabel('Vorhergesagte Werte')
    axes[0,1].set_ylabel('Residuen')
    axes[0,1].set_title('Train: Residuenplot')
    axes[0,1].grid(True)

    # Unten links: Test - Tatsächlich vs Vorhergesagt
    axes[1,0].scatter(y_test, y_test_pred, alpha=0.5)
    axes[1,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Tatsächliche Werte')
    axes[1,0].set_ylabel('Vorhergesagte Werte')
    axes[1,0].set_title('Test: Tatsächliche vs. Vorhergesagte Werte')
    axes[1,0].grid(True)

    # Unten rechts: Test - Residuenplot
    residuals_test = y_test - y_test_pred
    axes[1,1].scatter(y_test_pred, residuals_test, alpha=0.5)
    axes[1,1].axhline(y=0, color='r', linestyle='--')
    axes[1,1].set_xlabel('Vorhergesagte Werte')
    axes[1,1].set_ylabel('Residuen')
    axes[1,1].set_title('Test: Residuenplot')
    axes[1,1].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals_test, kde=True)
    plt.title('Verteilung der Residuen')
    plt.xlabel('Residuen')
    plt.show()

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
        feature_names,
    )

    return best_pipeline, evaluation, feature_names, best_params, preprocessor
