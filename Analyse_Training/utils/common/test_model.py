import pandas as pd
import numpy as np

def test_model(model, scaler, test_df, test_columns):
    # One-Hot-Encoding der Testdaten
    test_df = pd.get_dummies(test_df)

    # Fülle fehlende Dummy-Spalten mit 0, falls welche fehlen
    missing_columns = set(test_columns) - set(test_df.columns)
    test_df = test_df.copy()

    # Then add the missing columns
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
    y_pred = np.maximum(0, y_pred)

    # Die Vorhersagen als neue Spalte in das DataFrame einfügen
    test_df['Solar_MWh_pred'] = y_pred

    return test_df

def test_model_cv(pipeline, test_df, test_columns):

    categorical_cols = test_df.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        if test_df[col].isna().any():
            print(f"Spalte '{col}' enthält fehlende Werte und ist vom Typ 'category'.")
            print(f"Einzigartige Werte in der Spalte '{col}': {test_df[col].unique()}")


    # Fülle fehlende Dummy-Spalten mit 0, falls welche fehlen
    missing_columns = set(test_columns) - set(test_df.columns)
    for col in missing_columns:
        test_df[col] = 0

    if test_df.isna().sum().sum() > 0:
        test_df = test_df.fillna(0)

    # Skaliere nur die Spalten, die in test_columns enthalten sind
    X = test_df[test_columns]

    # Vorhersagen treffen
    y_pred = pipeline.predict(test_df)

    y_pred = np.maximum(0, y_pred)
    # Die Vorhersagen als neue Spalte in das DataFrame einfügen
    test_df['Solar_MWh_pred'] = y_pred

    return test_df

def test_model_cv(pipeline, test_df, test_columns):

    categorical_cols = test_df.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        if test_df[col].isna().any():
            print(f"Spalte '{col}' enthält fehlende Werte und ist vom Typ 'category'.")
            print(f"Einzigartige Werte in der Spalte '{col}': {test_df[col].unique()}")


    # Fülle fehlende Dummy-Spalten mit 0, falls welche fehlen
    missing_columns = set(test_columns) - set(test_df.columns)
    for col in missing_columns:
        test_df[col] = 0

    if test_df.isna().sum().sum() > 0:
        test_df = test_df.fillna(0)

    # Skaliere nur die Spalten, die in test_columns enthalten sind
    X = test_df[test_columns]

    # Vorhersagen treffen
    y_pred = pipeline.predict(test_df)

    y_pred = np.maximum(0, y_pred)
    # Die Vorhersagen als neue Spalte in das DataFrame einfügen
    test_df['Solar_MWh_pred'] = y_pred

    return test_df

def test_model_stacking(model, test_df, test_columns, preprocessor):
    # Fülle fehlende Dummy-Spalten mit 0, falls welche fehlen
    missing_columns = set(test_columns) - set(test_df.columns)
    for col in missing_columns:
        test_df[col] = 0

    if test_df.isna().sum().sum() > 0:
        test_df = test_df.fillna(0)

    print(test_df.isna().sum()[test_df.isna().sum() > 0])
    print(test_df[test_df.isna().any(axis=1)])
    print(test_df.head())

    X = preprocessor.transform(test_df)

    X = pd.DataFrame(
        X.toarray(),
        columns=preprocessor.get_feature_names_out()
    )
    # Vorhersagen treffen
    y_pred = model.predict(X)
    y_pred = np.maximum(0, y_pred)

    # Die Vorhersagen als neue Spalte in das DataFrame einfügen
    test_df['Solar_MWh_pred'] = y_pred

    return test_df
