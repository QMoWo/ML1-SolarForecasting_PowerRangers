import pandas as pd

def merge_forcast_and_train_df(df_forecast, df_train) -> pd.DataFrame:
    # Transformiere "ref_datetime" und "dtm" zu datetime
    df_train["ref_datetime"] = pd.to_datetime(df_train["ref_datetime"])
    df_train["dtm"] = pd.to_datetime(df_train["dtm"])
    df_forecast["ref_datetime"] = pd.to_datetime(df_forecast["ref_datetime"])

    # Gültigen Zeitpunkt für Wettervorhersagen berechnen
    df_forecast["valid_datetime"] = df_forecast["ref_datetime"] + pd.to_timedelta(df_forecast["valid_time"], unit="h")

    df_merged = pd.merge(
        df_train,
        df_forecast,
        left_on=["dtm", "ref_datetime"],
        right_on=["valid_datetime", "ref_datetime"],
        how="inner"
    )

    return df_merged

def remove_NaN_rows(df_train: pd.DataFrame, df_forecast: pd.DataFrame) -> pd.DataFrame:
    # remove NaN values in df_train
    df_train = df_train.dropna(subset=["Solar_MWh"])

    # remove NaN values
    df_forecast = df_forecast.dropna(subset=["SolarDownwardRadiation"])
    df_forecast = df_forecast.dropna(subset=["CloudCover"])
    df_forecast = df_forecast.dropna(subset=["Temperature"])

    # remove negative values
    df_forecast["SolarDownwardRadiation"] = df_forecast["SolarDownwardRadiation"].clip(lower=0)

    return df_train, df_forecast