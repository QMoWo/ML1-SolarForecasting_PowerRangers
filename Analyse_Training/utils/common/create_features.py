import pytz
import numpy as np
from datetime import datetime
import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction
from astral.sun import elevation, azimuth
from astral import Observer
from astral import LocationInfo
from astral.sun import sun
from statsmodels.tsa.seasonal import STL

async def create_rolling_avg(df):
    df['valid_datetime'] = pd.to_datetime(df['valid_datetime'])

    median_df = df.groupby('valid_datetime')['SolarDownwardRadiation'].median().reset_index()
    median_df.rename(columns={'SolarDownwardRadiation': 'Median_SolarDownwardRadiation'}, inplace=True)

    # Schritt 2: Merge die Medianwerte zurück ins Original-DataFrame
    df = df.merge(median_df, on='valid_datetime', how='left')

    # Schritt 3: Berechne den Rolling Average von 'SolarDownwardRadiation'
    # Zuerst nach valid_datetime sortieren, um den rolling average korrekt zu berechnen
    df = df.sort_values(by='valid_datetime')

    # Rolling Average (hier mit einem Fenster von 3, je nach Bedarf anpassen)
    df['Rolling_Avg_SolarDownwardRadiation'] = df['SolarDownwardRadiation'].rolling(window=4).mean()

    # ersetze die NaN-Werte mit einem Fallback-Wert (z.B. dem Median)
    df['Rolling_Avg_SolarDownwardRadiation'].fillna(df['Median_SolarDownwardRadiation'], inplace=True)

    df['rolling_mean_radiation'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1, center=True).mean()
    df['rolling_mean_temperature'] = df['Temperature'].rolling(window=3, min_periods=1, center=True).mean()
    df['radiation_rolling_std'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1).std().fillna(0)
    df['radiation_rolling_max'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1).max()

    df['radiation_lag1'] = df['SolarDownwardRadiation'].shift(1).fillna(0)
    df['radiation_lag2'] = df['SolarDownwardRadiation'].shift(2).fillna(0)
    df['temp_lag1'] = df['Temperature'].shift(1).fillna(0)

    return df


async def create_seasons(df):
    # Jahreszeit bestimmen basierend auf spezifischen Daten
    def determine_season(date):
        tz = date.tzinfo  # Behalte die Zeitzone des Datums bei
        if date >= datetime(date.year, 12, 21, tzinfo=tz) or date < datetime(date.year, 3, 20, tzinfo=tz):
            return 'Winter'
        elif date >= datetime(date.year, 3, 20, tzinfo=tz) and date < datetime(date.year, 6, 21, tzinfo=tz):
            return 'Frühling'
        elif date >= datetime(date.year, 6, 21, tzinfo=tz) and date < datetime(date.year, 9, 23, tzinfo=tz):
            return 'Sommer'
        else:
            return 'Herbst'

    # Wende die Funktion auf die Spalte 'dtm' an
    df['Season'] = df['dtm'].apply(determine_season)
    return df

# create adjusted temperature feature and adjustes SolarDownwardRadiation feature
async def create_adjusted_values(df):
    # 3. adjusted_radiation berechnen
    df['adjusted_radiation'] = df['SolarDownwardRadiation'] * (1 - df['CloudCover'])

    # 4. adjusted_temperature berechnen
    df['adjusted_temperature'] = df['Temperature'] * df['CloudCover']

    df['Temperature_sqaured'] = df['Temperature'] ** 2

    df['solar_efficiency'] = df["SolarDownwardRadiation"] * df["Solar_capacity_mwp"]
    df['solar_temperature'] = df["Temperature"] * df["Solar_capacity_mwp"]
    df['SolarDownwardRadiation_CloudCover'] = df['SolarDownwardRadiation'] * df['CloudCover']
    df['SolarDownwardRadiation_Temperature'] = df['SolarDownwardRadiation'] * df['Temperature']

    #log temperature
    # if Temperature is 0, log1p(0) = 0
    #df['Log_Temperature'] = np.log1p(df['Temperature'] + 1)

    #df['Log_adjusted_temperature'] = np.log1p(df['adjusted_temperature'] + 1)

    #df['Logadjusted_radiation'] = np.log1p(df['adjusted_radiation'] + 1)

    # df['Log_Solar_MWh'] = np.log1p(df['Solar_MWh'])



    return df

async def check_if_in_daylight(df):
    # Erstellen Sie eine leere Liste, um die Ergebnisse zu speichern
    earliest_hours = []
    latest_hours = []
    is_in_daylight = []

    # Um alle Zeilen nach Datum zu gruppieren (dtm ist das Zeitstempel-Datum)
    for date, group in df.groupby(df['dtm'].dt.date):
        # Filtern der Zeilen für den aktuellen Tag, bei denen SolarDownwardRadiation > 2.0
        radiation_above_2 = group[group['SolarDownwardRadiation'] > 2.0]

        # Falls keine Stunden mit SolarDownwardRadiation > 2.0 existieren, setzen wir NaT für früheste und späteste Stunde
        if not radiation_above_2.empty:
            # Früheste Stunde (erste Zeile)
            earliest_hour = radiation_above_2['dtm'].min().hour
            # Späteste Stunde (letzte Zeile)
            latest_hour = radiation_above_2['dtm'].max().hour
        else:
            earliest_hour = None
            latest_hour = None

        # Hinzufügen der frühesten und spätesten Stunden für alle Zeilen des Tages
        earliest_hours.extend([earliest_hour] * len(group))
        latest_hours.extend([latest_hour] * len(group))

        # Berechnen, ob dtm innerhalb des Intervalls liegt
        for dtm in group['dtm']:
            is_in_daylight.append(earliest_hour is not None and earliest_hour <= dtm.hour <= latest_hour)

    # Neue Spalten in das DataFrame einfügen
    df['earliest_hour'] = earliest_hours
    df['latest_hour'] = latest_hours
    df['is_in_daylight'] = is_in_daylight

    # Umwandeln der Stunden in numerische Werte, falls sie als Categorical vorliegen
    df['earliest_hour'] = pd.to_numeric(df['earliest_hour'], errors='coerce')
    df['latest_hour'] = pd.to_numeric(df['latest_hour'], errors='coerce')

    df['is_in_daylight'] = df['is_in_daylight'].astype('category')
    df['earliest_hour'] = df['earliest_hour'].astype('category')
    df['latest_hour'] = df['latest_hour'].astype('category')

    return df


# encode datetime features
def create_cyclical_features(df, col, period):
    df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/period)
    df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/period)
    return df

# create from dtm hour, month columns as features
async def create_datetime_features(df):
    """
    Erstelle Datums- und Zeit-Features inklusive zyklischer Kodierung für Stunden und Monate.
    """

    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Stelle sicher, dass 'dtm' eine Pandas DatetimeSpalte ist
    if not isinstance(df['dtm'], pd.Series):
        df['dtm'] = pd.to_datetime(df['dtm'])

    # Erstelle Stunde und Monat als Features
    df['hour'] = df['dtm'].dt.hour
    df['month'] = df['dtm'].dt.month
    df['week'] = df['dtm'].dt.isocalendar().week
    df['day_of_year'] = df['dtm'].dt.dayofyear
    df['year'] = df['dtm'].dt.year

    df = create_cyclical_features(df, 'hour', 24)
    df = create_cyclical_features(df, 'month', 12)
    df = create_cyclical_features(df, 'week', 52)
    df = create_cyclical_features(df, 'day_of_year', 365)

    rbf_hour = RepeatingBasisFunction(
        n_periods=3,
        remainder="drop",
        column="hour",
        input_range=(0, 23)
    )
    hour_features = rbf_hour.fit_transform(df[['hour']])
    for i in range(hour_features.shape[1]):
        df[f'rbf_hour_{i}'] = hour_features[:, i]

    rbf_month = RepeatingBasisFunction(
        n_periods=3,
        remainder="drop",
        column="month",
        input_range=(1, 12)
    )
    month_features = rbf_month.fit_transform(df[['month']])
    for i in range(month_features.shape[1]):
        df[f'rbf_month_{i}'] = hour_features[:, i]


    rbf_week = RepeatingBasisFunction(
        n_periods=3,
        remainder="drop",
        column="week",
        input_range=(1, 52)
    )
    week_features = rbf_week.fit_transform(df[['week']])
    for i in range(week_features.shape[1]):
        df[f'rbf_week_{i}'] = week_features[:, i]

    df['temp_hour_sin'] = df['Temperature'] * df['hour_sin']
    df['temp_hour_cos'] = df['Temperature'] * df['hour_cos']

    df['cloud_hour_sin'] =  df['hour_sin'] * (1-df['CloudCover'])
    df['cloud_hour_cos'] = df['hour_cos'] * (1-df['CloudCover'])

    df['hour'] = df['hour'].astype('category')
    df['month'] = df['month'].astype('category')
    df['week'] = df['week'].astype('category')
    # df['day_of_year'] = df['day_of_year'].astype('category')
    # day of the year ist nicht gut, wahrscheinlich wegen overfitting

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    return df

# create rolling avergae of SolarDownwardRadiation and Temperature
async def create_rolling_average(df):
    # Berechne den rollenden Durchschnitt für SolarDownwardRadiation und Temperature
    df['rolling_mean_radiation'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1, center=True).mean()
    df['rolling_mean_temperature'] = df['Temperature'].rolling(window=3, min_periods=1, center=True).mean()
    df['radiation_rolling_std'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1).std().fillna(0)
    df['radiation_rolling_max'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1).max().fillna(0)
    return df

async def create_lag_features(df):
    df = df.sort_values('dtm')

    df['radiation_lag1'] = df['SolarDownwardRadiation'].shift(1).fillna(0)
    df['radiation_lag2'] = df['SolarDownwardRadiation'].shift(2).fillna(0)
    df['temp_lag1'] = df['Temperature'].shift(1).fillna(0)

    return df

async def astral(df):
    city = LocationInfo("London", "England", "Europe/London", 51.5, -0.116)
    features = ["sunrise", "sunset", "dawn", "dusk", "noon"]
    for feature in features:
        df[feature] = df["dtm"].apply(lambda x: sun(city.observer, x, tzinfo=city.timezone).get(feature))
        df[feature] = pd.to_datetime(df[feature], errors='coerce')
        df[feature] = df[feature].dt.round('h')
        df[feature] = df[feature].dt.hour
    df['sun_altitude'] = df["dtm"].apply(lambda x: elevation(city.observer, x))
    df['sun_altitude_sin'] = np.sin(np.radians(df['sun_altitude']))
    df['sun_altitude_cos'] = np.cos(np.radians(df['sun_altitude']))
    df['sun_azimuth'] = df["dtm"].apply(lambda x: azimuth(city.observer, x))
    # transformierung eines zyklischen Zusammenhangs
    df['sun_azimuth_sin'] = np.sin(np.radians(df['sun_azimuth']))
    df['sun_azimuth_cos'] = np.cos(np.radians(df['sun_azimuth']))

    df['sunrise'] = df['sunrise'].astype('category')
    df['sunset'] = df['sunset'].astype('category')
    return df

async def split_datetime(df):
    time_columns = ["dtm", "ref_energy", "valid_time", "ref_weather"]
    for column in time_columns:
        df[f"{column}_hour"] = df[column].dt.hour
        df[f"{column}_day"] = df[column].dt.day
        df[f"{column}_month"] = df[column].dt.month
        df[f"{column}_year"] = df[column].dt.year
        df[f"{column}_week"] = df[column].dt.isocalendar().week
        df[f"{column}_weekday"] = df[column].dt.weekday
        df[f"{column}_quarter"] = df[column].dt.quarter
    return df

async def deseasonalized_temperature(df):
    period = 365
    stl_temperature = STL(df['Temperature'], seasonal=13, period=period)
    result_temperature = stl_temperature.fit()
    df["seasonal_temperature"] = result_temperature.seasonal
    df['seasonal_temperature_sin'] = np.sin(np.radians(df['seasonal_temperature']))
    df['seasonal_temperature_cos'] = np.cos(np.radians(df['seasonal_temperature']))
    return df

async def forecast_diff(df):

    df["energy_diff"] = (df['dtm'] - df['ref_energy']).dt.total_seconds() / 3600
    df['energy_diff'] = df['energy_diff'].astype(int)

    return df
