import numpy as np
from datetime import datetime
import pandas as pd
from astral.sun import elevation, azimuth
from astral import Observer
from astral import LocationInfo
from astral.sun import sun
from statsmodels.tsa.seasonal import STL

async def create_rolling_avg(df):
    df['valid_datetime'] = pd.to_datetime(df['valid_datetime'])

    median_df = df.groupby('valid_datetime')['SolarDownwardRadiation'].median().reset_index()
    median_df.rename(columns={'SolarDownwardRadiation': 'Median_SolarDownwardRadiation'}, inplace=True)
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

def assign_time_period(hour):
    if 5 <= hour < 9:
        return 'Morgen'
    elif 9 <= hour < 12:
        return 'Vormittag'
    elif 12 <= hour < 15:
        return 'Mittag'
    elif 15 <= hour < 18:
        return 'Nachmittag'
    elif 18 <= hour < 22:
        return 'Abend'
    else:
        return 'Nacht'

async def add_time_period_column(df, hour_column='hour', new_column='time_period'):
    # Anwenden der Funktion auf die Spalte
    df[new_column] = df['hour'].apply(assign_time_period)
    return df

async def shift_radiation(df):
    df['Solar_Radiation_lag_1h'] = df['SolarDownwardRadiation'].shift(1).fillna(0)
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
