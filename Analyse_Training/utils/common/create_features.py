import pytz
import numpy as np
from datetime import datetime
import pandas as pd
from sklego.preprocessing import RepeatingBasisFunction

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
    df['Rolling_Avg_SolarDownwardRadiation'] = df['SolarDownwardRadiation'].rolling(window=3).mean()

    # ersetze die NaN-Werte mit einem Fallback-Wert (z.B. dem Median)
    df['Rolling_Avg_SolarDownwardRadiation'].fillna(df['Median_SolarDownwardRadiation'], inplace=True)

    print(df.head(50))

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
    df['adjusted_temperature'] = df['Temperature'] / (1 + df['CloudCover'])

    df['solar_efficiency'] = df["SolarDownwardRadiation"] * df["Solar_capacity_mwp"]

    df['Temperature_squared'] = df['Temperature'] ** 2

    df['SolarDownwardRadiation_CloudCover'] = df['SolarDownwardRadiation'] * df['CloudCover']

    #log temperature
    # if Temperature is 0, log1p(0) = 0
    df['Log_Temperature'] = np.log1p(df['Temperature'] + 1)

    df['Log_adjusted_temperature'] = np.log1p(df['adjusted_temperature'] + 1)

    df['Logadjusted_radiation'] = np.log1p(df['adjusted_radiation'] + 1)

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

    # Zyklische Kodierung für die Stunden (24 Stunden Zyklus)
    df['earliest_hour_sin'] = np.sin(2 * np.pi * df['earliest_hour'] / 24)
    df['earliest_hour_cos'] = np.cos(2 * np.pi * df['earliest_hour'] / 24)

    df['latest_hour_sin'] = np.sin(2 * np.pi * df['latest_hour'] / 24)
    df['latest_hour_cos'] = np.cos(2 * np.pi * df['latest_hour'] / 24)

    df['is_in_daylight'] = df['is_in_daylight'].astype('category')
    df['earliest_hour'] = df['earliest_hour'].astype('category')
    df['latest_hour'] = df['latest_hour'].astype('category')
    
    return df

# encode datetime features

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
    df['day'] = df['dtm'].dt.day

    # Zyklische Kodierung für Stunde (24 Stunden Zyklus)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Zyklische Kodierung für Monat (12 Monate Zyklus)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Optional: Falls Wochen in zyklischer Form benötigt werden (52 Wochen Zyklus)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

    # Verfeinerte Kodierung unter Berücksichtigung von Stunde und Woche
    # Kombinierte Zyklische Kodierung für Woche und Stunde
    df['week_hour_combined_sin'] = np.sin(2 * np.pi * (df['week'] + df['hour'] / 24) / 52)  # Woche+Stunde
    df['week_hour_combined_cos'] = np.cos(2 * np.pi * (df['week'] + df['hour'] / 24) / 52)

    df['day_hour_combined_sin'] = np.sin(2 * np.pi * (df['day'] + df['hour'] / 24) / 365)  # Jahr: 365 Tage
    df['day_hour_combined_cos'] = np.cos(2 * np.pi * (df['day'] + df['hour'] / 24) / 365)

    # Alternativ: Falls du auch die Tageszeit innerhalb des Monats berücksichtigen möchtest
    df['day_in_month_sin'] = np.sin(2 * np.pi * (df['week'] * 7 + df['hour'] / 24) / 365)  # Jahr: 365 Tage
    df['day_in_month_cos'] = np.cos(2 * np.pi * (df['week'] * 7 + df['hour'] / 24) / 365)

    df['hour_day_week_month_combined_sin'] = np.sin(2 * np.pi * (df['hour'] + df['day'] + df['week'] + df['month'] * 30) / (24 + 30 + 7 + 12 * 30))
    df['hour_day_week_month_combined_cos'] = np.cos(2 * np.pi * (df['hour'] + df['day'] + df['week'] + df['month'] * 30) / (24 + 30 + 7 + 12 * 30))

    # Entferne die originalen "raw" Features, wenn sie nicht mehr benötigt werden
    # df.drop(columns=['hour', 'month', 'week'], inplace=True)

    rbf_hour = RepeatingBasisFunction(
        n_periods=1,  # Zyklus für Stunden (24 Stunden)
        remainder="drop",  # Behalte andere Spalten
        column="hour",  # Die Spalte, die transformiert wird
        input_range=(0, 23)  # Bereich für Stunden
    )
    rbf_hour.fit(df)
    np_array = rbf_hour.fit_transform(df[['hour']])
    df['rbf_hour'] = np_array[:, 0]

    print(df.head())

    rbf_month = RepeatingBasisFunction(
        n_periods=1,  # Zyklus für Monate (12 Monate)
        remainder="drop",  # Behalte andere Spalten
        column="month",  # Die Spalte, die transformiert wird
        input_range=(1, 12)  # Bereich für Monate
    )
    rbf_month.fit(df)
    np_array = rbf_month.fit_transform(df[['month']])
    df['rbf_month'] = np_array[:, 0]


    rbf_week = RepeatingBasisFunction(
        n_periods=1,  # Zyklus für Wochen (52 Wochen)
        remainder="drop",  # Behalte andere Spalten
        column="week",  # Die Spalte, die transformiert wird
        input_range=(1, 52)  # Bereich für Wochen
    )
    rbf_week.fit(df)
    np_array = rbf_week.fit_transform(df[['week']])
    df['rbf_week'] = np_array[:, 0]

    df['hour'] = df['hour'].astype('category')
    df['month'] = df['month'].astype('category')
    df['week'] = df['week'].astype('category')

    # Stelle sicher, dass der DataFrame immer noch ein Pandas DataFrame ist
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    
    return df

# create rolling avergae of SolarDownwardRadiation and Temperature
async def create_rolling_average(df):
    # Berechne den rollenden Durchschnitt für SolarDownwardRadiation und Temperature
    df['rolling_mean_radiation'] = df['SolarDownwardRadiation'].rolling(window=3, min_periods=1).mean()
    df['rolling_mean_temperature'] = df['Temperature'].rolling(window=3, min_periods=1).mean()
    
    return df
