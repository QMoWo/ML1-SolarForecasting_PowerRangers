import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns

def plot_solar_generation_by_date(df, selected_date):
    """
    Erstellt Plots für die solare Stromerzeugung an einem spezifischen Datum (über mehrere Jahre).
    
    Parameter:
    - df (pd.DataFrame): Der DataFrame mit den Energie-Daten. Muss eine Spalte 'dtm' und 'Solar_MWh' enthalten.
    - selected_date (str): Das Datum im Format 'MM-TT', z.B. '01-29'.
    
    Rückgabe:
    - None: Zeigt die Plots an.
    """
    # Sicherstellen, dass 'dtm' ein datetime-Objekt ist
    if not pd.api.types.is_datetime64_any_dtype(df['dtm']):
        df['dtm'] = pd.to_datetime(df['dtm'])

    # Daten für das angegebene Datum filtern
    filtered_dates = df[df['dtm'].dt.strftime('%m-%d') == selected_date]

    # Gruppen nach Jahr erstellen
    date_groups = filtered_dates.groupby(filtered_dates['dtm'].dt.year)

    # Anzahl der Subplots bestimmen
    n_plots = len(date_groups)
    n_cols = 2
    n_rows = math.ceil(n_plots / n_cols)

    # Subplots erstellen
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
    axes = axes.flatten()  # Achsen in ein 1D-Array umwandeln, um sie leichter anzusprechen

    for idx, (year, group) in enumerate(date_groups):
        # Daten für den Plot vorbereiten
        group = group[group['Solar_MWh'] > 0.0]  # Nur Werte > 0.0 verwenden

        # Plot erstellen
        axes[idx].plot(group['dtm'], group['Solar_MWh'], marker='o', label=f'{year}')
        axes[idx].set_title(f'Stromerzeugung am {selected_date} ({year})', fontsize=14)
        axes[idx].set_xlabel('Uhrzeit', fontsize=12)
        axes[idx].set_ylabel('Solarerzeugung (MWh)', fontsize=12)
        axes[idx].grid(alpha=0.3)
        axes[idx].legend(fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)

    # Unbenutzte Subplots entfernen
    for idx in range(n_plots, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.show()


def plot_solar_generation_over_time(df):
    """
    Visualisiert den Verlauf der solaren Stromerzeugung (Solar_MWh) über die gesamte Zeit im DataFrame.
    
    Parameter:
    - df (pd.DataFrame): Der DataFrame mit den Energie-Daten. Muss Spalten 'dtm' und 'Solar_MWh' enthalten.
    
    Rückgabe:
    - None: Zeigt die Visualisierung an.
    """
    # Sicherstellen, dass 'dtm' ein datetime-Objekt ist
    if not pd.api.types.is_datetime64_any_dtype(df['dtm']):
        df['dtm'] = pd.to_datetime(df['dtm'])

    # Daten vorbereiten
    df_sorted = df.sort_values(by='dtm')  # Sortieren nach Datum/Zeit
    
    # Plot erstellen
    plt.figure(figsize=(15, 6))
    plt.plot(df_sorted['dtm'], df_sorted['Solar_MWh'], label='Solarerzeugung (MWh)', color='orange', alpha=0.8)
    plt.title('Verlauf der solaren Stromerzeugung über die Zeit', fontsize=16)
    plt.xlabel('Zeit', fontsize=14)
    plt.ylabel('Solarerzeugung (MWh)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_solar_generation_seasonal_heatmap(df):
    """
    Visualisiert die zeitliche Verteilung der solaren Stromerzeugung pro Jahreszeit als Heatmap.
    
    Parameter:
    - df (pd.DataFrame): Der DataFrame mit den Energie-Daten. Muss Spalten 'dtm' und 'Solar_MWh' enthalten.
    
    Rückgabe:
    - None: Zeigt die Heatmap an.
    """
    # Sicherstellen, dass 'dtm' ein datetime-Objekt ist
    if not pd.api.types.is_datetime64_any_dtype(df['dtm']):
        df['dtm'] = pd.to_datetime(df['dtm'])

    # Jahreszeit bestimmen
    df['Season'] = df['dtm'].apply(lambda x: 'Winter' if x.month in [12, 1, 2] else
                                              ('Frühling' if x.month in [3, 4, 5] else
                                              ('Sommer' if x.month in [6, 7, 8] else 'Herbst')))
    
    # Reihenfolge der Jahreszeiten definieren
    season_order = ['Winter', 'Frühling', 'Sommer', 'Herbst']
    df['Season'] = pd.Categorical(df['Season'], categories=season_order, ordered=True)

    # Extrahieren der Stunde und Aggregation der Solarenergie nach Jahreszeit und Stunde
    df['hour'] = df['dtm'].dt.hour
    df_grouped = df.pivot_table(index='Season', columns='hour', values='Solar_MWh', aggfunc='mean')

    # Heatmap erstellen
    plt.figure(figsize=(18, 8))
    sns.heatmap(df_grouped, cmap='YlOrRd', cbar_kws={'label': 'Solarerzeugung (MWh)'}, annot=True, fmt='.2f')
    plt.title('Heatmap der solaren Stromerzeugung nach Jahreszeiten und Stunden', fontsize=16)
    plt.xlabel('Stunde des Tages', fontsize=14)
    plt.ylabel('Jahreszeit', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
