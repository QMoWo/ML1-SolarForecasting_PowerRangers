import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_attribute_vs_label_filtered(merged_df):
    # Sicherstellen, dass der Datentyp von Solar_MWh korrekt ist
    merged_df['Solar_MWh'] = merged_df['Solar_MWh'].astype(float)

    # Entfernen von Zeilen, bei denen 'Season' den Wert 0.0 hat (nur für den Boxplot)
    merged_df = merged_df[merged_df['Season'] != '0.0']

    # Plotten der Korrelationen als Heatmap (optional)
    # Wenn du die Korrelationen auch anzeigen möchtest
    correlation_matrix = merged_df[['Solar_capacity_mwp', 'Solar_MWh', 'SolarDownwardRadiation', 'CloudCover', 'Temperature', 'hour']].corr()
    print("Korrelationsmatrix:")
    print(correlation_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Korrelationsmatrix")
    plt.show()

    # Scatter Plots: Solar_MWh vs. verschiedene Attribute
    plt.figure(figsize=(16, 10))

    # Solar_MWh vs SolarDownwardRadiation
    plt.subplot(2, 3, 1)
    sns.scatterplot(data=merged_df, x='SolarDownwardRadiation', y='Solar_MWh')
    plt.title('Solar_MWh vs SolarDownwardRadiation')

    # Boxplot für Season (ohne die Zeilen, wo Season == 0.0)
    plt.subplot(2, 3, 2)
    sns.boxplot(data=merged_df, x='Season', y='Solar_MWh')
    plt.title('Solar_MWh vs Season')

    # Solar_MWh vs Temperature
    plt.subplot(2, 3, 3)
    sns.scatterplot(data=merged_df, x='Temperature', y='Solar_MWh')
    plt.title('Solar_MWh vs Temperature')

    # SolarDownwardRadiation vs Temperature
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=merged_df, x='Temperature', y='SolarDownwardRadiation')
    plt.title('Temperature vs SolarDownwardRadiation')

    plt.tight_layout()
    plt.show()




