from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
import numpy as np

def print_evaluation_simple_model(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, alpha, feature_names):
    """ Ausgabe von R2-Wert, MSE und MAE für Trainings- und Testset """
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Create markdown report
    markdown_content = (
        f"### {model} Evaluation\n\n"
        f"| Dataset | R² | RMSE | MAE | Rows | Columns |\n"
        f"|---------|--------:|------------:|--------:|-------:|-------:|\n"
        f"| Train   | {r2_train:.5f} | {rmse_train:.2f} | {mae_train:.2f} | {X_train.shape[0]} | {X_train.shape[1]} |\n"
        f"| Test    | {r2_test:.5f} | {rmse_test:.2f} | {mae_test:.2f} | {X_test.shape[0]} | {X_test.shape[1]} |\n\n"
    )

    # Sort and display the top 10 coefficients
    coefficients_lr = pd.DataFrame({"Feature Name": feature_names, "Coefficient": model.coef_})
    grenzwert = 0.1
    top_features = coefficients_lr[abs(coefficients_lr["Coefficient"]) > grenzwert]
    top_features = top_features.reindex(top_features["Coefficient"].abs().sort_values(ascending=False).index)

    # Append top 10 coefficients to markdown
    markdown_content += "### Top 10 Coefficients\n\n"
    markdown_content += top_features.to_markdown(index=False)

    # Append count of zero coefficients
    zero_count = len(coefficients_lr[np.isclose(coefficients_lr.Coefficient, 0.0)])
    markdown_content += f"\n\nNumber of coefficients that are zero: {zero_count}/{len(coefficients_lr)}\n"

    # Append alpha value
    # markdown_content += f"\n\nAlpha value: {alpha}\n"

    return markdown_content

def print_evaluation_DTR(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, alpha, feature_names):
    """ Ausgabe von R2-Wert, RMSE und MAE für Trainings- und Testset mit Feature Importances für DecisionTreeRegressor """
    # Evaluation für Training und Test
    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)

    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Markdown Bericht generieren
    markdown_content = (
        f"### DecisionTreeRegressor Evaluation\n\n"
        f"| Dataset | R² | RMSE | MAE | Rows | Columns |\n"
        f"|---------|--------:|------------:|--------:|-------:|-------:|\n"
        f"| Train   | {r2_train:.5f} | {rmse_train:.2f} | {mae_train:.2f} | {X_train.shape[0]} | {X_train.shape[1]} |\n"
        f"| Test    | {r2_test:.5f} | {rmse_test:.2f} | {mae_test:.2f} | {X_test.shape[0]} | {X_test.shape[1]} |\n\n"
    )

    # Feature Importances
    feature_importances = model.feature_importances_

    # Erstelle ein DataFrame mit den Feature Importances
    importances_df = pd.DataFrame({
        "Feature Name": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Top 10 Feature Importances auswählen
    top_features = importances_df.head(10)

    # Markdown für die Feature Importances hinzufügen
    markdown_content += "### Top 10 Feature Importances\n\n"
    markdown_content += top_features.to_markdown(index=False)

    # Append Hinweis zur Summe der Importances
    total_importance = feature_importances.sum()
    markdown_content += f"\n\nTotal feature importance (should be 1.0): {total_importance:.2f}\n"

    return markdown_content


def print_evaluation_stacking(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, best_score, feature_names):
    """
    Print evaluation metrics for stacking model
    """
    # Calculate metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # Create markdown table
    markdown_content = (
        f"### Stacking Model Evaluation\n\n"
        f"| Dataset | R² | RMSE | MAE | Rows | Columns |\n"
        f"|---------|-------|------|-----|------|----------|\n"
        f"| Train   | {r2_train:.5f} | {rmse_train:.2f} | {mae_train:.2f} | {X_train.shape[0]} | {X_train.shape[1]} |\n"
        f"| Test    | {r2_test:.5f} | {rmse_test:.2f} | {mae_test:.2f} | {X_test.shape[0]} | {X_test.shape[1]} |\n\n"
    )

    # Add information about base learners
    markdown_content += (
        f"Base Learners:\n"
        f"- Ridge Regression\n"
        f"- DecisionTreeRegressor\n"
        f"- GradientBoost\n\n"
        f"Best CV Score (RMSE): {best_score:.2f}\n"
    )

    # # Feature Importances vom finalen Estimator extrahieren
    # if hasattr(model.final_estimator_, "feature_importances_"):
    #     feature_importances = model.final_estimator_.feature_importances_

    #     print(feature_importances)

    #     # Erstelle ein DataFrame mit den Feature Importances
    #     importances_df = pd.DataFrame({
    #         "Feature Name": feature_names,
    #         "Importance": feature_importances
    #     }).sort_values(by="Importance", ascending=False)

    #     # Top 10 Feature Importances auswählen
    #     top_features = importances_df.head(10)

    #     # Markdown für die Feature Importances hinzufügen
    #     markdown_content += "### Top 10 Feature Importances\n\n"
    #     markdown_content += top_features.to_markdown(index=False)

    #     # Append Hinweis zur Summe der Importances
    #     total_importance = feature_importances.sum()
    #     markdown_content += f"\n\nTotal feature importance (should be 1.0): {total_importance:.2f}\n"
    # else:
    #     markdown_content += "\n### Feature Importances\n\n"
    #     markdown_content += "The final estimator does not provide feature importances."

    return markdown_content
