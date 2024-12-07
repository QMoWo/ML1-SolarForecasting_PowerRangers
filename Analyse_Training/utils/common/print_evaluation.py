from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import pandas as pd
import numpy as np

def print_evaluation(model, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, alpha, feature_names):
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

    if hasattr(model.final_estimator_, 'coef_'):
        # Extrahieren der Koeffizienten aus dem Meta-Modell
        # coefficients_lr = pd.DataFrame({"Feature Name": feature_names, "Coefficient": model.final_estimator_.coef_})
        # top_features = coefficients_lr.reindex(coefficients_lr["Coefficient"].abs().sort_values(ascending=False).index).head(10)
        
        # # Top 10 Koeffizienten zum Bericht hinzufügen
        # markdown_content += "### Top 10 Coefficients\n\n"
        # markdown_content += top_features.to_markdown(index=False)
        
        # # Zählen der Koeffizienten, die null sind
        # zero_count = len(coefficients_lr[np.isclose(coefficients_lr.Coefficient, 0.0)])
        # markdown_content += f"\n\nNumber of coefficients that are zero: {zero_count}/{len(coefficients_lr)}\n"
        pass
    else:

        # Sort and display the top 10 coefficients
        coefficients_lr = pd.DataFrame({"Feature Name": feature_names, "Coefficient": model.coef_})
        top_features = coefficients_lr.reindex(coefficients_lr["Coefficient"].abs().sort_values(ascending=False).index).head(10)
        
        # Append top 10 coefficients to markdown
        markdown_content += "### Top 10 Coefficients\n\n"
        markdown_content += top_features.to_markdown(index=False)
        
        # Append count of zero coefficients
        zero_count = len(coefficients_lr[np.isclose(coefficients_lr.Coefficient, 0.0)])
        markdown_content += f"\n\nNumber of coefficients that are zero: {zero_count}/{len(coefficients_lr)}\n"

    # Append alpha value
    markdown_content += f"\n\nAlpha value: {alpha}\n"
    
    return markdown_content