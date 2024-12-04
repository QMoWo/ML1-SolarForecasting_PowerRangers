import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from utils.common.print_evaluation import print_evaluation


def train_model(test_df: pd.DataFrame, columns: list, y_column: str) -> tuple:
    """
    columns: list of columns to use for training
    y_column: column to predict
    """
    # one hot encode
    test_df = pd.get_dummies(test_df)

    # get target column and store it in y
    y = test_df.pop(y_column)
    
    # split data into X and y
    X_train, X_test, y_train, y_test = train_test_split(test_df[columns], 
                                                        y, 
                                                        test_size=0.2)


    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # evaluate model
    evaluation = print_evaluation(model, 
                                  X_train, 
                                  X_test, 
                                  y_train, 
                                  y_test, 
                                  y_train_pred, 
                                  y_test_pred, 
                                  0, 
                                  columns)
    
    return model, evaluation, scaler, test_df[columns].columns
    

def test_model(model, scaler, test_df, test_columns):
    # one hot encode
    test_df = pd.get_dummies(test_df)
    
    # scale features
    X = scaler.transform(test_df[test_columns])
    
    # make predictions
    y_pred = model.predict(X)

    test_df['Solar_MWh_pred'] = y_pred
    
    return test_df
    