import pandas as pd
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np
from Prepare import data, df_test
from Prepare import Y as prices
from Prepare import scaler


models = {
    'Linear Regression': LinearRegression(),
    'k-Nearest Neighbors': KNeighborsRegressor(),
    'Support Vector Machine': SVR(),
    'Random Forest': RandomForestRegressor(),
    'Neural Network': MLPRegressor()
}

rf_params = {
    'n_estimators': [50, 100, 200],
    'max_features': [1.0, 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['squared_error', 'absolute_error']
}

gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 5, 6, 7, 8],
    'loss': ['huber', 'absolute_error', 'squared_error', 'quantile']
}


def train(x, y):
    results = {}
    results_meanstd = {}
    for model_name, model in models.items():
        model.fit(x, y)
        scores = cross_val_score(model, x, y, cv=5)
        results[model_name] = scores.mean()
        results_meanstd[model_name] = {'mean': scores.mean(), 'std': scores.std()}
        for model_name, score in results.items():
            print(f'{model_name}: {score}')
        print('\n\n')

    for model_name, score in results_meanstd.items():
        print(f'{model_name}: Mean={score["mean"]}, Std={score["std"]}')
    print('\n\n')


def analyze_errors(m, m_name, x, y):
    # x_scaled = scaler.fit_transform(x)
    predicted_values = m.predict(x)
    errors = predicted_values - y

    print(m_name)
    print("Mean Error:", np.mean(errors))
    print("Standard Deviation of Errors:", np.std(errors))
    print("Maximum Error:", np.max(errors))
    print("Minimum Error:", np.min(errors))
    print('_______________________________________________________')


def find_error(x, y, columns):
    for model_name, model in models.items():
        model.fit(x, y)
        analyze_errors(model, model_name, x, y)

    for i in range(2):
        selected_features = x
        scaled_features = scaler.fit_transform(selected_features)
        x = pd.DataFrame(scaled_features, columns=columns)
        for model_name, model in models.items():
            model.fit(x, y)
            analyze_errors(model, model_name, x, y)
    return models


def grid_searching(x, y):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, scoring=scorer, cv=5, n_jobs=-1)
    gb_grid = GridSearchCV(GradientBoostingRegressor(), gb_params, scoring=scorer, cv=5, n_jobs=-1)
    rf_grid.fit(x, y)
    gb_grid.fit(x, y)
    print('Random Forest: Best Parameters', rf_grid.best_params_)
    print('Random Forest: Best Score', rf_grid.best_score_)
    print('Gradient Boosting: Best Parameters', gb_grid.best_params_)
    print('Gradient Boosting: Best Score', gb_grid.best_score_)
    return gb_grid, rf_grid


if __name__ == '__main__':
    df_price = pd.read_csv('Data/sample_submission.csv')
    data_scaled = scaler.transform(data)

    train(data, prices)
    find_error(data, prices, data.columns)
    gb, rf = grid_searching(data, prices)

    X_test = df_test
    X_test_scaled = scaler.transform(X_test)
    model = models['Linear Regression']
    model.fit(data, prices)

    predictions_lin_reg = model.predict(X_test_scaled)
    predictions_gb = gb.predict(X_test_scaled)
    predictions_rf = rf.predict(X_test_scaled)

    df_predictions_lin_reg = pd.DataFrame({'PredictedPriceLinReg': predictions_lin_reg})
    df_predictions_gb = pd.DataFrame({'PredictedPriceGB': predictions_gb})
    df_predictions_rf = pd.DataFrame({'PredictedPriceRF': predictions_rf})

    df_predictions_lin_reg.to_csv('df_predictions_lin_reg.csv', index=False)
    df_predictions_gb.to_csv('df_predictions_gb.csv', index=False)
    df_predictions_rf.to_csv('df_predictions_rf.csv', index=False)

    result = df_price.join(df_predictions_lin_reg)
    result = result.join(df_predictions_gb)
    result = result.join(df_predictions_rf)
    result.to_csv('result.csv', index=False)

