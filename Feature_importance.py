import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from Prepare import data
from Prepare import Y as prices

if __name__ == '__main__':
    model = ExtraTreesRegressor()
    model.fit(data, prices)

    # Use inbuilt class feature_importances_ of tree based classifiers
    print(model.feature_importances_)

    # Define the hyperparameters and their values
    param_grid = {
        'max_depth': [80, 100, 120],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300]
    }

    model = ExtraTreesRegressor()

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1)
    grid_search.fit(data, prices)

    model = grid_search.best_estimator_
    print(model.feature_importances_)
    plt.figure(figsize=(10, 20), dpi=300)

    feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    feat_importances.nlargest(80).plot(kind='barh')

    plt.yticks(fontsize=5)
    plt.yticks(rotation=0)

    # Use tight_layout to avoid cutting off labels
    plt.tight_layout()

    plt.savefig('feature_importance.png', bbox_inches="tight")
    plt.show()
