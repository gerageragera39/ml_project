import pandas as pd
import matplotlib.pyplot as plt
from Prepare import change_type

if __name__ == '__main__':
    data = pd.read_csv('../Data/train.csv')
    X, num = change_type(data)

    Y = data['SalePrice']

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))
    features = [['LotArea', 'MSSubClass'], ['MSZoning', 'OverallQual'], ['YearBuilt', 'YearRemodAdd']]
    for index_raw, raw in enumerate(features):
        for index_col, col in enumerate(raw):
            axes[index_raw, index_col].scatter(X[features[index_raw][index_col]], Y, color = 'red')
            axes[index_raw, index_col].set_title(features[index_raw][index_col])
            if features[index_raw][index_col] == 'MSSubClass':
                axes[index_raw, index_col].set_xticks([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190])
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18))

    for index_raw, raw in enumerate(features):
        for index_col, col in enumerate(raw):
            numerical_data = pd.DataFrame(X[features[index_raw][index_col]]).select_dtypes(include=[float, int])
            axes[index_raw, index_col].boxplot(numerical_data)
            axes[index_raw, index_col].set_title(features[index_raw][index_col])

    plt.tight_layout()
    plt.show()