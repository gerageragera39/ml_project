import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

label_encoder = LabelEncoder()
scaler = StandardScaler()


def change_type(df):
    data_types = df.dtypes
    numeric_columns = []
    for column, dtype in data_types.items():
        if dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])
            numeric_columns.append(column)
        else:
            numeric_columns.append(column)

    return df, numeric_columns


def fill_missing_values(df):
    filled_data = df.copy()
    for column in filled_data.columns:
        if filled_data[column].isnull().sum() > 0:
            dtype = filled_data[column].dtype
            if np.issubdtype(dtype, np.number):
                filled_data[column].fillna(filled_data[column].median(), inplace=True)
            elif dtype == object:
                filled_data[column].fillna(filled_data[column].mode().iloc[0], inplace=True)
            else:
                filled_data[column].fillna(0, inplace=True)
    return filled_data


def remove_outliers(df):
    for column in df.columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df
    # for column in df_train.columns:
    #     q1 = df[column].quantile(0.25)
    #     q3 = df[column].quantile(0.75)
    #     iqr = q3 - q1
    #     lower_bound = q1 - 1.5 * iqr
    #     upper_bound = q3 + 1.5 * iqr
    #     median = df[column].median()
    #     df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = median
    # return df

df_test = pd.read_csv('../Data/test.csv')
df_train = pd.read_csv('../Data/train.csv')
df_price = pd.read_csv('../Data/sample_submission.csv')

# df_train = df_train.drop(
#     ['LotFrontage', 'Street', 'LotShape', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#      'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#      'BsmtFinSF2',
#      'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
#      'BsmtHalfBath',
#      'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish',
#      'GarageCars',
#      'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Fence',
#      'MoSold', 'SaleCondition', 'Alley', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2',
#      'RoofMatl', 'ExterQual', 'ExterCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#      'BsmtFinSF2',
#      'BsmtUnfSF', 'CentralAir', 'Electrical', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
#      'Functional',
#      'Fireplaces', 'FireplaceQu', 'MiscFeature'], axis=1)
#
# df_test = df_test.drop(
#     ['LotFrontage', 'Street', 'LotShape', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#      'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#      'BsmtFinSF2',
#      'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
#      'BsmtHalfBath',
#      'FullBath', 'HalfBath', 'BedroomAbvGr', 'Fireplaces', 'FireplaceQu', 'GarageYrBlt', 'GarageFinish',
#      'GarageCars',
#      'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Fence',
#      'MoSold', 'SaleCondition', 'Alley', 'LandContour', 'Utilities', 'LandSlope', 'Condition1', 'Condition2',
#      'RoofMatl', 'ExterQual', 'ExterCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
#      'BsmtFinSF2',
#      'BsmtUnfSF', 'CentralAir', 'Electrical', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
#      'Functional',
#      'Fireplaces', 'FireplaceQu', 'MiscFeature'], axis=1)

df_train, numeric_columns_train = change_type(df_train)
df_test, numeric_columns_test = change_type(df_test)
df_train = fill_missing_values(df_train)
df_test = fill_missing_values(df_test)
df_train = remove_outliers(df_train)

Y = df_train['SalePrice']

df_train[numeric_columns_train] = scaler.fit_transform(df_train[numeric_columns_train])
df_test[numeric_columns_test] = scaler.fit_transform(df_test[numeric_columns_test])

data = df_train.drop(['SalePrice'], axis=1)


