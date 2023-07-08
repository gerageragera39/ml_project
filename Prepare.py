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
        q1 = df[column].quantile(0.05)
        q3 = df[column].quantile(0.95)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df


df_test = pd.read_csv('Data/test.csv')
df_train = pd.read_csv('Data/train.csv')
df_price = pd.read_csv('Data/sample_submission.csv')

reject = ['Id', 'Street', 'Alley', 'Utilities', 'RoofStyle', 'BsmtFinSF1', 'BsmtFinType2',
          'BsmtFinType1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
          'GarageFinish', 'GarageCars', 'PavedDrive', 'Fence', 'MoSold', 'YrSold',
          'Fireplaces', 'FireplaceQu', 'MasVnrType', 'MasVnrArea', 'Condition1', 'Condition2', 'OverallQual',
          'OverallCond', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
          'GarageQual', 'GarageCond', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']

data = [df_train, df_test]
for df in data:
    df['CombinedCondition'] = df['Condition1'] + ' | ' + df['Condition2']
    df['CombinedOverallProperties'] = df['OverallQual'] + df['OverallCond']
    df['CombinedExterior'] = df['Exterior1st'] + ' | ' + df['Exterior2nd']
    df['CombinedExteriorProperties'] = df['ExterQual'] + ' | ' + df['ExterCond']
    df['CombinedBsmtrProperties'] = df['BsmtQual'] + ' | ' + df['BsmtCond']
    df['CombinedGarageProperties'] = df['GarageQual'] + ' | ' + df['GarageCond']
    df['PorchOrWoodArea'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df[
        'ScreenPorch']
df_train = df_train.drop(reject, axis=1)
df_test = df_test.drop(reject, axis=1)

df_train, numeric_columns_train = change_type(df_train)
df_test, numeric_columns_test = change_type(df_test)
df_train = fill_missing_values(df_train)
df_test = fill_missing_values(df_test)
df_train = remove_outliers(df_train)

Y = df_train['SalePrice']

df_train[numeric_columns_train] = scaler.fit_transform(df_train[numeric_columns_train])
df_test[numeric_columns_test] = scaler.fit_transform(df_test[numeric_columns_test])

data = df_train.drop(['SalePrice'], axis=1)

