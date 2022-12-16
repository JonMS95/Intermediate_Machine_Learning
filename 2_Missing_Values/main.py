# Sometimes, datasets that include empty data cells have to be used. There are some ways to cope with this kind of issues. In this chapter, we will see the most
# common ones.
# 1-Drop columns with missing values: it's not the most suitable one, as a whole feature gets removed from the dataset. For example, while analyzing real estate
# data, dropping a column may imply removing data about the usable surface extension in a home/flat.
# 2-Imputation: instead of removing any value, some values may be introduced arbitrarily within the dataset. And yes, the data can be made up, but the dataset
# (and models that use the dataset in question) usually behave better than those which just remove the whole feature from the dataset. The inputted value can
# be something like a mean value of the other values found in the same dataset column.
# 3-Imputation with extra data: sometimes, which values were originally missing has to be taken into account. To do so, another column may be added next to the
# one that has missing values within it, telling if that row within that column included data originally or not.

from constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

melb_data = pd.read_csv(PATH_MELBOURNE_CSV)

# Target is going to be the price, as usually.
y = melb_data.Price

# First, drop the price column (axis = 1 to select columns, axis = 0 to select rows), same as with dropna function.
melb_pred = melb_data.drop(['Price'], axis = 1)
# Then, select data types to exclude (or include) in the DataFrame. In this case, exclude = 'object' is being used. As explained in the pandas library documentation,
# 'object' data type is used to refer to strings. Therefore, every numerical value is going to be taken as a feature in this case.
# The aproach below is the same as choosing by hand all the numerical features, then getting them from the dataset.
X = melb_pred.select_dtypes(exclude = 'object')

# Divide the data into training and validation subsets:
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = SIZE_TRAIN, test_size = SIZE_VAL, random_state = SPLIT_RANDOM_STATE)

# Now a function that measures how good a dataset performs is written.
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators = 10, random_state = RFR_RANDOM_STATE)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)

# Save (in a variable) the names of every column number which includes at least a single missing value.
cols_with_missing = []

for column in X_train:
    if X_train[column].isnull().values.any():
        cols_with_missing.append(column)

# # Drop columns in training and validation data
# reduced_X_train = X_train.drop(missing_value_columns, axis=1)
# reduced_X_valid = X_valid.drop(missing_value_columns, axis=1)

# 1st approach: remove columns that contain at least a single missing value.
reduced_X_train = X_train.dropna(axis=1)
reduced_X_valid = X_valid.dropna(axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

# 2nd approach: use a simple input.
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))