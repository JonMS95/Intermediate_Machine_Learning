# This exercise starts by taking an overview of the last exercise done in the begginer level repo.
from constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read the data. In this case, train.csv is for training, test.csv is for testing.
# read_csv function allows us to choose the column that's going to be used as index. In this case, 'Id' column, which, by the way, is the first one, is arguably
# the best choice, as it contains the row number.
train_data = pd.read_csv(PATH_TRAIN_CSV, index_col = 'Id')
test_data  = pd.read_csv(PATH_TEST_CSV, index_col = 'Id')

# Obtain target ('y') and features/predictors ('X').
y = train_data['SalePrice']
# y_test = test_data['SalePrice']
#If we print the test data columns, we will find no sale price series.
print(test_data.columns)

# Trivia fact: 'y' variable could have been declared as y = train_data.SalePrice as well. However, declaring it as it's done above is strongly recommended, because
# it covers a more general case. If the variable name has special characters within it, then the dot ('.') method, it's to say, accessing the variable as a member
# of a class will no longer be possible.

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
X_test = test_data[features]

# Until this point, y and X variables have been recorded for the train and test cases. Note that testing is not considered to be the same as validation.
# In fact, the next line is going to break y and X variables into training and validation (not testing) data chunks.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = SIZE_TRAIN, test_size = SIZE_VAL, random_state = SPLIT_RANDOM_STATE)

print(X_train[0:5])