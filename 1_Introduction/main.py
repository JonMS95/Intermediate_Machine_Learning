# This exercise starts by taking an overview of the last exercise done in the begginer level repo.
from constants import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the data. In this case, train.csv is for training, test.csv is for testing.
# read_csv function allows us to choose the column that's going to be used as index. In this case, 'Id' column, which, by the way, is the first one, is arguably
# the best choice, as it contains the row number.
train_data = pd.read_csv(PATH_TRAIN_CSV, index_col = 'Id')
test_data  = pd.read_csv(PATH_TEST_CSV, index_col = 'Id')

# Obtain target ('y') and features/predictors ('X').
y = train_data['SalePrice']
# y_test = test_data['SalePrice']
#If we print the test data columns, we will find no sale price series.
# print(test_data.columns)

# Trivia fact: 'y' variable could have been declared as y = train_data.SalePrice as well. However, declaring it as it's done above is strongly recommended, because
# it covers a more general case. If the variable name has special characters within it, then the dot ('.') method, it's to say, accessing the variable as a member
# of a class will no longer be possible.

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = train_data[features]
X_test = test_data[features]

# Until this point, y and X variables have been recorded for the train and test cases. Note that testing is not considered to be the same as validation.
# In fact, the next line is going to break y and X variables into training and validation (not testing) data chunks.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = SIZE_TRAIN, test_size = SIZE_VAL, random_state = SPLIT_RANDOM_STATE)

# Some models are going to be defined now, all of them based in the RFR structure.
# Some notes about the passed arguments:
#   ·n_estimators is the number of decision trees that each forest is going to use.
#   ·criterion is the way the quality of each split in each tree is measured. In this case, minimizing the MAE is wanted, getting the median of each terminal node.
#   ·max_depth indicates which the maximum depth of each tree should be.

model_1 = RandomForestRegressor(n_estimators = 50,                                                                  random_state = RFR_RANDOM_STATE)
model_2 = RandomForestRegressor(n_estimators = 100,                                                                 random_state = RFR_RANDOM_STATE)
model_3 = RandomForestRegressor(n_estimators = 100, criterion ='absolute_error',                                    random_state = RFR_RANDOM_STATE)
model_4 = RandomForestRegressor(n_estimators = 200,                                 min_samples_split = 20,         random_state = RFR_RANDOM_STATE)
model_5 = RandomForestRegressor(n_estimators = 100, max_depth = 7,                                                  random_state = RFR_RANDOM_STATE)

models = {
    model_1 : -1,
    model_2 : -1,
    model_3 : -1,
    model_4 : -1,
    model_5 : -1
    }

# Below, a function that gets a model and determines its mean absolute error is defined.
# Trivia fact: python allows the developer to set some defafult variables, by using var = default_value kind of statement when declaring the input parameters.
def score_model(model, X_t = X_train, X_v = X_valid, y_t = y_train, y_v = y_valid):
    model.fit(X_t, y_t)
    prediction = model.predict(X_v,)
    return mean_absolute_error(y_v, prediction)

for i in models.keys():
    MAE = score_model(i)
    print("Model: %d\t MAE %d" % (list(models.keys()).index(i) + 1, MAE))

# As seen, the third model seems to be the mos effective one. Thus, it is going to be stored in a variable. Note that the method that generates the random forest
# is called again, because the current third model is already fitted, and it's consequently biased.
best_model = RandomForestRegressor(n_estimators = 100, criterion ='absolute_error', random_state = RFR_RANDOM_STATE)

# Same as in the previous exercises (done in the begginer tutorial series), the whole data set (not only the train side) is used to fit the model again.
best_model.fit(X, y)

# Now, the model should be tested against the test data set (not the validation data set).