import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
# Need to specify the headers for this dataset

cols = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
       "num_doors", "body_style", "drive_wheels", "engine_location",
       "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
       "num_cylinders", "engine_size", "fuel_system", "bore", "stroke",
       "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg",
       "price"]
cars = pd.read_csv("features.data", names=cols)
print(cars.dtypes)
print(cars.head())


cars = cars.replace('?', np.nan)

# Now lets make things numeric
num_vars = ['normalized_losses', "bore", "stroke", "horsepower", "peak_rpm",
            "price"]

for i in num_vars:
       cars[i] = cars[i].astype('float64')

print(cars.head())
print("normalized losses: ", cars['normalized_losses'].isnull().sum())


print(cars.isnull().sum())



cars = cars.dropna(subset = ['price'])
print(cars.isnull().sum())


cars = cars.dropna(subset = ['bore', 'stroke', 'horsepower', 'peak_rpm'])
print()
print(cars.isnull().sum())


cols = ['wheel_base', 'length', 'width', 'height',
        'curb_weight', 'engine_size', 'bore', 'stroke', 'horsepower',
        'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
cars = cars[cols]

normalized_cars = (cars - cars.mean()) / (cars.std())
print(normalized_cars)

# Writing a simple function that trains and tests univariate models
# This function takes in three arguments: the predictor, the outcome, & the data

def knn_train_test(train_col, target_col, df):
       knn = KNeighborsRegressor()
       np.random.seed(1)

       # Randomize order of rows in data frame.
       shuffled_index = np.random.permutation(df.index)
       rand_df = df.reindex(shuffled_index)

       # Divide number of rows in half and round.
       last_train_row = int(len(rand_df) / 2)

       # Select the first half and set as training set.
       # Select the second half and set as test set.
       train_df = rand_df.iloc[0:last_train_row]
       test_df = rand_df.iloc[last_train_row:]

       # Fit a KNN model using default k value.
       knn.fit(train_df[[train_col]], train_df[target_col])

       # Make predictions using model.
       predicted_labels = knn.predict(test_df[[train_col]])

       # Calculate and return RMSE.
       mse = mean_squared_error(test_df[target_col], predicted_labels)
       rmse = np.sqrt(mse)
       return rmse


# Lets test a couple of predictors
print('city mpg: ', knn_train_test('city_mpg', 'price', normalized_cars))
print('width: ', knn_train_test('width', 'price', normalized_cars))
print('highway mpg: ', knn_train_test('highway_mpg', 'price', normalized_cars))
print('engine size: ', knn_train_test('engine_size', 'price', normalized_cars))
print('horsepower: ', knn_train_test('horsepower', 'price', normalized_cars))

