# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, Ridge, ElasticNet, LogisticRegression
# from sklearn.svm import SVR, LinearSVR
# from sklearn.ensemble import BaggingRegressor, StackingRegressor, RandomForestRegressor, VotingRegressor, AdaBoostRegressor
# import numpy as np
#
#
# def kn_regressor_train_test(train_col, target_col, df):
#     knn = KNeighborsRegressor()
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     knn.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = knn.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
# def linear_regression_train_test(train_col, target_col, df):
#     lr = LinearRegression()
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     lr.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = lr.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
# def ridge_regression_train_test(train_col, target_col, df):
#     ridge_regr = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     ridge_regr.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = ridge_regr.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
# def lasso_regression_train_test(train_col, target_col, df):
#     lasso_regr = Lasso(alpha=0.1)
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     lasso_regr.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = lasso_regr.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
# def elasticnet_regression_train_test(train_col, target_col, df):
#     regressor = ElasticNet()
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     regressor.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = regressor.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
#
#
# def bagging_regression_train_test(train_col, target_col, df):
#     regressor = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     regressor.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = regressor.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
#
#
# def stacking_regression_train_test(train_col, target_col, df):
#     estimators = [
#         ('lr', RidgeCV()),
#         ('svr', LinearSVR(random_state=42))
#     ]
#     regressor = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10,
#                                               random_state=42))
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     regressor.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = regressor.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
# def voting_regression_train_test(train_col, target_col, df):
#     r1 = LinearRegression()
#     r2 = RandomForestRegressor(n_estimators=10, random_state=1)
#     regressor = VotingRegressor([('lr', r1), ('rf', r2)])
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     regressor.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = regressor.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse
#
#
# def ada_boost_regression_train_test(train_col, target_col, df):
#     regressor = AdaBoostRegressor(random_state=0, n_estimators=100)
#     np.random.seed(1)
#
#     # Randomize order of rows in data frame.
#     shuffled_index = np.random.permutation(df.index)
#     rand_df = df.reindex(shuffled_index)
#
#     # Divide number of rows in half and round.
#     last_train_row = int(len(rand_df) / 2)
#
#     # Select the first half and set as training set.
#     # Select the second half and set as test set.
#     train_df = rand_df.iloc[0:last_train_row]
#     test_df = rand_df.iloc[last_train_row:]
#
#     # Fit a KNN model using default k value.
#     regressor.fit(train_df[[train_col]], train_df[target_col])
#
#     # Make predictions using model.
#     predicted_labels = regressor.predict(test_df[[train_col]])
#     # print(predicted_labels)
#
#     # Calculate and return RMSE.
#     mse = mean_squared_error(test_df[target_col], predicted_labels)
#     rmse = np.sqrt(mse)
#     return rmse