import numpy as np
from sklearn.metrics import mean_squared_error


def regression_process_with_given_method_train_test(train_col, target_col, df, regressor, print_predicted = False):
    np.random.seed(1)

    # Randomize order of rows in data frame
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round
    last_train_row = int(len(rand_df) / 2)

    # Select the first half and set as training set
    # Select the second half and set as test set
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]

    # Fit model
    regressor.fit(train_df[[train_col]], train_df[target_col])

    # Make predictions
    predicted_labels = regressor.predict(test_df[[train_col]])

    if print_predicted:
        print(predicted_labels)

    # Calculate RMSE
    mse = mean_squared_error(test_df[target_col], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse
