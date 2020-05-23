import regr.regression_processor as rm

def calc_resRMSE_for_hp_and_width(normalized_cars, regressor, printPredicted):
    print("--PREDICTING WIDTH--")
    width = rm.regression_process_with_given_method_train_test('width', 'price', normalized_cars, regressor, printPredicted)
    print("--PREDICTING HORSEPOWER--")
    horsepower = rm.regression_process_with_given_method_train_test('horsepower', 'price', normalized_cars, regressor, printPredicted)
    print('WIDTH RMSE RESULT -> ', width)
    print('HORSEPOWER RMSE RESULT -> ',horsepower)