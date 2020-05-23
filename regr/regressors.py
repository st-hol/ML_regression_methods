from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNet
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import BaggingRegressor, StackingRegressor, RandomForestRegressor, VotingRegressor, \
    AdaBoostRegressor

estimators = [
    ('lr', RidgeCV()),
    ('svr', LinearSVR(random_state=42))
]

r1 = LinearRegression()
r2 = RandomForestRegressor(n_estimators=10, random_state=1)

REGRESSORS = [KNeighborsRegressor(),
              LinearRegression(),
              RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1]),
              Lasso(alpha=0.1),
              ElasticNet(),
              BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0),
              StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10,
                                                                                             random_state=42)),

              VotingRegressor([('lr', r1), ('rf', r2)]),
              AdaBoostRegressor(random_state=0, n_estimators=100)]
