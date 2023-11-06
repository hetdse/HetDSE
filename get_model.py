from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge

from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern


from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import  *

def get_surrogate_model(surrogate_model_tag):
    base_estimator = None
    base_estimator2 = None
    surrogate_model_tag_real = surrogate_model_tag
    surrogate_model_dict = {}

    surrogate_model_dict['model_is_iterative'] = False
    surrogate_model_dict['semi_train'] = False
    surrogate_model_dict['semi_train_adapt'] = False
    surrogate_model_dict['kernel_train'] = False
    surrogate_model_dict['different_model'] = False
    surrogate_model_dict['acq_func'] = 'HVI'
    #surrogate_model_dict['acq_func'] = "acq_hmp_aware"
    surrogate_model_dict['cv_ranking'] = 'no'
    surrogate_model_dict['cv_pool_size'] = 100
    surrogate_model_dict['ucb'] = 0.0
    surrogate_model_dict['ucb_v'] = 0
    surrogate_model_dict['warm_start'] = False
    surrogate_model_dict['labeled_predict_mode'] = False
    surrogate_model_dict['semi_train_iter_max'] = 10
    surrogate_model_dict['predict_last'] = True
    surrogate_model_dict['cv_ranking_beta'] = 0
    surrogate_model_dict['cv_ranking_beta_v'] = 0
    surrogate_model_dict['non_uniformity_explore'] = 0
    surrogate_model_dict['filter'] = 0
    surrogate_model_dict['filter_min'] = 0.2
    surrogate_model_dict['filter_max'] = 1.5  
    surrogate_model_dict['sample_weight'] = 0
    surrogate_model_dict['hv_scale_v'] = 0

    surrogate_model_dict['all_workloads_per_iter_mode'] = False
    surrogate_model_dict['acq_hmp_aware'] = 0
    surrogate_model_dict['est_each_workload'] = 0

    #surrogate_model_dict['multi_est'] = True
    surrogate_model_dict['multi_est'] = False

    # 'sobol', 'lhs', 'halton', 'hammersly','random', or 'grid'
    surrogate_model_dict['initial_point_generator'] = "random"
    # surrogate_model_dict['initial_point_generator'] = "orthogonal"

    if "smoke_test" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor
        hidden_layer_1 = 2
        dt_stump = MLPRegressor(hidden_layer_sizes=(hidden_layer_1),
                                max_iter=2,
                                solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                activation='relu',
                                )
        # dt_stump.fit(train_X, train_Y)
        base_estimator = MLPRegressor(hidden_layer_sizes=(hidden_layer_1),
                                max_iter=2,
                                solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                activation='relu',
                                )
        base_estimator2 = MLPRegressor(hidden_layer_sizes=(hidden_layer_1),
                                max_iter=2,
                                solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                activation='relu',
                                )
        #surrogate_model_dict['semi_train'] = 1
        surrogate_model_dict['model_is_iterative'] = False
    elif "smoke_test2" == surrogate_model_tag:
        kernel = Matern()
        if False:
            for hyperparameter in kernel.hyperparameters:
                print(hyperparameter)
            params = kernel.get_params()
            for key in sorted(params): print(f"{key} : {params[key]}")
        # noise_level = 0.0  0.00958
        base_estimator = Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression())
        ])
        base_estimator2 = base_estimator
        surrogate_model_dict['model_is_iterative'] = True
        surrogate_model_dict['acq_hmp_aware'] = 0
        #surrogate_model_dict['acq_func'] = 'acq_hmp_aware'
    elif "PolyLinear" == surrogate_model_tag:
        base_estimator = Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression())
        ])
        base_estimator2 = Pipeline([
            ("poly", PolynomialFeatures(degree=3)),
            ("std_scaler", StandardScaler()),
            ("lin_reg", LinearRegression())
        ])        
        surrogate_model_dict['model_is_iterative'] = True
        surrogate_model_dict['acq_func'] = 'ei'
    elif "Ridge" == surrogate_model_tag:
        base_estimator = KernelRidge(kernel="rbf")
        base_estimator2 = KernelRidge(kernel="rbf")
        surrogate_model_dict['model_is_iterative'] = False
    elif "LGBMQuantileRegressor" == surrogate_model_tag:
        from LGBMQuantileRegressor import LGBMQuantileRegressor

        base_estimator = LGBMQuantileRegressor()
        # base_estimator.fit(train_X, train_Y)
        # predict_value_R2 = base_estimator.score(train_X, train_Y)
        # print(f"R2={predict_value_R2}")
        # predict_value = base_estimator.predict(train_X)
        # print(f"MSE={mean_squared_error(predict_value, train_Y)}")
    elif "SVR_Matern" == surrogate_model_tag:
        kernel = 1.0 * Matern(length_scale=3.0, length_scale_bounds=(1e-1, 20.0), nu=2.5)
        kernel2 = 1.0 * Matern(length_scale=3.0, length_scale_bounds=(1e-1, 20.0), nu=2.5)
        base_estimator = SVR(kernel=kernel, degree=8, )
        base_estimator2 = SVR(kernel=kernel2, degree=1, )
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_dict['different_model'] = True
    elif "MLP" == surrogate_model_tag:
        HBO_params_cpi = {'learning_rate_init': 0.01, 'activation': 'logistic', 'solver': 'lbfgs'} 
        HBO_params_power = {'learning_rate_init': 0.02, 'activation': 'logistic', 'solver': 'lbfgs'} 
        base_estimator = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                      solver='lbfgs',  # ['adam', 'sgd', 'lbfgs'],
                                      activation='relu',
                                      max_iter=10000,
                                      )
        base_estimator = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                      **HBO_params_cpi,
                                      max_iter=10000,
                                      )
        #base_estimator2 = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
        #                               solver='lbfgs',  # ['adam', 'sgd', 'lbfgs'],
        #                               activation='relu',
        #                               max_iter=10000,
        #                               # verbose=True,
        #                               )
        base_estimator2 = MLPRegressor(hidden_layer_sizes=(16, 32, 32),
                                      **HBO_params_power,
                                      max_iter=10000,
                                      )            
        # print(f"base_estimator={base_estimator}")
        surrogate_model_dict['model_is_iterative'] = False
    elif "ASPLOS06" == surrogate_model_tag:
        base_estimator = MLPRegressor(hidden_layer_sizes=(16),
                                      solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                      activation='relu',
                                      max_iter=10000,
                                      )
        surrogate_model_dict['model_is_iterative'] = False
        #surrogate_model_tag_real += "_pred"
        # surrogate_model_dict['initial_point_generator'] = "hammersly"
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
    elif "GP" == surrogate_model_tag:
        '''
        1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0)),
        1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1),
        1.0 * ExpSineSquared(length_scale=1.0, periodicity=3.0,
                            length_scale_bounds=(0.1, 10.0),
                            periodicity_bounds=(1.0, 10.0)),
        ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2),
        '''            
        #kernel = 1.0 * Matern()
        kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0), nu=2.5)
        #kernel2 = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 20.0), nu=1.5)
        #noise_level = 0.00958
        HBO_params_cpi = {'n_restarts_optimizer': 2, 'kernel__k2__nu': 2.5, 'normalize_y': False} 
        HBO_params_power = {'n_restarts_optimizer': 9, 'kernel__k2__nu': 1.5, 'normalize_y': False} 
        base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                  #alpha=noise_level ** 2,
                                                  normalize_y=False,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=2
                                                  )
        base_estimator2 = GaussianProcessRegressor(kernel=kernel,
                                                  #alpha=noise_level ** 2,
                                                  normalize_y=False,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=2, #9
                                                  )
        surrogate_model_tag_real += "_Matern"
        # surrogate_model_dict['acq_func'] = 'EHVI'
        surrogate_model_dict['model_is_iterative'] = True
        surrogate_model_dict['acq_func'] = 'ei'
        if 1:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
    elif "BOOM-Explorer" == surrogate_model_tag:
        #kernel = Sklearn_DKL_GP(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        if False:
            for hyperparameter in kernel.hyperparameters:
                print(hyperparameter)
            params = kernel.get_params()
            for key in sorted(params): print(f"{key} : {params[key]}")
        base_estimator = GaussianProcessRegressor(kernel=kernel,
                                                  # alpha=noise_level ** 2,
                                                  #alpha=0.00958,
                                                  normalize_y=True,
                                                  # noise="gaussian",
                                                  n_restarts_optimizer=8
                                                  )
        #kernel2 = Sklearn_DKL_GP(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        kernel2 = Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=2.5)
        base_estimator2 = GaussianProcessRegressor(kernel=kernel2,
                                                   # alpha=noise_level ** 2,
                                                   #alpha=0.00958,
                                                   normalize_y=False,
                                                   # noise="gaussian",
                                                   n_restarts_optimizer=6
                                                   )
        #surrogate_model_tag_real += "_DKL_GP"
        #surrogate_model_dict['kernel_train'] = True
        # surrogate_model_dict['acq_func'] = 'EHVI'
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        #surrogate_model_dict['initial_point_generator'] = "kmeans"
        #surrogate_model_dict['initial_point_generator'] = "distance"
        surrogate_model_dict['model_is_iterative'] = True
        if 0:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
        if 0:
            surrogate_model_dict['filter'] = 2
            #surrogate_model_tag_real += '_fifix-5' should modify code
        surrogate_model_tag_real += '_v6'
    elif "RF_custom" == surrogate_model_tag:
        HBO_params_cpi = {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1} 
        HBO_params_power = {'n_estimators': 50, 'max_depth': 10, 'min_samples_leaf': 1} 
        base_estimator = RandomForestRegressor(**HBO_params_cpi)
        base_estimator2 = RandomForestRegressor(**HBO_params_power)
        surrogate_model_dict['model_is_iterative'] = True
        if 1:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
        surrogate_model_dict['acq_func'] = 'ei'            
    elif "RF" == surrogate_model_tag:
        base_estimator = RandomForestRegressor()
        base_estimator2 = RandomForestRegressor()
        surrogate_model_dict['model_is_iterative'] = True
        if 1:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
        #surrogate_model_dict['cv_ranking'] = 'maxsort'
        surrogate_model_dict['acq_func'] = 'ei'
        #surrogate_model_dict['sample_weight'] = 6
        #surrogate_model_dict['multi_est'] = True
    elif "ET_custom" == surrogate_model_tag:
        HBO_params_cpi = {'n_estimators': 199, 'max_depth': 24, 'min_samples_leaf': 1, 'max_features': 'auto'}
        HBO_params_power = {'n_estimators': 197, 'max_depth': 13, 'min_samples_leaf': 1, 'max_features': 'auto'}
        base_estimator = ExtraTreesRegressor(**HBO_params_cpi,n_jobs=2)
        base_estimator2 = ExtraTreesRegressor(**HBO_params_power, n_jobs=2)
        surrogate_model_dict['model_is_iterative'] = False
    elif "ET" == surrogate_model_tag:
        base_estimator = "ET"
    elif "AdaBoost_DTR" == surrogate_model_tag:
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import AdaBoostRegressor

        #dt_stump = DecisionTreeRegressor(max_depth=8)
        # dt_stump.fit(train_X, train_Y)
        HBO_params_cpi = {'n_estimators': 168, 'learning_rate': 0.001, 'base_estimator': DecisionTreeRegressor(max_depth=8)} 
        HBO_params_power = {'n_estimators': 85, 'learning_rate': 0.001, 'base_estimator': DecisionTreeRegressor(max_depth=8)} 
        base_estimator = AdaBoostRegressor(
            #base_estimator=dt_stump,
            **HBO_params_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            #base_estimator=dt_stump,
            **HBO_params_power,
        )            
        surrogate_model_dict['model_is_iterative'] = False
    elif "AdaBoost_MLP" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor

        hidden_layer_1 = 16
        hidden_size = (hidden_layer_1, hidden_layer_1 * 2, hidden_layer_1 * 2)
        dt_stump = MLPRegressor(hidden_layer_sizes=hidden_size, max_iter=10000, )
        # dt_stump.fit(train_X, train_Y)
        HBO_params_cpi = {'n_estimators': 100, 'learning_rate': 0.005} 
        HBO_params_power = {'n_estimators': 100, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            **HBO_params_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=dt_stump,
            **HBO_params_power,
        )
        #surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = False
    elif "ActBoost" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor
        hidden_size = (8, 6)
        dt_stump = MLPRegressor(hidden_layer_sizes=hidden_size, max_iter=10000, )
        # dt_stump.fit(train_X, train_Y)
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            learning_rate=0.001,
            n_estimators=2 * 10,
        )
        surrogate_model_dict['cv_ranking'] = 'maxsort'
        surrogate_model_dict['acq_func'] = 'cv_ranking'
        surrogate_model_tag_real += '_v4'
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = True
    elif "SemiBoost" == surrogate_model_tag:
        from sklearn.ensemble import AdaBoostRegressor
        hidden_layer_1 = 8
        surrogate_model_dict['warm_start'] = False
        dt_stump = MLPRegressor(hidden_layer_sizes=(hidden_layer_1, hidden_layer_1),
                                max_iter=10000,
                                solver='sgd',  # ['adam', 'sgd', 'lbfgs'],
                                activation='relu',
                                warm_start=surrogate_model_dict['warm_start'],
                                )
        # dt_stump.fit(train_X, train_Y)
        base_estimator = AdaBoostRegressor(
            base_estimator=dt_stump,
            learning_rate=0.001,
            n_estimators=20,
        )
        surrogate_model_dict['cv_ranking'] = 'minsort'
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_dict['semi_train'] = 1
        surrogate_model_dict['semi_train_iter_max'] = 50
        surrogate_model_dict['initial_point_generator'] = "lhs"
    elif "M5P" == surrogate_model_tag:
        #surrogate_model_tag_real += '_v2'
        from model_tree.src.ModelTree import ModelTree
        from model_tree.models.linear_regr import linear_regr
        model = linear_regr()
        base_estimator = ModelTree(model, max_depth=4, min_samples_leaf=10, search_type="greedy", n_search_grid=100)
        base_estimator2 = base_estimator
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = False            
    elif "GBRT-base" == surrogate_model_tag:
        # base_estimator = "GBRT"
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        # base_estimator = GradientBoostingRegressor(loss="squared_error",
        #                                                       n_estimators=73,
        #                                                       learning_rate=0.1,
        #                                                       max_depth=19,
        #                                                       subsample=0.5,
        #                                                       )
        base_estimator = GradientBoostingRegressor(**HBO_params_cpi)
        base_estimator2 = GradientBoostingRegressor(**HBO_params_power)
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_tag_real += '_v2'
    elif "GBRT-orh" == surrogate_model_tag:
        surrogate_model_tag_real += '_MART'
        # base_estimator = "GBRT"
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        # base_estimator = GradientBoostingRegressor(loss="squared_error",
        #                                                       n_estimators=73,
        #                                                       learning_rate=0.1,
        #                                                       max_depth=19,
        #                                                       subsample=0.5,
        #                                                       )
        base_estimator = GradientBoostingRegressor(**HBO_params_cpi)
        base_estimator2 = GradientBoostingRegressor(**HBO_params_power)
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_tag_real += '_v2'
    elif "GBRT" == surrogate_model_tag:
        if 36864 == N_SPACE_SIZE:
            HBO_params_cpi = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            #HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}            
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        # base_estimator = GradientBoostingRegressor(loss="squared_error",
        #                                                       n_estimators=73,
        #                                                       learning_rate=0.1,
        #                                                       max_depth=19,
        #                                                       subsample=0.5,
        #                                                       )
        base_estimator = GradientBoostingRegressor(**HBO_params_cpi)
        base_estimator2 = GradientBoostingRegressor(**HBO_params_power)
        surrogate_model_dict['different_model'] = True
        #surrogate_model_dict['ucb'] = 0.001
        # surrogate_model_dict['ucb_scale'] = 1.01
        #surrogate_model_dict['semi_train'] = 1
        #surrogate_model_dict['semi_train_iter_max'] = 30
        #surrogate_model_dict['semi_train_adapt'] = True
        if surrogate_model_dict['semi_train']:
            surrogate_model_dict['cv_ranking'] = 'minsort'
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['cv_ranking_beta'] = 0.1
        #surrogate_model_dict['acq_func'] = 'EHVI'
        surrogate_model_dict['model_is_iterative'] = True
        surrogate_model_dict['acq_func'] = 'ei'
        #surrogate_model_tag_real += '_v2'
    elif "XGBoost" == surrogate_model_tag:
        from xgboost import XGBRegressor
        #HBO_params_cpi = {'n_estimators': 70, 'learning_rate': 0.2, 'max_depth': 48, 'booster': 'gbtree', 'subsample': 0.8}
        HBO_params_cpi = {'n_estimators': 174, 'learning_rate': 0.15, 'max_depth': 5, 'booster': 'dart', 'subsample': 0.8} 
        # HBO_params_power = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 23, 'objective': 'reg:squarederror', 'booster': 'gbtree', 'subsample': 0.5}
        #HBO_params_power = {'n_estimators': 97, 'learning_rate': 0.1, 'max_depth': 30, 'objective': 'reg:squarederror', 'booster': 'gbtree', 'subsample': 0.5}
        HBO_params_power = {'n_estimators': 132, 'learning_rate': 0.2, 'max_depth': 4, 'booster': 'dart', 'subsample': 0.7} 
        base_estimator = XGBRegressor(
            # max_depth=25,
            # learning_rate=0.1,
            # n_estimators=90,
            # objective='reg:squarederror',
            # booster='gbtree',
            **HBO_params_cpi,
            n_jobs=2,
            nthread=None,
        )
        base_estimator2 = XGBRegressor(
            **HBO_params_power,
            n_jobs=2,
            nthread=None,
        )
        surrogate_model_tag_real += '_v3'
        #surrogate_model_dict['initial_point_generator'] = "orthogonal"
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = False
    elif "LGBMRegressor" == surrogate_model_tag:
        from lightgbm import LGBMRegressor
        base_estimator = LGBMRegressor(
            n_estimators=71,
            learning_rate=1.0,
            n_jobs=2,
            nthread=None,
        )
        # num_leaves = 31
    elif "CatBoostRegressor" == surrogate_model_tag:
        from catboost import CatBoostRegressor
        HBO_params_cpi = {'n_estimators': 155, 'depth': 10, 'subsample': 1.0, 'early_stopping_rounds': 144,
                          'grow_policy': 'Lossguide'}
        HBO_params_power = {'n_estimators': 200, 'depth': 10, 'subsample': 0.7, 'early_stopping_rounds': 37,
                            'grow_policy': 'Lossguide'}
        base_estimator = CatBoostRegressor(
            # n_estimators=90,
            # depth=6,
            # subsample=0.8,
            # early_stopping_rounds=164,
            **HBO_params_cpi,
            verbose=False,
            thread_count=-1
        )
        base_estimator2 = CatBoostRegressor(
            # n_estimators=198,
            # depth=6,
            # subsample=0.7,
            # early_stopping_rounds=50,
            **HBO_params_power,
            verbose=False,
            thread_count=-1
        )
        surrogate_model_dict['different_model'] = True
    elif "AdaGBRT-no-iter" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = False
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
    elif "AdaGBRT-base" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        surrogate_model_dict['different_model'] = True
    elif "AdaGBRT" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        surrogate_model_dict['initial_point_generator'] = "orthogonal"
        #surrogate_model_dict['initial_point_generator'] = "kmeans"
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['sample_weight'] = 6
        surrogate_model_dict['model_is_iterative'] = True        
    elif "BagGBRT" == surrogate_model_tag:
        if 36864 <= N_SPACE_SIZE:
            #HBO_params_cpi = {'n_estimators': 139, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            #HBO_params_power = {'n_estimators': 199, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}
            HBO_params_cpi = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 0.6}
            HBO_params_power = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.6}                         
        else:
            HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
            HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        HBO_params_ada_power = {'n_estimators': 20, 'n_jobs': 1}
        HBO_params_ada_cpi = {'n_estimators': 20}
        from sklearn.ensemble import BaggingRegressor
        base_estimator = BaggingRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = BaggingRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        #surrogate_model_dict['initial_point_generator'] = "orthogonal"
        #surrogate_model_dict['initial_point_generator'] = "kmeans"
        surrogate_model_dict['different_model'] = True
        surrogate_model_dict['model_is_iterative'] = True
        surrogate_model_dict['sample_weight'] = 6
        if 1:
            surrogate_model_dict['ucb'] = 0.05
            surrogate_model_dict['ucb_v'] = 12
        #surrogate_model_dict['cv_ranking'] = 'maxsort'
        surrogate_model_dict['acq_func'] = 'ei'
        if 0:
            surrogate_model_dict['filter'] = 2
        surrogate_model_tag_real += '_v3'
    elif "AdaGBRT-cv" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        # 'sobol', 'lhs', 'halton', 'hammersly','random', or 'grid', "orthogonal"
        surrogate_model_dict['initial_point_generator'] = 'orthogonal'
        #surrogate_model_dict['initial_point_generator'] = "kmeans"        
        #surrogate_model_tag_real += '_v2'
        surrogate_model_dict['different_model'] = True
        #surrogate_model_dict['ucb'] = 0.01
        #surrogate_model_dict['ucb_v'] = 27
        # surrogate_model_dict['ucb_scale'] = 1.01
        #surrogate_model_dict['semi_train'] = 1
        surrogate_model_dict['semi_train_iter_max'] = 30
        #surrogate_model_dict['semi_train_adapt'] = True
        if surrogate_model_dict['semi_train']:
            surrogate_model_dict['cv_ranking'] = 'minsort'

        #surrogate_model_dict['cv_ranking_beta'] = 0.1
        surrogate_model_dict['cv_ranking_beta_v'] = 12
        surrogate_model_dict['cv_pool_size'] = 40
        #surrogate_model_dict['acq_func'] = 'EHVI'

        surrogate_model_dict['non_uniformity_explore'] = 16
        #surrogate_model_dict['hv_scale_v'] = 1
        surrogate_model_dict['sample_weight'] = 6
    elif "AdaGBRT-fi" == surrogate_model_tag:
        HBO_params_cpi = {'loss': 'squared_error', 'n_estimators': 98, 'learning_rate': 0.1, 'max_depth': 6,
                          'subsample': 0.8}
        HBO_params_power = {'loss': 'squared_error', 'n_estimators': 99, 'learning_rate': 0.1, 'max_depth': 4,
                            'subsample': 0.5}
        from sklearn.ensemble import AdaBoostRegressor
        HBO_params_ada_cpi = {'n_estimators': 40, 'learning_rate': 0.005}
        HBO_params_ada_power = {'n_estimators': 26, 'learning_rate': 0.01}
        base_estimator = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_cpi),
            **HBO_params_ada_cpi,
        )
        base_estimator2 = AdaBoostRegressor(
            base_estimator=GradientBoostingRegressor(**HBO_params_power),
            **HBO_params_ada_power,
        )
        # 'sobol', 'lhs', 'halton', 'hammersly','random', or 'grid', "orthogonal"
        surrogate_model_dict['initial_point_generator'] = 'orthogonal'
        #surrogate_model_dict['initial_point_generator'] = "kmeans"        
        #surrogate_model_tag_real += '_v2'
        surrogate_model_dict['different_model'] = True
        #surrogate_model_dict['ucb'] = 0.01
        #surrogate_model_dict['ucb_v'] = 27
        # surrogate_model_dict['ucb_scale'] = 1.01
        #surrogate_model_dict['semi_train'] = 1
        surrogate_model_dict['semi_train_iter_max'] = 30
        #surrogate_model_dict['semi_train_adapt'] = True
        if surrogate_model_dict['semi_train']:
            surrogate_model_dict['cv_ranking'] = 'minsort'

        #surrogate_model_dict['cv_ranking_beta'] = 0.1
        surrogate_model_dict['cv_ranking_beta_v'] = 12
        surrogate_model_dict['cv_pool_size'] = 40
        #surrogate_model_dict['acq_func'] = 'EHVI'

        surrogate_model_dict['non_uniformity_explore'] = 16
        #surrogate_model_dict['hv_scale_v'] = 1
        surrogate_model_dict['sample_weight'] = 6
        surrogate_model_dict['filter'] = 2
        surrogate_model_dict['filter_min'] = 0.5
        surrogate_model_dict['filter_max'] = 2      
    else:
        print(f"no def surrogate_model_tag={surrogate_model_tag}")
        exit(1)

    if surrogate_model_dict['semi_train']:
        surrogate_model_tag_real += '_semiv6-' + str(surrogate_model_dict['semi_train'])
        surrogate_model_tag_real += "_w" + str(surrogate_model_dict['semi_train_iter_max'])
        if surrogate_model_dict['semi_train_adapt']:
            surrogate_model_tag_real += '_adapt'

    if surrogate_model_dict['model_is_iterative'] is False:
        surrogate_model_tag_real += "_no_iter"
    if base_estimator2 is not None:
        surrogate_model_dict['different_model'] = True
    if surrogate_model_dict['multi_est']:
        surrogate_model_tag_real += "_se"
    if surrogate_model_dict['different_model']:
        surrogate_model_tag_real += "_diff_model"
    if 'HVI' != surrogate_model_dict['acq_func']:
        surrogate_model_tag_real += '_' + surrogate_model_dict['acq_func']
        if 'ei' == surrogate_model_dict['acq_func']:
            surrogate_model_tag_real += '2'
    if 'no' != surrogate_model_dict['cv_ranking']:
        surrogate_model_tag_real += "_cv" + surrogate_model_dict['cv_ranking']
    if surrogate_model_dict['cv_ranking_beta']:
        surrogate_model_tag_real += "_cvbetav" + str(surrogate_model_dict['cv_ranking_beta_v']) + "-" + str(surrogate_model_dict['cv_ranking_beta'])
    if surrogate_model_dict['ucb_v']:
        # surrogate_model_tag_real += "_ucb-" + str(surrogate_model_dict['ucb']) + '-' + str(surrogate_model_dict['ucb_scale'])
        surrogate_model_tag_real += "_ucb3v" + str(surrogate_model_dict['ucb_v']) + "-" + str(surrogate_model_dict['ucb'])
    if surrogate_model_dict['non_uniformity_explore']:
        surrogate_model_tag_real += "_univ" + str(surrogate_model_dict['non_uniformity_explore'])
        #surrogate_model_tag_real += "_univMAX" + str(surrogate_model_dict['non_uniformity_explore'])
    if surrogate_model_dict['warm_start']:
        surrogate_model_tag_real += "_warm"
    if surrogate_model_dict['filter']:
        surrogate_model_tag_real += "_fivg" + str(surrogate_model_dict['filter']) + 'm' + str(surrogate_model_dict['filter_min']) + '-' + str(surrogate_model_dict['filter_max'])   
    if "random" != surrogate_model_dict['initial_point_generator']:
        surrogate_model_tag_real += "_" + surrogate_model_dict['initial_point_generator']        
    #if surrogate_model_dict['labeled_predict_mode'] is False:
    if surrogate_model_dict['hv_scale_v']:
        surrogate_model_tag_real += "_v5"
    else:
        surrogate_model_tag_real += "_v4"
    if surrogate_model_dict['sample_weight']:
        surrogate_model_tag_real += "_sw" + str(surrogate_model_dict['sample_weight'])
    #surrogate_model_tag_real += "_n" + str(N_SAMPLES_ALL)

    surrogate_model_dict['tag'] = surrogate_model_tag_real
    return base_estimator, base_estimator2, surrogate_model_dict