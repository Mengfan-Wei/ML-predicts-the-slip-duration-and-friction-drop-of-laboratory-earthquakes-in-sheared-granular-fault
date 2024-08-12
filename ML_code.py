# %%
#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd

import warnings
import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold,train_test_split
from sklearn.feature_selection import SelectFromModel
import shap
from hyperopt import fmin, tpe, hp
import multiprocessing

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc

from scipy.stats import skew, kurtosis
from scipy.signal import wiener

warnings.filterwarnings("ignore")
# %%            #! Load data  
DxAveTop = pd.read_csv('DxAveTop.txt')
DyAveTop = pd.read_csv('DyAveTop.txt')
VxAveTop = pd.read_csv('VxAveTop.txt')
VyAveTop = pd.read_csv('VyAveTop.txt')
events = np.load('Slip_events.npy')

# %%            #! Create label 
duration_slip = events[1]- events[1::2,0]
magnitude_slip = events[2]

# %%            #! Create features
def stat_fpenc(fpenc, features, events, name):
    A = events.shape[0]
    X = pd.DataFrame()

    for i in range(A):
        event = events[i]
        time = int((event[1] - event[0]) * fpenc)
        feature = features[int(event[0]):int(event[0])+time+1]
        X.loc[i, name + '_mean'.format(fpenc*100)] = np.mean(feature)
        X.loc[i, name + '_var'.format(fpenc*100)] = np.var(feature)
        X.loc[i, name + '_skew'.format(fpenc*100)] = skew(feature)
        X.loc[i, name + '_kurt'.format(fpenc*100)] = kurtosis(feature)
        X.loc[i, name + '_median'.format(fpenc*100)] = np.median(feature)
        X.loc[i, name + '_max'.format(fpenc*100)] = np.max(feature)
        X.loc[i, name + '_91'.format(fpenc*100)] = np.quantile(feature, 0.90)-np.quantile(feature, 0.10)
        X.loc[i, name + '_iqr'.format(fpenc*100)] = np.quantile(feature, 0.75)-np.quantile(feature, 0.25)
            
    return X 

# %%            #! Train and test the model under 10 time Windows
def return_r2(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, shuffle=False)
    space = {
        'learning_rate': hp.loguniform("learning_rate", np.log(0.005), np.log(0.1)),
        'max_depth': hp.choice('max_depth', [2, 3, 4, 5]),
        'num_leaves': hp.quniform('num_leaves', 2, 20, 1),
        'min_child_samples': hp.quniform('min_child_samples', 1, 15, 1),  
        'lambda_l1':  hp.uniform('lambda_l1', 0, 2),    
        'lambda_l2': hp.uniform('lambda_l2', 0, 20),
    }

    callbacks = [early_stopping(stopping_rounds=300, verbose=False)]
    def lgb_cv(params): 
        predictions = np.zeros(X_test.shape[0])
        flods = KFold(n_splits=3, shuffle=False)
        for _, (train_idx, val_idx) in enumerate(flods.split(X_train, y_train)):
            train_data = lgb.Dataset(X_train.iloc[train_idx], label=y_train[train_idx])
            val_data = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx])
            param = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'learning_rate': params["learning_rate"],
                'max_depth': int(params["max_depth"]),
                'num_leaves': int(params["num_leaves"]),
                'min_child_samples': int(params["min_child_samples"]), 
                'max_bin':  255,
                'lambda_l1': params['lambda_l1'],
                'lambda_l2':  params['lambda_l2'],
                'subsample':0.9, 
                'colsample_bytree':0.9,
                'metric': 'rmse',
                'seed': 42,
                'verbosity': -1,
                'num_threads': 40
            }
            reg = lgb.train(params=param, train_set=train_data, num_boost_round=5000, valid_sets=[
                train_data, val_data], valid_names=['Training', 'Validation'], 
                callbacks=callbacks)
            predictions += reg.predict(X_test, num_iteration=reg.best_iteration) / flods.n_splits

        return 1 - r2_score(predictions, y_test)

    params_final = fmin(fn=lgb_cv, space=space, algo=tpe.suggest, max_evals=2000, rstate=np.random.default_rng(42))
    print(params_final)     
    lgb_train = lgb.Dataset(X_train, y_train, feature_name=list(X_train.columns))
    lgb_val = lgb.Dataset(X_test, y_test, feature_name=list(X_train.columns)) 

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'learning_rate': params_final['learning_rate'],
        'max_depth': int(params_final['max_depth']),
        'num_leaves': int(params_final['num_leaves']), 
        'min_child_samples': int(params_final["min_child_samples"]), 
        'max_bin':  255,
        'lambda_l1': params_final['lambda_l1'],
        'lambda_l2':  params_final['lambda_l2'],
        'subsample':0.9,
        'colsample_bytree':0.9,
        'metric': 'rmse',
        'seed': 42,
        'verbosity': -1,
        'num_threads': 40
    }

    lgb_model = lgb.train(params, lgb_train, num_boost_round=5000, valid_sets=[
                lgb_train, lgb_val], valid_names=('Train', 'Validation'),
                callbacks=callbacks)
    
    y_pred_vd = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    r2 = r2_score(y_test, y_pred_vd)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_vd))
    return r2, rmse

r2_array, rmse_array = [], []

def fpenc_test(fpenc):
    fpenc = fpenc * 0.1
    Dx_stat = stat_fpenc(DxAveTop, events, 'Dx')
    Dy_stat = stat_fpenc(DyAveTop, events, 'Dy')
    Vx_stat = stat_fpenc(VxAveTop, events, 'Vx')
    Vy_stat = stat_fpenc(VyAveTop, events, 'Vy')
    ffeatures_stat_magn = pd.concat([Dx_stat, Dy_stat, Vx_stat, Vy_stat], axis=1)
    # ffeatures_stat_dura = pd.concat([Dx_stat, Dy_stat, Vx_stat, Vy_stat], axis=1)

    return return_r2(ffeatures_stat_magn, magnitude_slip)
    # return return_r2(ffeatures_stat_dura, duration_slip)

p = multiprocessing.Pool(10)        
result = p.map_async(fpenc_test, range(1, 11))
for r2, rmse in result.get():
    r2_array.append(r2)
    rmse_array.append(rmse)
p.close()
p.join()

np.savetxt('r2_slip_slip_magn.csv', r2_array)
np.savetxt('rmse_slip_slip_magn.csv', rmse_array)
# np.savetxt('r2_slip_slip_dura.csv', r2_array)
# np.savetxt('rmse_slip_slip_dura.csv', rmse_array)

# %%            #! Select the two models with the best predictions for slip duration and friction drop
def test_r2(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.4, shuffle=False)
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_test, y_test)

    boost_round = 5000
    early_stop = 300

    #* dura_params
    # params = {
    #      'boosting_type': 'gbdt',
    #     'objective': 'regression',
    #     'learning_rate': 0.005,
    #     'max_depth': 2,
    #     'num_leaves': 3,
    #     'min_child_samples':8,
    #     'max_bin': 128,  
    #     'lambda_l1': 1.712941,
    #     'lambda_l2':  14.6973555035,
    #     'subsample':0.9, 
    #     'colsample_bytree':0.9,
    #     'metric': 'rmse',
    #     'seed': 42,
    #     'verbosity': -1,        
    # }

    #* magn_params
    params =  {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'learning_rate': 0.005,
        'max_depth': 3,
        'num_leaves': 8,
        'min_child_samples':4,
        'max_bin': 255,  
        'lambda_l1': 0.0002,
        'lambda_l2':  0.5273555035,
        'subsample':0.9, 
        'colsample_bytree':0.9,
        'metric': 'rmse',
        'seed': 42,
        'verbosity': -1,        
    }

    lgb_model = lgb.train(params, lgb_train, num_boost_round=boost_round,
        valid_sets=[lgb_train, lgb_val], valid_names=('Train', 'Validation'), 
        early_stopping_rounds=early_stop, verbose_eval=100)

    y_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    print('The R2 = ', r2_score(y_test, y_pred))
    return lgb_model, y_pred

# %%
fpenc_dura = 0.3
Dx_stat_dura = stat_fpenc(fpenc_dura, DxAveTop, events, 'Dx')
Dy_stat_dura = stat_fpenc(fpenc_dura, DyAveTop, events, 'Dy')
Vx_stat_dura = stat_fpenc(fpenc_dura, VxAveTop, events, 'Vx')
Vy_stat_dura = stat_fpenc(fpenc_dura, VyAveTop, events, 'Vy')

fpenc_magn = 1.0
Dx_stat_magn = stat_fpenc(fpenc_magn, DxAveTop, events, 'Dx')
Dy_stat_magn = stat_fpenc(fpenc_magn, DyAveTop, events, 'Dy')
Vx_stat_magn = stat_fpenc(fpenc_magn, VxAveTop, events, 'Vx')
Vy_stat_magn = stat_fpenc(fpenc_magn, VyAveTop, events, 'Vy')

ffeatures_stat_magn = pd.concat([Dx_stat_magn, Dy_stat_magn, Vx_stat_magn, Vy_stat_magn], axis=1)
ffeatures_stat_dura = pd.concat([Dx_stat_dura, Dy_stat_dura, Vx_stat_dura, Vy_stat_dura], axis=1)

lgb_model_dura, dura_pred= test_r2(ffeatures_stat_dura, duration_slip)
lgb_model_magn, magn_pred= test_r2(ffeatures_stat_magn, magnitude_slip)

# %%            #! SHAP analysis
shap_feature_magn = ffeatures_stat_magn
explainer_magn = shap.TreeExplainer(lgb_model_magn)
shap_values_magn = explainer_magn.shap_values(shap_feature_magn)

shap_feature_dura = ffeatures_stat_dura
explainer_dura = shap.TreeExplainer(lgb_model_dura)
shap_values_dura = explainer_dura.shap_values(shap_feature_dura)

# %%
#! global importance
shap.summary_plot(shap_values_magn, shap_feature_magn, plot_type="bar", max_display=10, show=False)
plt.xlabel('mean(|SHAP value|)')
plt.show()

shap.summary_plot(shap_values_dura, shap_feature_dura, plot_type="bar", max_display=10, show=False)
plt.xlabel('mean(|SHAP value|)')
plt.show()

# %%
#! Summary plot
shap.summary_plot(shap_values_magn, shap_feature_magn, max_display=10,show=False, plot_type='violin')
plt.xlabel('SHAP value')
plt.show()

shap.summary_plot(shap_values_dura, shap_feature_dura, max_display=10,show=False, plot_type='violin')
plt.xlabel('SHAP value')
plt.show()


