import os

import pandas as pd
from arch import arch_model
import numpy as np

from scipy.stats import uniform as sp_rand
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV


def calc_garch(spt, vol='EGARCH'):
    """
    计算年化garch
    """
    start_dt = spt.index[1]
    ret = (spt['s_chg'] * 100)

    param = dict(y=ret, mean='zero')
    best_param = tuple()

    if vol == 'EGARCH':
        param['vol'] = 'EGARCH'

    elif vol == 'GJR-GARCH':
        param['o'] = 1

    elif vol == 'GARCH':
        param['vol'] = 'GARCH'
        param['o'] = 0

    bic_garch = []
    for p in range(1, 5):
        for q in range(1, 5):
            param.update(dict(p=p, q=q))
            garch = arch_model(**param).fit(disp='off')
            bic_garch.append(garch.bic)
            if garch.bic == np.min(bic_garch):
                best_param = p, q

    param.update(dict(p=best_param[0], q=best_param[1]))

    garch = arch_model(**param).fit(disp='off')
    pred = garch.forecast(start=ret.index[0])
    pred_sigma = np.sqrt(pred.variance) / 100

    return (pred_sigma.rolling(21).mean().loc[start_dt:] * np.sqrt(242)).shift(1)


def train_model(kernel, spt, **kwargs):
    """
    用当日价格变动+当月价格变动+当月波动率（年化），预测下月波动率（年化）
    """

    rets_daily = (spt['s_chg'] * 100) ** 2
    rets_monthly = ((spt['s_chg'] * 100).rolling(21).sum()) ** 2
    realized_vol = spt['s_chg'].rolling(21).std() * np.sqrt(242)  # 未来30天后的真实波动率

    d = pd.concat([rets_daily, rets_monthly, realized_vol], axis=1).dropna()
    d.columns = ['d', 'm', 'v']
    X = d[:-21].values
    y = d.iloc[21:]['v'].values.reshape(-1, )

    if kernel in ('linear', 'rbf', 'poly'):
        train_svr_model(kernel=kernel, X=X, y=y, **kwargs)
    elif kernel in ('nn'):
        train_nn_model(kernel=kernel, X=X, y=y, **kwargs)


def train_nn_model(kernel, X, y, **kwargs):
    NN_vol = MLPRegressor(learning_rate_init=0.001, random_state=1)
    para_grid_NN = {'hidden_layer_sizes': [(100, 50), (50, 50), (10, 100)],
                    'max_iter': [500, 1000],
                    'alpha': [0.00005, 0.0005]}

    clf = RandomizedSearchCV(NN_vol, para_grid_NN)
    clf.fit(X, y)
    models[kernel] = clf

    return clf


def train_svr_model(kernel, X, y, models, **kwargs):
    para_grid = {'gamma': sp_rand(),
                 'C': sp_rand(),
                 'epsilon': sp_rand()}

    kwargs.update({'kernel': kernel})
    print(kwargs)
    model = SVR(**kwargs)
    clf = RandomizedSearchCV(model, para_grid)
    clf.fit(X, y)

    models[kernel] = clf

    return clf


def pred_vols(spt, clf):
    rets_daily = (spt['s_chg'] * 100) ** 2
    rets_monthly = ((spt['s_chg'] * 100).rolling(21).sum()) ** 2
    realized_vol = spt['s_chg'].rolling(21).std() * np.sqrt(242)

    X = pd.concat([rets_daily, rets_monthly, realized_vol], axis=1).dropna()
    pred = clf.predict(X.values)
    pred = pd.DataFrame(pred, index=X.index, columns=['vol(pred)'])
    return pred


if __name__ == '__main__':

    data_path = os.path.join('.', 'history_data')
    model_path = os.path.join('.', 'models')
    vol_path = os.path.join('.', 'volatility')

    symbols = ['IC', 'IM', 'IF', 'IH']
    spts = dict()
    futs = dict()

    for sym in symbols:
        spts[sym] = pd.read_pickle(os.path.join(data_path, f'{sym}_spt.pkl'))
        futs[sym] = pd.read_pickle(os.path.join(data_path, f'{sym}_fut.pkl'))


    pred = dict()

    # garch_type = ['EGARCH', 'GARCH']
    # for sym in symbols:
    #     pred[sym] = dict()
    #     for g in garch_type:
    #         pred[sym][g]: pd.DataFrame = calc_garch(spts[sym], vol=g)
    #         pred[sym][g].to_pickle(os.path.join(vol_path, f'{sym}_{g}_vol.pkl'))