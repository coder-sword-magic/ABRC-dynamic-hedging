import os

from tqdm import tqdm
import numpy as np
import pandas as pd

from data_table import prepare_bt_data, add_sigma
from monte_carlo import calc_delta

from datetime import datetime


def dynamic_hedging(df, small_target, multiplier=200, mgn_ratio=0.14, int_mgn=0.015, int_cash=0.055, sigma=None,
                    fee_ratio=0.001, **kwargs):
    t = df.copy()
    s0 = t.iloc[0]['close']
    t['delta'] = t.apply(
        lambda x: -calc_delta(cal=t, s0=s0, vd=x.name, vd_price=x['close'], sigma=x.vol, knocked_in=x['knocked_in'],
                              **kwargs), axis=1)
    t['hedging_value'] = -small_target * t['delta']
    t['sz'] = (t['hedging_value'] / (t['close'] * multiplier)).apply(lambda x: int(x))
    t['sz'] = np.where(abs(t['sz']) * multiplier * t['close'] > small_target,
                       small_target / (multiplier * t['close']) * t['sz'] / abs(t['sz']), t['sz'])
    t['sz'] = np.where(np.logical_and(t['knocked_in'] == True, t['sz'] < 0), 0, t['sz'])  # 当敲入的时候只做多不做空
    t.loc[t.index[-1], 'sz'] = 0
    t['sz_chg'] = (t['sz'] - t['sz'].shift(1)).fillna(t['sz']).apply(lambda x: int(x))
    t['mgn_trans'] = np.where(t['sz'] > 0, t['sz_chg'] * t['close'] * multiplier * mgn_ratio,
                              -t['sz_chg'] * t['close'] * multiplier * mgn_ratio)
    t['mgn_trans'] = np.where(t['sz'] * t['sz'].shift(1) < 0,
                              (abs(t['sz']) - abs(t['sz'].shift(1))) * t['close'] * multiplier * mgn_ratio,
                              t['mgn_trans'])
    t['variation'] = ((t['close'] - t['close'].shift(1)) * multiplier * t['sz'].shift(1)).fillna(0)
    t['mgn_acc'] = (t['mgn_trans'] + t['variation']).cumsum()
    t['cash_acc'] = small_target - (t['mgn_trans']).cumsum()
    t['int_mgn'] = (int_mgn * t['mgn_acc'].shift(1) * (t['d'] - t['d'].shift(1)).dt.days / 365).fillna(0).cumsum()
    t['int_cash'] = (int_cash * t['cash_acc'].shift(1) * (t['d'] - t['d'].shift(1)).dt.days / 365).fillna(0).cumsum()
    t['fee'] = (t['sz_chg'] * t['close'] * multiplier * fee_ratio).apply(lambda x: -abs(x)).cumsum()
    t['total_asset'] = t['mgn_acc'] + t['cash_acc'] + t['int_mgn'] + t['int_cash'] + t['fee']

    return t


def fix_pls_backtest(df,
                     fut,
                     small_target=1e8,  # 初始投资金额
                     mgn_acc_ratio=0.2,  # 保证金账户存款
                     cash_inv_rate=0.05,  # 现金账户利息收入
                     mgn_inv_rate=0.015,  # 保证金账户利息收入
                     fee_rate=0.003,  # 交易摩擦成本
                     mgn_ratio_i=0.14,  # 初始保证金
                     mgn_ratio_m=0.08,  # 维持保证金
                     stop_loss_ratio=0.95,  # 止损线
                     **kwargs,
                     ):
    orders = list()

    def add_order(sym, dt, price, side, sz, amt, mgn):
        return dict(sym=sym, dt=dt, price=price, side=side, sz=sz, amt=amt, mgn=mgn)

    init_cash = small_target * (1 - mgn_acc_ratio)  # 初始现金账户
    stop_loss = small_target * stop_loss_ratio  # 止损余额

    df['target_mkv'] = (small_target * df['delta']).fillna(0)  # 目标市值与delta方向，负数为做空
    df['target_mkv'] = np.where(abs(df['target_mkv']) > small_target, small_target, df['target_mkv'])  # 合约市值的上限

    ''' todo '''
    df['target_mkv'] = np.where(df['knocked_in'] == True, 0, df['target_mkv'])  # todo 设置亏损边界，敲入停止交易
    df.loc[df.index[-1], 'target_mkv'] = 0
    df['ct_mkv'] = np.nan
    df['fee'] = np.nan
    df['variation'] = np.nan
    df['hld'] = np.nan
    df['sz'] = 0
    df['chg_pos'] = np.nan
    df['mgn_call'] = False
    df['cash_bal'] = init_cash
    df['mgn_rsv'] = 0
    df['mgn_ocp'] = 0
    df['tot_ast_bi'] = 0
    df['passed'] = (df['d'] - df['d'].shift(1)).dt.days.fillna(0)
    df['cash_int'] = ((cash_inv_rate * (df['d'] - df.iloc[0]['d']).dt.days / 365) * init_cash)
    df['mgn_int'] = 0
    df['tot_ast'] = 0

    '''运行时变量'''
    mul = 200
    hld = None
    sz = 0
    ct_mkv = 0
    mgn_rsv = small_target * mgn_acc_ratio
    mgn_ocp = 0
    chg_pos = 0
    mgn_int = 0

    for vd in df.index:
        sp = df.loc[vd]
        f = fut.loc[sp['d']].sort_values('成交量(手)')  # 找到期货价格对应日期的数据
        main_ct = f.iloc[-1]  # 找到主力合约
        fee = 0
        variation = 0
        mgn_int += sp['passed'] / 365 * mgn_inv_rate * (mgn_rsv + mgn_ocp)  # 保证金利息

        if hld is None and sp['target_mkv'] != 0:
            '''
            新建仓：
            '''
            hld = main_ct.name  # 修改：持仓合约

            sz = int(sp['target_mkv'] / (main_ct['收盘价'] * mul))  # 计算：合约张数
            ct_mkv_tr = abs(main_ct['收盘价'] * mul * sz)  # 计算：合约市值
            open_fee = ct_mkv_tr * fee_rate  # 计算：该笔交易费用
            fee -= open_fee  # 计算：当日交易费用

            '''仓位变化'''
            mgn_ocp = ct_mkv_tr * mgn_ratio_i  # 计算：交易保证金
            mgn_rsv -= mgn_ocp  # 结算保证金 转入 交易保证金
            mgn_rsv -= open_fee  # 交易费用  计入  结算保证金

            '''记录订单'''
            orders.append(add_order(sym=hld,
                                    dt=sp['d'],
                                    price=main_ct['收盘价'],
                                    side='buy' if sz > 0 else 'sell',
                                    sz=abs(sz),
                                    amt=ct_mkv_tr,
                                    mgn=abs(mgn_ocp),
                                    ))

            '''结算变化'''
            open_var = (main_ct['结算价'] - main_ct['收盘价']) * sz * mul  # 计算：该笔交易-结算金额变动
            variation += open_var  # 计算：当日交易-结算金额变动
            mgn_ocp += open_var  # 结算金额变动 计入 交易保证金
            ct_mkv = main_ct['收盘价'] * mul * sz  # 计算：结算后合约市值

            df.loc[vd, 'hld'] = main_ct.name
            df.loc[vd, 'sz'] = sz
            df.loc[vd, 'variation'] = variation
            df.loc[vd, 'mgn_ocp'] = mgn_ocp
            df.loc[vd, 'mgn_rsv'] = mgn_rsv
            df.loc[vd, 'fee'] = fee
            df.loc[vd, 'ct_mkv'] = ct_mkv
            df.loc[vd, 'chg_pos'] = chg_pos
            df.loc[vd, 'mgn_int'] = mgn_int

            continue

        if main_ct.name != hld:  # 如果主力切换
            '''
            换合约：
            '''

            hld_ct = f.loc[hld]  # 定位持仓合约

            '''卖出'''
            hld_var = (hld_ct['收盘价'] - hld_ct['前结算']) * sz * mul  # 计算：该笔交易-结算金额变动
            variation += hld_var  # 计算：当日交易-结算金额变动
            ct_mkv_tr = abs(hld_ct['收盘价'] * mul * sz)
            close_fee = ct_mkv_tr * fee_rate  # 计算：该笔交易费用
            fee -= close_fee  # 计算：当日交易费用

            mgn_ocp += hld_var  # 结算金额变动 计入 交易保证金
            mgn_ocp -= close_fee  # 交易费用变动 计入 交易保证金
            mgn_rsv += mgn_ocp  # 交易保证金 转入 结算保证金

            orders.append(add_order(sym=hld,
                                    dt=sp['d'],
                                    price=hld_ct['收盘价'],
                                    side='buy' if -sz > 0 else 'sell',  # 当前持仓的反向交易
                                    sz=abs(sz),
                                    amt=ct_mkv_tr,
                                    mgn=abs(mgn_ocp),
                                    ))

            mgn_ocp = 0  # 交易保证金 清零

            '''买入'''
            hld = main_ct.name  # 修改：持仓合约
            target_sz = int(sp['target_mkv'] / (main_ct['收盘价'] * mul))  # 计算：合约张数
            if target_sz != sz:
                chg_pos += 1
            ct_mkv_tr = abs(main_ct['收盘价'] * mul * target_sz)  # 计算：合约市值
            open_fee = ct_mkv_tr * fee_rate  # 计算：该笔交易费用
            fee -= open_fee  # 计算：当日交易费用
            sz = target_sz

            '''仓位变化->保证金变化'''
            mgn_ocp = ct_mkv * mgn_ratio_i  # 计算：交易保证金
            mgn_rsv -= mgn_ocp  # 结算保证金 转入 交易保证金
            mgn_rsv += fee  # 交易费用  计入  结算保证金

            orders.append(add_order(sym=main_ct.name,
                                    dt=sp['d'],
                                    price=main_ct['收盘价'],
                                    side='buy' if sz > 0 else 'sell',  # 当前持仓的反向交易
                                    sz=abs(sz),
                                    amt=ct_mkv_tr,
                                    mgn=abs(mgn_ocp),
                                    ))

            '''结算变化->保证金变化'''
            open_var = (main_ct['结算价'] - main_ct['收盘价']) * sz * mul  # 计算：该笔交易-结算金额变动
            variation += open_var  # 计算：该笔结算保证金- 当日结算金额变动
            mgn_ocp += open_var  # 结算金额变动 计入 交易保证金
            ct_mkv = hld_ct['结算价'] * sz * mul

        else:
            '''主力没有变化观察每日变化'''

            '''调仓'''
            hld_ct = f.loc[hld]  # 定位持仓合约

            target_sz = int(sp['target_mkv'] / (main_ct['收盘价'] * mul))  # 计算：目标合约张数
            if target_sz != sz:
                chg_pos += 1

            diff_sz = target_sz - sz  # 计算：目标与持仓差额

            if diff_sz != 0:
                '''目标与持仓有差别'''
                hld_var = (main_ct['收盘价'] - main_ct['前结算']) * sz * mul  # 计算：该笔交易-结算金额变动 （假设原有份额持有到收盘）
                variation += hld_var  # 计算：该笔结算保证金- 当日结算金额变动
                mgn_ocp += hld_var  # 结算金额变动 计入 交易保证金

                ct_mkv_tr = abs(main_ct['收盘价'] * mul * abs(diff_sz))  # 计算：交易市值
                fee_tr = ct_mkv_tr * fee_rate  # 计算：该笔交易费用
                fee -= fee_tr  # 计算：当日交易费用

                '''仓位变化->保证金变化'''
                mgn_chg = (abs(target_sz) - abs(sz)) * main_ct['收盘价'] * mul * mgn_ratio_i  # 调仓：计算 交易保证金变动
                mgn_rsv -= mgn_chg  # 保证金变动 计入 结算保证金
                mgn_ocp += mgn_chg  # 保证金变动 计入 交易保证金
                mgn_rsv -= fee_tr  # 交易费用 计入 结算保证金

                '''无持仓-> 结算交易保证金'''
                if target_sz == 0:
                    mgn_rsv += mgn_ocp
                    mgn_ocp = 0

                '''结算变化->保证金变化'''
                sz = target_sz
                hld_var_2 = (main_ct['结算价'] - main_ct['收盘价']) * sz * mul  # 计算：调仓后 - 结算金额变动
                variation += hld_var_2  # 计算：该笔结算保证金- 当日结算金额变动
                mgn_ocp += hld_var_2  # 结算金额变动 计入 交易保证金
                ct_mkt = hld_ct['结算价'] * mul * sz

                orders.append(add_order(sym=hld,
                                        dt=sp['d'],
                                        price=main_ct['收盘价'],
                                        side='buy' if diff_sz > 0 else 'sell',  # 当前持仓的反向交易
                                        sz=abs(diff_sz),
                                        amt=ct_mkv_tr,
                                        mgn=abs(mgn_chg),
                                        ))

            else:
                '''目标与持仓无差别'''
                hld_ct = f.loc[hld]
                ct_mkv = hld_ct['结算价'] * mul * sz  # 计算：合约市值

                '''结算变化->保证金变化'''
                hld_var = (main_ct['结算价'] - main_ct['前结算']) * sz * mul  # 计算：该笔交易-结算金额变动
                variation += hld_var  # 计算：当日交易-结算金额变动
                mgn_ocp += hld_var

        df.loc[vd, 'hld'] = hld
        df.loc[vd, 'sz'] = sz
        df.loc[vd, 'variation'] = variation
        df.loc[vd, 'mgn_ocp'] = mgn_ocp
        df.loc[vd, 'mgn_rsv'] = mgn_rsv
        df.loc[vd, 'fee'] = fee
        df.loc[vd, 'ct_mkv'] = ct_mkv
        df.loc[vd, 'chg_pos'] = chg_pos
        df.loc[vd, 'mgn_int'] = mgn_int

        if mgn_rsv + mgn_ocp + init_cash < stop_loss:  # 低于止损金额，第二个交易日收盘价平仓
            df.loc[vd:, 'target_mkv'] = 0
            df.loc[vd:, 'mgn_call'] = True

    df['tot_ast_bi'] = df['mgn_rsv'] + df['mgn_ocp'] + df['cash_bal']
    df['tot_ast'] = df['tot_ast_bi'] + df['mgn_int'] + df['cash_int']

    order_df = pd.DataFrame(orders)
    order_df = order_df[order_df['sz'] != 0]

    return df, order_df


def batch_fix_pls_backtest(fut, spt, pro_args: dict, dt_range: list, bt_result: dict, vol_pred: dict, **kwargs):
    fmt = '%Y%m%d_%H%M'
    bt_time = datetime.strftime(datetime.now(), fmt)

    if bt_time not in os.listdir('bt_result'):
        os.mkdir(os.path.join('bt_result', f'{bt_time}'))
        os.mkdir(os.path.join('bt_result', f'{bt_time}', 'raw_data'))

    with open(os.path.join('bt_result', f'{bt_time}', 'product_args'), 'a') as file:
        file.write(bt_time + '\n')
        for k, v in pro_args.items():
            file.write(str(k) + ' = ' + str(v) + '\n')

    bt_data = dict()
    orders_df = pd.DataFrame()

    period = pro_args['period']
    pro_args.update(kwargs)
    small_target = pro_args['small_target']

    for st_date in tqdm(dt_range):

        data = prepare_bt_data(fut, spt, pro_args['k_in_ratio'], pro_args['k_out_ratio'], st_date, period)
        s0 = data['s'].iloc[0]['close']
        pro_args.update(dict(cal=data['s'], monitor=data['m'], s0=s0))

        for k, pred in vol_pred.items():
            bt_data[k] = add_sigma(data['s'], pred)

            bt_data[k]['delta'] = bt_data[k].apply(
                lambda x: calc_delta(vd=x.name, vd_price=x['close'], sigma=x['vol'], knocked_in=x['knocked_in'],
                                     **pro_args), axis=1)
            bt_data[k], order = fix_pls_backtest(df=bt_data[k], fut=fut, **pro_args)

            order['t0'] = st_date.date()
            orders_df = pd.concat([orders_df, order], axis=0)

            if k not in bt_result:
                bt_result[k] = {'rets': [], 'rets_bi': [], 'pnl_ratio': [], 'trading_days': [], 'start_dt': [],
                                's0': [], 'vol_0': [], 'vol_max': [], 'vol_min': []}

            rets = bt_data[k].iloc[-1]['tot_ast'] / small_target - 1
            rets_bi = bt_data[k].iloc[-1]['tot_ast_bi'] / small_target - 1
            chg_pnl = bt_data[k].groupby('chg_pos')['variation'].sum()
            pnl_ratio = abs(chg_pnl[chg_pnl > 0].mean()) / abs(chg_pnl[chg_pnl < 0].mean())
            trading_days = bt_data[k].query('sz!=0').index[-1]
            vol_max = bt_data[k].query('sz!=0')['vol'].max()
            vol_min = bt_data[k].query('sz!=0')['vol'].min()

            bt_result[k]['rets'].append(rets)
            bt_result[k]['rets_bi'].append(rets_bi)
            bt_result[k]['pnl_ratio'].append(pnl_ratio)
            bt_result[k]['trading_days'].append(trading_days)
            bt_result[k]['start_dt'].append(st_date)
            bt_result[k]['s0'].append(s0)
            bt_result[k]['vol_0'].append(bt_data[k].iloc[0]['vol'])
            bt_result[k]['vol_max'].append(vol_max)
            bt_result[k]['vol_min'].append(vol_min)

            pd.DataFrame(bt_result[k]).to_csv(os.path.join('bt_result', f'{bt_time}', f'{fut["sym"].iloc[0]}_{k}.csv'))
            bt_data[k].to_csv(
                os.path.join('bt_result', f'{bt_time}', 'raw_data', f'{fut["sym"].iloc[0]}_{k}_{st_date.date()}.csv'))
            orders_df.to_csv(os.path.join('bt_result', f'{bt_time}', f'{fut["sym"].iloc[0]}_{k}_orders.csv'))

    return bt_result


if __name__ == '__main__':

    import argparse
    from dateutil import parser

    arg_parser = argparse.ArgumentParser(
        prog='back_test',
        description='back test futures'
    )
    arg_parser.add_argument('-s', '--symbol', required=True)
    arg_parser.add_argument('-i', '--initial', default='2001-01-01')
    arg_parser.add_argument('-e', '--end', default='2022-12-31')
    arg_parser.add_argument('-p', '--process', default=1)
    arg_parser.add_argument('-z', default=False)
    arg_parser.add_argument('-m', default=10000)

    args = arg_parser.parse_args()

    assert args.symbol in ['IC', 'IF', 'IM', 'IH'], '仅限于 IC / IF / IM / IH'
    assert int(args.m)
    assert parser.parse(args.initial)
    assert parser.parse(args.end)

    sym = args.symbol
    garch_type = ['GARCH', 'EGARCH']
    _vol_pred = list(filter(lambda x: True if sym in x else False, os.listdir('volatility')))
    vol_pred = dict()
    for g in garch_type:
        for pred in _vol_pred:
            if g in pred:
                vol_pred[g] = pd.read_pickle(os.path.join('volatility', pred)).dropna()

    _spt = list(filter(lambda x: True if f'{sym}_spt.pkl' == x else False, os.listdir('history_data')))[0]
    _fut = list(filter(lambda x: True if f'{sym}_fut.pkl' == x else False, os.listdir('history_data')))[0]
    spt = pd.read_pickle(os.path.join('history_data', _spt))
    fut = pd.read_pickle(os.path.join('history_data', _fut))
    from cfg import pro_args

    dt_range = fut.index.levels[0]
    dt_range = dt_range.to_frame().loc[args.initial:args.end].index

    bt_result = dict()

    if args.z is True: # 输入值就是true
        Z = np.random.standard_normal((252, 10000))  # 指定用同样的随机数
    else:
        Z = None

    print(Z,int(args.process),int(args.m))

    bt_result = batch_fix_pls_backtest(fut, spt, pro_args, dt_range, bt_result, vol_pred,
                                       Z=Z,
                                       sub_process=int(args.process),
                                       M=int(args.m)
                                       )
