import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import datetime


def calc_delta(vd_price, s0, **kwargs):
    _s = datetime.now()
    if np.isnan(vd_price):
        return np.nan
    vh = calc_payoff(vd_price=vd_price * 1.005, s0=s0, **kwargs)
    vl = calc_payoff(vd_price=vd_price * 0.995, s0=s0, **kwargs)
    delta = (vh - vl) / (0.02 * vd_price)
    print(datetime.now() - _s, s0, delta)
    return delta


def calc_payoff(sub_process: int, **kwargs):
    if isinstance(sub_process, int) and sub_process > 1:
        pool = mp.Pool(processes=sub_process)
        sub_kwargs = kwargs.copy()
        sub_kwargs['M'] = int(sub_kwargs['M'] / sub_process)
        res = pool.map(__calc_payoff, sub_process * [sub_kwargs])
        v = (sum(res) / sub_process)
        return v
    else:
        return _calc_payoff(**kwargs)


def __calc_payoff(params: dict):
    return _calc_payoff(**params)


def _calc_payoff(cal: pd.DataFrame,  # 交易日历
                 monitor,  # 观察日在cal中的序号
                 q=0,
                 s0=1,  # 期初资产价格
                 vd=0,  # 估值日距离期初日天数
                 vd_price=1,  # 估值日当期价格
                 T=1,  # 期限
                 r=0.03,  # 无风险利率
                 sigma=0.20,  # 波动率
                 k_out_ratio=1.03,  # 敲出价格
                 k_in_ratio=0.85,  # 敲入价格
                 k_coupon=0.1,  # 敲出票息
                 d_coupon=0.1,  # 到期票息
                 knocked_in=False,  # 是否已经敲入
                 M=10000,  # 模拟次数
                 Z=None,  # 随机数矩阵
                 **kwargs,
                 ) -> float:
    np.random.seed(42)

    residuals = cal.loc[vd:]
    residual_trading_days = residuals.index
    N = len(residual_trading_days) - 1

    dt = 1 / 243

    forward_r = vd / 243

    if Z is None:
        Z = np.random.standard_normal((N, M))
    else:
        Z = Z[:N, :M]

    delta_st = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    delta_st = np.concatenate((np.zeros((1, M)), delta_st))
    st = np.cumprod(np.exp(delta_st), axis=0) * vd_price
    st_df = pd.DataFrame(st,
                         index=residual_trading_days)
    kout_days = cal.loc[monitor].loc[
                vd:].index
    kout = np.tile(kout_days, (M, 1)).T

    kout = np.where(st_df.loc[kout_days] > k_out_ratio * s0, kout,
                    np.inf)

    kout_date = np.min(kout, axis=0)

    kin = np.any(st < k_in_ratio * s0, axis=0)

    pnl_kout = np.sum(
        kout_date[kout_date != np.inf] / 365 * k_coupon * np.exp(-r * kout_date[kout_date != np.inf] / 365))
    pnl_htm = np.count_nonzero((kout_date == np.inf) & (kin == False)) * d_coupon * np.exp(-r * T)
    pnl_loss = np.sum((st[-1, (kout_date == np.inf) & (kin == True) & (st[-1] < s0)] / s0 - 1) * np.exp(-r * T))

    if not knocked_in:
        V = (pnl_htm + pnl_kout + pnl_loss) / M
    else:
        V = (pnl_kout + pnl_loss) / M

    return V * np.exp(r * forward_r) * s0
