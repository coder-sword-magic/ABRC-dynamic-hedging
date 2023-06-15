import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
import re
import os

data_path = os.path.join('.', 'history_data')
model_path = os.path.join('.', 'models')
WOM_3FRI = pd.date_range('2001-01-01', '2023-12-31', freq='WOM-3FRI').to_frame(name='date')  # 每月第3个周五为期货交割日


# IC 中证500
# IH 上证50
# IF 沪深300
# IM 中证1000
def fetch_spot_futures(sym='IC', from_='2020-01', to_='2022-12'):
    import akshare as ak
    """
    从akshare下载数据
    """
    spots = {'IC': 'sh000905',
             'IH': 'sh000016',
             'IF': 'sh000300',
             'IM': 'sh000852', }

    futures = dict()  #
    time_idx = pd.date_range(parser.parse(from_), parser.parse(to_), freq='M')

    futures_list = [sym + datetime.strftime(x, '%y%m') for x in time_idx]
    for symbol in futures_list:
        df = ak.futures_zh_daily_sina(symbol=symbol)
        futures[symbol] = pd.DataFrame([])
        futures[symbol][symbol] = df['close']
        futures[symbol]['date'] = df['date']
        futures[symbol]['date'] = pd.to_datetime(futures[symbol]['date'])
        futures[symbol].set_index('date', inplace=True)

    fut = pd.concat(futures.values(), axis=1, join='outer')

    spot = ak.stock_zh_index_daily(symbol=spots[sym])  # 中证1000
    spot['spot'] = spot['close']
    spot['date'] = pd.to_datetime(spot['date'])
    spot.set_index('date', inplace=True)

    return fut, spot


def fetch_raw_glidict():
    """
    读取期货数据
    来源：聚源数据
    """
    raw = pd.read_csv(os.path.join(data_path, '期货历史行情.csv'))
    v = ['ID', '合约内部编码', '交易日期', '合约代码', '交易所代码', '合约标的', '合约序列标志', '前结算', '前收盘',
         '开盘价', '最高价', '最低价', '收盘价', '收盘较前结算涨跌', '收盘较前结算涨跌幅(%)', '收盘价涨跌',
         '收盘价涨跌幅(%)', '结算价', '结算价涨跌', '结算价涨跌幅(%)', '持仓量(手)', '持仓量变化(手)',
         '持仓量变化幅度(%)', '成交量(手)', '成交量变化(手)', '成交量变化幅度(%)', '成交金额(元)',
         '成交金额变化(元)', '成交金额变化幅度(%)', '基差', '更新时间', 'JSID', '主力标志',
         'basisannualyield', 'dt']
    k = list(raw.columns)
    col = dict(zip(k, v))
    raw = raw.rename(columns=col)
    df = raw[['交易日期', '合约代码', '前结算', '开盘价', '收盘价', '结算价', '成交量(手)']].copy()
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    df['sym'] = df['合约代码'].str.replace(r'(\d+)', '')
    return df


def fetch_spot_history(sym):
    """
    读取股指历史数据
    数据来源:akshare(东方财富)
    """

    if f'{sym}.pkl' in os.listdir(data_path):
        return pd.read_pickle(os.path.join(data_path, f'{sym}.pkl'))

    import akshare as ak
    spots = {'IC': 'sh000905',
             'IH': 'sh000016',
             'IF': 'sh000300',
             'IM': 'sh000852', }
    s = ak.stock_zh_index_daily(symbol=spots[sym])
    s['date'] = pd.to_datetime(s['date'])
    s.set_index('date', inplace=True)
    s = s[['close']]
    s['s_chg'] = np.log(s['close'] / s['close'].shift(1))
    s['vol(real)'] = s['s_chg'].rolling(5).std() * np.sqrt(242)
    return s.dropna()


def ex_dt(x):
    """
    判断期货交割日
    """
    dt_str = re.search(r'\d{4}', x).group()
    fmt = '%y%m'
    dt = datetime.strptime(dt_str, fmt)
    return WOM_3FRI.loc[dt:, 'date'].iloc[0]


def fetch_fut_history(sym, spt):
    """
    整理特定品种期货数据
    来源：聚源数据
    """
    df = fetch_raw_glidict()

    f = df[df['sym'] == sym].reset_index().drop('index', axis=1).copy()
    f = pd.merge(f, spt[['close']], left_on='交易日期', right_index=True)
    f = f.rename(columns={'close': '股指收盘价'})
    f['基差'] = f['收盘价'] / f['股指收盘价'] - 1
    f['合约交割日'] = f['合约代码'].apply(ex_dt)
    f['剩余日期'] = (f['合约交割日'] - f['交易日期']).dt.days
    f['升/贴水率(年化)'] = f['基差'] * 365 / f['剩余日期']
    f = f.groupby(['交易日期', '合约代码']).last().sort_index(level=0)

    return f


def prepare_bt_data(futures, spot, knock_in_ratio, knock_out_ratio, start_dt, period=365) -> dict:
    """
    根据产品要素生成历史回测数据
    """

    if isinstance(start_dt, str):
        start_dt = parser.parse(start_dt)

    s = pd.DataFrame(index=pd.date_range(start_dt, periods=period, freq='D'))
    s = pd.concat([s, spot[['close']]], axis=1, join='inner')
    s['vd'] = (s.index - start_dt).days

    monitor_days = list(
        pd.date_range(start_dt.date(), periods=int(period / 12), freq="M") + pd.Timedelta(start_dt.day - 1, 'D'))
    monitor_days[-1] = s.index[-1]

    s['d'] = s.index
    s = s.set_index('vd')

    monitor = s[s['d'].isin(monitor_days)].index

    knock_in_price = spot[start_dt:].iloc[0]['close'] * knock_in_ratio  # 现货价格
    knock_out_price = spot[start_dt:].iloc[0]['close'] * knock_out_ratio

    monitor_sp = s.loc[monitor]
    if not monitor_sp[monitor_sp['close'] > knock_out_price].empty:
        end_dt = monitor_sp[monitor_sp['close'] > knock_out_price].index[0]
        s.loc[end_dt:, 'close'] = np.nan
        monitor = list(monitor[:monitor.get_loc(end_dt) + 1])

    knocked_in = s[s['close'] <= knock_in_price]
    s['knocked_in'] = np.nan
    if not knocked_in.empty:
        knocked_in_idx = knocked_in.index[0]
        s.loc[knocked_in_idx:, 'knocked_in'] = True
    s['knocked_in'] = s['knocked_in'].fillna(False)

    return dict(s=s, m=monitor, ki=knock_in_price, ko=knock_out_price)


def add_sigma(df, vol):
    vol.columns = ['vol']
    df = pd.merge(df, vol, left_on='d', right_index=True, how='left')
    df['vol'] = np.where(df['close'].isna(), np.nan, df['vol'])
    return df


if __name__ == '__main__':
    symbols = ['IC', 'IM', 'IF', 'IH']
    spts = dict()
    futs = dict()

    for sym in symbols:
        spts[sym] = fetch_spot_history(sym)
        spts[sym].to_pickle(os.path.join(data_path, f'{sym}_spt.pkl'))
        futs[sym] = fetch_fut_history(sym, spts[sym])
        futs[sym].to_pickle(os.path.join(data_path, f'{sym}_fut.pkl'))
