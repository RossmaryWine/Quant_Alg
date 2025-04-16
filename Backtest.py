import yfinance as yf
#import bloomberg if we can use API
import pandas as pd
import numpy as np
import time
import os
# from scipy import stats
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta.momentum

tick = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
start_ = "1991-01-01"
end_ = str(date.today())
close_df = yf.download(tick, start=start_, end=end_, interval='1d', actions=True)
close_df = close_df[['Close', 'Dividends']]
df = (close_df['Close'] + close_df['Dividends']).pct_change()
df.columns.name = None
df = df.reset_index()

#make sure the index is consistent
df.set_index("Date", inplace=True)
df.index = df.index.astype('datetime64[ns]')
#gets rid of rows with all NaNs
df = df.dropna(how='all')
#benchmark, used later
SPY = df[['SPY']]


def calc(df, tick, timeframe_general=62, time_signal=62, timeframe_TriMA=62, mask = 0b0000):
    # this line is subject to change, there could be an issue with how the calculatations are cancelling eachother out
    momentum = np.exp(np.log1p(df).rolling(window=timeframe_general).sum()) - 1
    # rid of NaN rows
    momentum = momentum.dropna(how='all')
    spy_momentum = momentum[['SPY']]
    momentum = momentum.drop(columns='SPY').dropna(how='all')

    #vola calc
    vola = df.rolling(window=timeframe_general).std(ddof=1)
    vola = vola.drop(columns='SPY')
    vola = vola.dropna(how='all')

    #calculate the 3 month MAs of each sector, with different MA techniques
    df = df.drop(columns=['SPY'])
    indicator_mean_rever_SMA = df.rolling(window=time_signal).mean().dropna(how='all') #SMA
    indicator_mean_rever_TriMA = indicator_mean_rever_SMA.rolling(window=timeframe_TriMA).mean().dropna(how='all') # doulbe SMA = Triangular MA
    indicator_mean_rever_EMA = df.ewm(span=time_signal, adjust=False).mean().dropna(how='all') # EMA
    tick_ = tick[1:] # rid of spy
    indicator_mean_rever_KAMA = pd.DataFrame(columns=tick_, index=df.index) 
    # no broacasting for ta module, had to use loop
    for col in tick_: # KAMA
        indicator_mean_rever_KAMA[col] = ta.momentum.KAMAIndicator(df[col], window=time_signal).kama()
    indicator_mean_rever_KAMA = indicator_mean_rever_KAMA.dropna(how='all')

    #mask for values for SMA, EMA, KAMA, TriMA mean reversion signals
    SMA_signal = indicator_mean_rever_SMA > df.reindex(indicator_mean_rever_SMA.index).values
    EMA_signal = indicator_mean_rever_EMA > df.reindex(indicator_mean_rever_EMA.index).values
    KAMA_signal = indicator_mean_rever_KAMA > df.reindex(indicator_mean_rever_KAMA.index).values
    TriMA_signal = indicator_mean_rever_TriMA > df.reindex(indicator_mean_rever_TriMA.index).values

    signal_list = [SMA_signal, EMA_signal, KAMA_signal, TriMA_signal]
    mask_list = []

    #masking bit to check if one MA is selected
    bit = 0b1000
    # loop through the masking value to get rid of uneeded signal
    for i in signal_list:
        if (mask & bit) == 0:
            i = pd.DataFrame(True, index=i.index, columns=i.columns)
        mask_list.append(i)
        bit >>= 1

    #3 month momentum signal
    momentum_signal = momentum > spy_momentum.reindex(momentum.index).values
    signal = momentum_signal

    # final mask anded with the selection of MA_signals on top of momentum
    for i in mask_list:
        signal &= i

    #signal into int for calculation
    signal = signal.astype(int)
    # multiply by vola for weighting calc
    signal *= vola
    signal = signal.replace(0, np.nan) # number * NaN = NaN

    inv_vol = 1/signal # invert
    weighting = inv_vol.div(inv_vol.sum(axis=1), axis=0) # weighting calculation, each value divided by sum of that row

    strat = weighting.shift()*df.reindex(weighting.index) # get the weights * etf values

    strat = (1 + strat.sum(axis=1)).cumprod() # the calc for the actual graph
    strat = strat.squeeze()
    
    return strat


def draw_graph(strat):
    SPYy = (1 + SPY.reindex(strat.index)).cumprod()
    SPYy = SPYy.squeeze()

    # Calculate Drawdowns
    strat_dd = (strat / strat.cummax()) - 1
    spy_dd = (SPYy / SPYy.cummax()) - 1

    # Create subplots

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Cumulative Returns", "Drawdowns"),
        row_heights=[0.6, 0.4], vertical_spacing=0.05
    )

    # Cumulative Returns Plot
    fig.add_trace(go.Scatter(x=strat.index, y=strat, name="Strategy", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=SPYy.index, y=SPYy, name="SPY", line=dict(color='red')), row=1, col=1)
    #fig.add_trace(go.Scatter(x=SPYy.index, y=SPYy, name="SPY", line=dict(color='cyan')), row=1, col=1)
    #fig.add_trace(go.Scatter(x=SPYy.index, y=SPYy, name="SPY", line=dict(color='pink')), row=1, col=1)

    # Drawdowns
    fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name="Strategy DD", line=dict(color='blue', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd, name="SPY DD", line=dict(color='red', dash='dot')), row=2, col=1)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=700,
        title="Performance and Drawdowns",
        yaxis1_title="Value",
        yaxis2_title="Drawdown",
    )

    # Drawdowns shown as %
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    #fig.update_yaxes(type='log')

    fig.show()

stratt = calc(df, tick, 62, 200, 200, 0b1111)
draw_graph(stratt)