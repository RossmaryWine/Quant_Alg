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

def calc(df, timeframe=62, type='SMA'):
    SMA = df.rolling(window=timeframe).mean().dropna(how='all')
    EMA = df.ewm(span=timeframe, adjust=False).mean().dropna(how='all')
    tick_ = tick[1:]
    KAMA = pd.DataFrame(columns=tick_, index=df.index)
    for col in tick_:
        KAMA[col] = ta.momentum.KAMAIndicator(df[col], window=timeframe).kama()
    KAMA = KAMA.dropna(how='all')
    TriMA = SMA.rolling(window=timeframe).mean().dropna(how='all')
    type_dict = {'SMA' : SMA,
                'EMA' : EMA, 
                'KAMA' : KAMA,
                'TriMA' : TriMA}
    
    # this line is subject to change, there could be an issue with how the calculatations are cancelling eachother out
    momentum = np.exp(np.log1p(type_dict[type]).rolling(window=62).sum()) - 1
    # rid of NaN rows
    momentum = momentum.dropna(how='all')
    spy_momentum = momentum[['SPY']]
    momentum = momentum.drop(columns='SPY').dropna(how='all')

    #vola calc
    vola = type_dict[type].rolling(window=62).std(ddof=1)
    vola = vola.drop(columns='SPY')
    vola = vola.dropna(how='all')

    #calculate the 3 month moving average of each sector
    df = df.drop(columns=['SPY'])

    #3 month momentum signal
    momentum_signal = momentum > spy_momentum.reindex(momentum.index).values

    # mask anded with MA_signal on the better momentum
    signal = momentum_signal

    #signal into int for calculation
    signal = signal.astype(int)
    # multiply by vola for weighting calc
    signal *= vola
    signal = signal.replace(0, np.nan) # number * NaN = NaN
    #signal

    inv_vol = 1/signal # invert
    weighting = inv_vol.div(inv_vol.sum(axis=1), axis=0) # weighting calculation, each value divided by sum of that row

    strat = weighting.shift()*df.reindex(weighting.index) # get the weights * etf values

    strat = (1 + strat.sum(axis=1)).cumprod() # the calc for the actual graph
    strat = strat.squeeze()
    return strat

def draw_graph(strat, SPY):
    SPYy = (1 + SPY.reindex(strat.index)).cumprod()
    SPYy = SPYy.squeeze()

    # Calculate Drawdowns
    strat_dd = (strat / strat.cummax()) - 1
    spy_dd = (SPYy / SPYy.cummax()) - 1

    # === Create subplots ===

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=("Cumulative Returns", "Drawdowns"),
        row_heights=[0.6, 0.4], vertical_spacing=0.05
    )

    # --- Cumulative Returns Plot ---
    fig.add_trace(go.Scatter(x=strat.index, y=strat, name="Strategy", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=SPYy.index, y=SPYy, name="SPY", line=dict(color='red')), row=1, col=1)

    # --- Drawdowns Plot ---
    fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name="Strategy DD", line=dict(color='blue', dash='dot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd, name="SPY DD", line=dict(color='red', dash='dot')), row=2, col=1)

    # === Layout ===
    fig.update_layout(
        template="plotly_dark",
        height=700,
        title="Performance and Drawdowns",
        yaxis1_title="Value",
        yaxis2_title="Drawdown",
    )

    # Drawdowns shown as % (e.g., -30%)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    #fig.update_yaxes(type='log')

    fig.show()

stratt = calc(df, 62, 'SMA')
draw_graph(stratt, SPY)