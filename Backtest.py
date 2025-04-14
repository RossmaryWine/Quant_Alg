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

#extract data from already existing csv file on computers
df = pd.read_csv(r"C:\Users\thefa\quant\check_tot_return.csv")

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

Close_ = close_df[['Close']]
Close_.columns = Close_.columns.get_level_values(1)
Close_.columns.name = None
Close_ = Close_.reset_index()
Close_.set_index("Date", inplace=True)
Close_.index = Close_.index.astype('datetime64[ns]')

# this line is subject to change, there could be an issue with how the calculatations are cancelling eachother out
momentum = np.exp(np.log1p(df).rolling(window=62).sum()) - 1
# rid of NaN rows
momentum = momentum.dropna(how='all')
spy_momentum = momentum[['SPY']]
momentum = momentum.drop(columns='SPY').dropna(how='all')

#vola calc
vola = df.rolling(window=62).std(ddof=1)
vola = vola.drop(columns='SPY')
vola = vola.dropna(how='all')

#calculate the 3 month moving average of each sector
df = df.drop(columns=['SPY'])
Close_ = Close_.drop(columns=['SPY']).reindex(df.index)
indicator_mean_rever_SMA = df.rolling(window=63).mean().dropna(how='all')
indicator_close_MA = Close_.rolling(window=21).mean().dropna(how='all') # ass
indicator_mean_rever_EMA = df.ewm(span=200, adjust=False).mean().dropna(how='all')

#mask for values for SMA and EMA mean reversion signals
SMA_signal = indicator_mean_rever_SMA > df.reindex(indicator_mean_rever_SMA.index)
SMA_signal2 = indicator_close_MA > Close_.reindex(indicator_close_MA.index) # ass
EMA_signal = indicator_mean_rever_EMA > df.reindex(indicator_mean_rever_EMA.index)

#3 month momentum signal
momentum_signal = momentum > spy_momentum.reindex(momentum.index).values

# mask anded with MA_signal on the better momentum
signal = momentum_signal & EMA_signal

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
SPYy = (1 + SPY.reindex(strat.index)).cumprod()

#turn into series first
strat = strat.squeeze()
SPYy = SPYy.squeeze()

# calc drawdowns
strat_dd = (strat / strat.cummax()) - 1
spy_dd = (SPYy / SPYy.cummax()) - 1


# subplots
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=("Cumulative Returns", "Drawdowns"),
    row_heights=[0.6, 0.4], vertical_spacing=0.05
)

# cumulative return
fig.add_trace(go.Scatter(x=strat.index, y=strat, name="Strategy", line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=SPYy.index, y=SPYy, name="SPY", line=dict(color='red')), row=1, col=1)

#Drawdowns
fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd, name="Strategy DD", line=dict(color='blue', dash='dot')), row=2, col=1)
fig.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd, name="SPY DD", line=dict(color='red', dash='dot')), row=2, col=1)

# layout
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
