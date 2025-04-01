import yfinance as yf
#import bloomberg if we can use API
import pandas as pd
import numpy as np
import time
import os
# from scipy import stats
from statistics import mean # note to self: mean() takes in one list of some sort, NOT multiple values
from datetime import date
import pandas_datareader.data as web
import plotly.graph_objects as go


# remaking the elements within the df_dict into a class
# pass the elements into the class attributes instead of hotwiring them every time
# 
class TickData:


    def __init__(self, start, end, ticker):
        self.start = start
        self.end = end
        self.tick = ticker # list
        self.ticker_df = pd.DataFrame()
        self.tot_return = pd.DataFrame()
        self.volatility = pd.DataFrame()
        self.momentum = pd.DataFrame()
        self.volatility_baseline = 0
        self.HQM = 0
        self.vola_adj_re = 0
        self.M1 = 0


    def scrape_tick(self):
        # get tickers info from current selected time
        tickers = yf.Tickers(" ".join(self.tick))
        # there is a slight bug that makes yf extract a few days less of data than what is required. future possible fix ticket
        self.ticker_df = yf.download(self.tick, start=self.start, end=self.end, interval='1d', actions=True)
        if self.ticker_df.index.empty:
            print(f"No data for {tickers}, skipping...")
            self.ticker_df = pd.DataFrame()
            return 1
        
        #filter out useless columns
        self.ticker_df = self.ticker_df[['Close', 'Dividends']]
        #total return
        self.tot_return = (self.ticker_df['Close'] + self.ticker_df['Dividends']).pct_change()
        #TRI = 100 * np.exp(self.ticker_df['Log_Return'].cumsum())
        
        return 0


    def calc_vola(self, days=62):
        self.volatility = self.tot_return.rolling(window=days).std()
        #save_df_csv(self.volatility, 'vola_test')

    def calc_momentum(self, days=62):
        self.momentum = np.exp(np.log1p(self.tot_return).rolling(window=days).sum()) - 1
        #save_df_csv(self.momentum, 'momentum_test')


    def main_df_format(self):
        main_frame = pd.DataFrame()
        # ranking with simple 3 month momentum
        baseline = self.momentum['SPY']
        self.momentum.drop(columns=['SPY'], inplace=True)

        baseline = baseline.reindex(self.momentum.index)
        temp = self.volatility.where(self.momentum > baseline.values[:, None], other=0) 
        temp2 = self.momentum.where(self.momentum > baseline.values[:, None], other=0)
        save_df_csv(temp2, 'test_multicolumn_mask')

        tot_weigh = pd.Series(0, index=self.momentum.index)
        
        temp = temp.dropna(how='all')
        temp = temp.replace(0, np.nan)
        temp = 1/temp
        tot_weigh = temp.div(temp.sum(axis=1), axis=0)

        #save_df_csv(tot_weigh, 'tot_weigh')

        close = self.ticker_df['Close']
        close.drop(columns=['SPY'], inplace=True)
        #save_df_csv(close, 'close')
        value = close*tot_weigh.shift()
        value = value.sum(axis=1).to_frame()

        save_df_csv(value, 'value_added')

        return value


def save_df_csv(df, tick):

    if os.path.exists(f"check_{tick}.csv"):
        try:
            os.remove(f"check_{tick}.csv")
        except PermissionError:
            print("the file is currently open, force shut down")
            os.system(f"taskkill /f /im excel.exe")
            time.sleep(0.7)
            os.remove(f"check_{tick}.csv")
        print(f"deleted prev version of check_{tick}.csv")

    df.to_csv(f"check_{tick}.csv")


def extract_csv(tick):
    df = pd.read_csv(f"check_{tick}")
    return df


def make_graph(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.iloc[:, 0], mode='lines', name='Value'))

    fig.update_layout(
        title="Time Series Graph",
        xaxis_title="Date",
        yaxis_title="Value",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    fig.show()
    return 0


def main():

    #list of SPX sectors including SPX itself
    sect_list = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

    start_ = "1991-01-01"
    end_ = str(date.today())
    #end_ = "2009-01-01"
    start_adj = pd.to_datetime(end_) - pd.DateOffset(years=1, days=5)
    end_adj = pd.to_datetime(start_) + pd.DateOffset(years=1)

    process_SP500 = TickData(start_, end_, sect_list)
    process_SP500.scrape_tick()
    process_SP500.calc_vola()
    process_SP500.calc_momentum()
    final_frame = process_SP500.main_df_format()
    make_graph(final_frame) 
    """
"""
if __name__ == "__main__":
    main()