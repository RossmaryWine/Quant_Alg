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
            self.volatility = 1000
            return 1
        
        #filter out useless columns
        self.ticker_df = self.ticker_df[['Close', 'Dividends']]
        self.tot_return = (self.ticker_df['Close'] + self.ticker_df['Dividends']).pct_change()
        save_df_csv(self.tot_return, 'thisissoshit')

        for ticker in self.tick:  # Assuming ticker_list contains SPY, XLF, etc.
            self.ticker_df[('temp', ticker)] = self.ticker_df[('Close', ticker)] + self.ticker_df[('Dividends', ticker)]
            self.ticker_df[('Tot_Return', ticker)] = self.ticker_df[('temp', ticker)].pct_change()
            self.ticker_df[("Log_Return", ticker)] = np.log1p(self.ticker_df[('Tot_Return', ticker)])
            #self.ticker_df[("TRI", ticker)] = 100 * np.exp(self.ticker_df[('Log_Return', ticker)].cumsum())
            self.ticker_df.drop(columns=[('temp', ticker)], inplace=True)
            #self.ticker_df.drop(columns=[('Tot_Return', ticker)], inplace=True)

        #self.ticker_df = self.ticker_df.fillna(0)
        
        return 0


    def calc_vola(self, days=62):
        for ticker in self.tick:
            self.ticker_df[('Vol_3M', ticker)] = (
                    self.ticker_df[("Tot_Return", ticker)].rolling(window=days).std()#*np.sqrt(252)
                    )

    def calc_momentum(self, days=62):

        for i in range(0, len(self.tick)):
            # convert to simple momentum
            self.ticker_df[('Sim_Mom_3M', self.tick[i])] = (
                    np.exp(self.ticker_df[("Log_Return", self.tick[i])].rolling(window=days).sum()) - 1#*np.sqrt(252)
                    )
        # reset index for daily return calc
        save_df_csv(self.ticker_df, "test_tick_all")
        return 0
    

    def main_df_format(self):
        main_frame = pd.DataFrame()
        # ranking with simple 3 month momentum
        baseline = self.ticker_df[('Sim_Mom_3M', 'SPY')]
        self.ticker_df.drop(columns=[('Sim_Mom_3M', 'SPY')], inplace=True)
        mom_df = self.ticker_df['Sim_Mom_3M']
        adj_df = self.ticker_df['Vol_3M']
        close_df = self.ticker_df['Close']
        main_frame['baseline'] = baseline

        tot_weigh = pd.Series(0, index=main_frame.index)
        for i in range(1, len(self.tick)):
            mask = mom_df[self.tick[i]] >= baseline # True for eligible tickers

            # Keep only eligible momentum values
            eligible_return = adj_df[self.tick[i]].where(mask, np.nan)
            main_frame[f'filtered_{self.tick[i]}'] = eligible_return

            inverse_weigh = 1/eligible_return.replace(0, np.nan)
            main_frame[f'inverse_{self.tick[i]}'] = inverse_weigh

            tot_weigh = tot_weigh.add(inverse_weigh, fill_value=0)

        for i in range(1, len(self.tick)):
            main_frame[f'tru_w_{self.tick[i]}'] = main_frame[f'inverse_{self.tick[i]}']/tot_weigh

        main_frame.fillna(0, inplace=True)
        final_index = pd.Series(0, index=main_frame.index)

        # 
        for i in range(1, len(self.tick)):
            #print(self.tick[i])
            final_index = final_index.add((main_frame[f'tru_w_{self.tick[i]}']*close_df[self.tick[i]]), fill_value=0)
            #print(final_index[-1])
        final_frame = pd.DataFrame(index=main_frame.index)
        final_frame['strat_performance'] = final_index
        save_df_csv(final_frame, 'value')

        save_df_csv(main_frame, 'huge test')
        return final_frame


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
    """
    process_SP500.calc_vola()
    process_SP500.calc_momentum()
    final_frame = process_SP500.main_df_format()
    make_graph(final_frame) 
"""
if __name__ == "__main__":
    main()