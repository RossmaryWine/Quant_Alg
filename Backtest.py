import yfinance as yf
#import bloomberg if we can use API
import pandas as pd
import numpy as np
import time
import os
# from scipy import stats
from statistics import mean
from datetime import date
import pandas_datareader.data as web

# note to self: mean() takes in one list of some sort, NOT multiple values

df_dict = {}
cpi = pd.DataFrame(columns=['CPIAUCSL', 'Inflation_YoY', 'Inflation_MoM'])
# dict containing the different lookback periods for momentum and normalized return calulation
momentum_diff_dict = {
    "1MOmomentum" : 1,
    "3MOmomentum" : 3,
    "6MOmomentum" : 6,
    "1YRmomentum" : 12
    }


# remaking the elements within the df_dict into a class
# pass the elements into the class attributes instead of hotwiring them every time
# 
class TickData:


    def __init__(self, start, end, ticker):
        self.start = start
        self.end = end
        self.tick = ticker
        self.tick_df = pd.DataFrame()
        self.volatility = 0
        self.HQM = 0
        self.vola_adj_re = 0
        self.M1 = 0


    def get_money_supply(self):
        # reading the last bit of the M1 data for US(most recent)
        M1 = web.DataReader("WM1NS", "fred", pd.to_datetime(self.end) - pd.Timedelta(days=60), self.end)
        self.M1 = M1
        return M1["WM1NS"].iloc[-1]


    def scrape_tick(self):
        # get ticker info from current selected time to a year from then
        ticker = yf.Ticker(self.tick)
        # there is a slight bug that makes yf extract a few days less of data than what is required. future possible fix ticket
        ticker_df = yf.download(self.tick, start=self.start, end=self.end, interval='1d')
        if ticker_df.index.empty:
            print(f"No data for {ticker}, skipping...")
            self.tick_df = pd.DataFrame()
            self.volatility = 1000
            return 1
        
        # get the dividends of the ticker
        div_ = ticker.dividends

        #remove timezone from it being new york
        try:
            div_.index = div_.index.tz_localize(None)
        # this error doesn't occur unless there is nothing in the date column, AKA failed to retrieve data
        except AttributeError:
            print("failed to retrieve info! possibly due to ticker name or yf timing out")
            self.tick_df = pd.DataFrame()
            self.volatility = 1000
            return 2

        # extract the dividends of specificed start to end date
        dividends_filtered = div_.loc[self.start:self.end]
        # total_dividends = dividends_filtered.sum()

        # Convert dividends to DataFrame
        dividends_df = dividends_filtered.to_frame().reset_index()
        dividends_df.columns = ["Date", "Dividends"] # set the columns

        # reset index
        ticker_df = ticker_df.reset_index()

        # the ticker df defaults to a bunch of multi-indexes, to merge later with dividend df, have to convert 
        # the multi-indexes into normal ones, we go through all columns and join the pairs with _(chatGPT)
        ticker_df.columns = ['_'.join(col).strip() if isinstance(col, tuple)\
                            else col for col in ticker_df.columns]
        # reset the Date_ column after conjointing, there is a pair column for some reason that looks like ('Date', '')
        ticker_df.rename(columns={"Date_": "Date"}, inplace=True)

        # reset index again
        ticker_df = ticker_df.reset_index()
        # print(based.columns) debug statement

        # merge 2 dfs with Date column as the merge key and add matching dividends rows, if none dividend = NaN(left join)
        ticker_df = ticker_df.merge(dividends_df, on="Date", how="left")

        # Fill missing dividend values with 0
        ticker_df.fillna({"Dividends" : 0}, inplace=True)

        # use the total return formula on every single row: (Pend + D - Pstart)/Pstart
        # first row no value, replace with 0
        ticker_df[f"Total_Return_{self.tick}"] = [((ticker_df[f"Close_{self.tick}"][i] + ticker_df["Dividends"][i] - \
                                                ticker_df[f"Close_{self.tick}"][i - 1])/ticker_df[f"Close_{self.tick}"][i - 1]) \
                                                if i != 0 else 0 for i in range(len(ticker_df[f"Close_{self.tick}"]))]
        
        self.tick_df = ticker_df
        self.tick_df["Date"] = pd.to_datetime(self.tick_df["Date"])
        self.tick_df.set_index("Date", inplace=True)
        return 0


    def calc_vola(self, periods = {'1MO' : 21, '3MO' : 60, '6MO' : 120, '1YR' : 252}):
        start_adj = pd.to_datetime(self.end) - pd.DateOffset(years=1)
        ticker_df = self.tick_df.loc[start_adj:self.end].copy()
        # use the volatility formula to get vola
        for name, i in periods.items():
            actual_day = min(i, len(ticker_df))
            self.tick_df[f"Vol_{name}"] = (
                    # important, the formula for volatility calculation is:
                    # std(return over selected period) X root(SELECTED PERIOD)
                    # and to annualize it for consistency, multiply again with 
                    # root(252/selected period)
                    # could also consider rolling().std() if needed, instead of all same value as most recent value
                    np.std(ticker_df[f"Total_Return_{self.tick}"].iloc[-actual_day:])\
                    *np.sqrt(actual_day)*np.sqrt(252/actual_day)
                    )
        
        # standard annualized vola calc, same as the 1 yr vola
        # self.tick_df = ticker_df
        self.volatility = np.std(ticker_df[f"Total_Return_{self.tick}"].dropna()) * np.sqrt(252)


    def calc_momentum(self, main_df):
        global cpi
        global momentum_diff_dict

        # Merge CPI data using index instead of 'Date' column
        self.tick_df = self.tick_df.merge(cpi[["Inflation_YoY"]], left_index=True, right_index=True, how="left")
        self.tick_df = self.tick_df.merge(cpi[["Inflation_MoM"]], left_index=True, right_index=True, how="left")

        # Fill missing values
        self.tick_df["Inflation_YoY"] = self.tick_df["Inflation_YoY"].ffill()
        self.tick_df["Inflation_MoM"] = self.tick_df["Inflation_MoM"].ffill()

        # Convert start and end dates to datetime
        self.start = pd.to_datetime(self.start)
        self.end = pd.to_datetime(self.end)

        # Adjust start and end based on available data
        if self.tick_df.index[0] > self.start:
            self.start = self.tick_df.index[0]

        if self.tick_df.index[-1] < self.end:
            self.end = self.tick_df.index[-1]

        # Count rows within the time slot
        count_rows = self.tick_df.loc[self.start:self.end].shape[0]

        if count_rows < 248:
            print("Not enough time for full momentum calculation!")
            return -1

        # Compute inflation-adjusted log returns
        self.tick_df["Inflation_DoD"] = (1 + self.tick_df["Inflation_MoM"] / 100) ** (1 / 21) - 1
        self.tick_df[f'Inflation_Adjusted_Log_Return_{self.tick}'] = np.log(1 + self.tick_df[f'Total_Return_{self.tick}']) 

        target_vola = 0.1  # Target volatility
        norm_re_list = []

        for type, month in momentum_diff_dict.items():
            # Get the last available date before self.end
            end_index = self.tick_df.index.get_loc(self.end)  # Finds position of end date in index

            # Find the closest available date for the lookback period
            lookback = self.end - pd.DateOffset(months=month)
            lookback_index = self.tick_df.index.get_loc(self.tick_df.index[self.tick_df.index.get_indexer([lookback], method="nearest")[0]])

            # Splice the dataframe using index positions
            temp_df = self.tick_df.iloc[lookback_index:end_index + 1]

            # Calculate cumulative momentum
            log_momentum = temp_df[f'Inflation_Adjusted_Log_Return_{self.tick}'].sum()
            compounded_momentum = (np.exp(log_momentum) - 1) * 100  # Convert back to percentage

            # Normalize return using volatility
            type_mod = type.replace("momentum", "")
            normalized_return = ((compounded_momentum / 100) / self.tick_df[f"Vol_{type_mod}"].iloc[-1]) * target_vola
            norm_re_list.append(normalized_return)

            # Store value in main dataframe
            main_df.loc[main_df['ticker'] == self.tick, type] = compounded_momentum

        # Store mean normalized return
        norm_mean_re = mean(norm_re_list)
        main_df.loc[main_df['ticker'] == self.tick, 'Norm_return'] = norm_mean_re

        # Debugging: Save CSV if it's GLD
        if self.tick == "GLD":
            save_df_csv(self.tick_df, self.tick)


def save_df_csv(df, tick):

    if os.path.exists(f"check_{tick}.csv"):
        try:
            os.remove(f"check_{tick}.csv")
        except PermissionError:
            print("the file is currently open, force shut down")
            os.system(f"taskkill /f /im excel.exe")
            time.sleep(0.5)
            os.remove(f"check_{tick}.csv")
        print(f"deleted prev version of check_{tick}.csv")

    df.to_csv(f"check_{tick}.csv")


def extract_csv(tick):
    df = pd.read_csv(f"check_{tick}")
    return df


def main():
    #list of SPX sectors including SPX itself
    global df_dict
    global cpi
    global momentum_diff_dict
    SP_INDEX = "^GSPC"
    sect_list = ["SPY", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    
    # list of bond ETFs, first one is baseline
    bond_list = ["AGG", "BKLN", "EMB", "LQD", "HYG", "MBB", "MUB", "TLT", "VCIT", "VTEB", "VCSH",\
                 "SGOV", "BSV", "IUSB", "IEF", "VGIT", "BIL", "BND", "GOVT", "SHY", "JAAA",\
                 "USHY", "TMF", "SPIB", "SHV"]

    # list of commodity ETFs
    com_list = ["DBC", "IBIT", "CPER", "CWB", "DBA", "DBB", "DBO", "GLD", "PALL", "PPLT", "SLV", \
                "U-U.TO", "UNG", "PDBC", "FTGC", "BCI"]

    # list of sectors on wiki
    # sect_list_name = ["Industrials", "Health Care", "Consumer Discretionary", "Information Technology", \
    #                   "Financials", "Consumer Staples", "Real Estate", "Communication Services", "Energy",
    #                   "Utilities", "Materials"]

    # the dataframe that stores morst of the cacluations for each of the ranked ETFs
    main_col = ['ticker', '1MOmomentum', '1MOmomentumScore', '3MOmomentum', '3MOmomentumScore', '6MOmomentum', \
                '6MOmomentumScore', '1YRmomentum', '1YRmomentumScore', 'HQM', 'Norm_return', 'vola']
    main_frame = pd.DataFrame(columns = main_col)

    start_ = "1991-01-01"
    end_ = str(date.today())
    #end_ = "2009-01-01"
    start_adj = pd.to_datetime(end_) - pd.DateOffset(years=2, days=5)

    # a list for the volatility of stocks for ONE sectzzor, later integrated into volatility_frame
    volatility_list = [] 
    volatility_baseline = 0.0
    volatility_baseline_bond = 0.0
    volatility_baseline_com = 0.0

    # read from FRED for CPI values for the US for the past year
    # the months = 14 could be a subject to change in the future since this is to buffer that this month
    # CPI hasn't come out yet and we need another month of values in order to do YtoY calculation
    # pd.to_datetime(end_) - pd.DateOffset(months=15)
    cpi = web.DataReader("CPIAUCSL", "fred", start_, end_)
    cpi["Inflation_YoY"] = (cpi["CPIAUCSL"].pct_change(12) * 100).fillna(0)  # Year-over-Year Inflation
    cpi["Inflation_MoM"] = (cpi["CPIAUCSL"].pct_change() * 100).fillna(0)   # MtoM inflation
    save_df_csv(cpi, "cpi")

    # elongate cpi to fill values on every single day, this is done to ensure successful merge
    # later on in calc_momentum where the inflation merge would lose values
    date_range = pd.date_range(start=pd.to_datetime(end_) - pd.DateOffset(months=14), end=end_, freq='D')
    date_range2 = pd.date_range(start=start_adj, end=end_, freq='D')
    temp_df = pd.DataFrame({"Date" : date_range})
    temp_df2 = pd.DataFrame({"Date" : date_range2})
    temp_df = temp_df.merge(cpi, how="left", left_on="Date", right_on="DATE")
    temp_df.ffill(inplace=True)
    temp_df = temp_df.merge(temp_df2, how="inner", on="Date")
    cpi = temp_df
    cpi = cpi.set_index("Date")
    # save_df_csv(cpi, "cpi2")

    """
    df = TickData(start_adj, end_, "XLU")
    df.scrape_tick()
    df.calc_vola()
    print(df.volatility)
    save_df_csv(df.tick_df, 'XLU')"""
    
    ETF_pool = sect_list + bond_list + com_list
    for i in ETF_pool:
        df = TickData(start_adj, end_, i)
        df.scrape_tick()
        df_dict[i] = df
    
    for i in ETF_pool:
        df_dict[i].calc_vola()
        if(i == "SPY"):
            volatility_baseline = df_dict[i].volatility
            print(df_dict[i].volatility)
        elif(i == 'AGG'):
            volatility_baseline_bond = df_dict[i].volatility
            print(df_dict[i].volatility)
        elif(i == 'DBC'):
            volatility_baseline_com = df_dict[i].volatility
            print(df_dict[i].volatility)

    for i in sect_list[1:]:
        #print(f"{i} vola: {df_dict[i].volatility}")
        if (df_dict[i].volatility <= float(1.1*volatility_baseline)):
            
            volatility_list.append((i, df_dict[i].volatility))
    
    for i in bond_list[1:]:
        if df_dict[i].volatility <= float(1.1*volatility_baseline_bond):
            volatility_list.append((i, df_dict[i].volatility))
    
    for i in com_list[1:]:
        if df_dict[i].volatility <= float(1.15*volatility_baseline_com):
            volatility_list.append((i, df_dict[i].volatility))

    print("lower vola: ")
    for i in volatility_list:
        print(i)
    
    # unzip the list of tuple pairs into 2 lists
    list_tick, list_vola = zip(*volatility_list)
    for i in range(len(list_tick)):
        new_row = pd.DataFrame({
            'ticker' : [list_tick[i]], # list of tickers
            'vola' : [list_vola[i]]     # list of their respective vola
        })
        main_frame = pd.concat([main_frame, new_row], ignore_index = True)
    
    for i in list_tick:

        # momentum calculation using data spanning from a year ago to today
        # senarios: inflation YoY, 6Mo6M, MoM, WoW, DoD ---
        #                                                 |
        #                                                 V
        #           normalized return with or without inflation
        # 10 different senarios to test out when backtesting
        df_dict[i].calc_momentum(main_frame)
 
    # calc momentum score, aka the percentile rank for every ticker in the 4 momentum catagories
    for type in momentum_diff_dict.keys():
        main_frame[f"{type}Score"] = main_frame[type].rank(pct=True) * 100
     
    # calc HQM score for each ticker using the mean of the 4 momentum catagories
    for row in main_frame.index:
        mean_list = []
        for month in momentum_diff_dict.keys():
            mean_list.append(main_frame.loc[row, f'{month}Score'])
        main_frame.loc[row, 'HQM'] = mean(mean_list)

    # sort and splice
    main_frame.sort_values('HQM', ascending=False, inplace=True)
    main_frame = main_frame[:8] # instead of this, maybe HQM >= 50 
    main_frame = main_frame.reset_index()

    # normalize HQM and volatility-adjusted return into the same scale with min-max tech
    # the min-max is modified with epslion to prevent the bounded 0 and 1 occuring
    epsilon = 0.02
    main_frame['norm_HQM'] = epsilon + (1 - 2*epsilon)*(main_frame["HQM"] - main_frame["HQM"].min())\
                            /(main_frame["HQM"].max() - main_frame["HQM"].min())
    main_frame['Norm_norm_return'] = epsilon + (1 - 2*epsilon)*(main_frame["Norm_return"] \
                                     - main_frame["Norm_return"].min())/(main_frame["Norm_return"].max()\
                                     - main_frame["Norm_return"].min())
    
    # weight comprised of both HQM and normalized return, with a slight bias towards HQM
    main_frame['get_weight'] = 0.55*main_frame["norm_HQM"] + 0.45*main_frame["Norm_norm_return"]

    #weighted average
    main_frame['true_weight'] = main_frame["get_weight"]/main_frame["get_weight"].sum()

    main_frame = main_frame.drop(columns=['norm_HQM', 'Norm_norm_return', 'get_weight',\
                                          '1MOmomentumScore', '3MOmomentumScore', '6MOmomentumScore',\
                                          '1YRmomentumScore'])

    # test o/p
    save_df_csv(main_frame, "main")
    """
    #the loop starts
    # idea: download the maximum/ start date till current timeframe of information
    # this could be in a dictionary or a hashmap, but a lot of the data structure would change
    # and separate it from the calc_vola function
    # - gets the rolling window on the volatility instead of constant
    # - ask money invested, use this and the weighted average to determine
    # how many shares of each needed to buy/sell
    # -each selection: extract the last month worth of data from each of the selected ticker dfs
    # using the index number to determine element position in df_dict
    # - accumilate shares*(close + dividends) for every selected ETF ticker daily
    # - get month worth of closing data from the calculation above, since its a chart, append it to 
    # the output chart
    # - the frist selection would only have 1 day of data, the subsequent would have a month worth

    # minimize calculation time by using as little functions as possible
# """
    #return 0

if __name__ == "__main__":
    main()