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

df_list = []
cpi = pd.DataFrame(columns=['CPIAUCSL', 'Inflation_YoY', 'Inflation_MoM'])
# dict containing the different lookback periods for momentum and normalized return calulation
momentum_diff_dict = {
    "1MOmomentum" : 1,
    "3MOmomentum" : 3,
    "6MOmomentum" : 6,
    "1YRmomentum" : 12
    }


def get_money_supply(end_):

    M1 = web.DataReader("WM1NS", "fred", pd.to_datetime(end_) - pd.Timedelta(days=32), end_)
    #print(M1["WM1NS"].iloc[-1])
    return M1["WM1NS"].iloc[-1]


def calc_vola(tick, start_date, end_date, periods = {'1MO' : 20, '3MO' : 60, '6MO' : 120, '1YR' : 252}):
    # get ticker info from current selected time to a year from then
    ticker = yf.Ticker(tick)
    ticker_df = yf.download(tick, start_date, end=end_date)
    if ticker_df.index.empty:
        print(f"No data for {ticker}, skipping...")
        return pd.DataFrame(), 100

    # get the dividends of the ticker
    div_ = ticker.dividends

    #remove timezone from it being new york
    try:
        div_.index = div_.index.tz_localize(None)
    # this error doesn't occur unless there is nothing in the date column, AKA failed to retrieve data
    except AttributeError:
        print("failed to retrieve info! possibly due to ticker name or yf timing out")
        return pd.DataFrame(), 100

    # extract the dividends of specificed start to end date
    dividends_filtered = div_.loc[start_date:end_date]
    # total_dividends = dividends_filtered.sum()

    # Convert dividends to DataFrame
    dividends_df = dividends_filtered.to_frame().reset_index()
    dividends_df.columns = ["Date", "Dividends"] # set the columns

    # Merge with stock price data
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
    ticker_df[f"Total_Return_{tick}"] = [((ticker_df[f"Close_{tick}"][i] + ticker_df["Dividends"][i] - \
                                            ticker_df[f"Close_{tick}"][i - 1])/ticker_df[f"Close_{tick}"][i - 1]) \
                                            if i != 0 else 0 for i in range(len(ticker_df[f"Close_{tick}"]))]

    # use the volatility formula to get vola
    for name, i in periods.items():
        actual_day = min(i, len(ticker_df))
        ticker_df[f"Vol_{name}"] = (
                # important, the formula for volatility calculation is:
                # std(return over selected period) X root(SELECTED PERIOD)
                # and to annualize it for consistency, multiply again with 
                # root(252/selected period)
                # could also consider rolling().std() if needed, instead of all same value as most recent value
                np.std(ticker_df[f"Total_Return_{tick}"].iloc[-actual_day:])\
                *np.sqrt(actual_day)*np.sqrt(252/actual_day)
                )
    
    # standard annualized vola calc, same as the 1 yr vola
    volatility = np.std(ticker_df[f"Total_Return_{tick}"].dropna()) * np.sqrt(252)
    
    return ticker_df, volatility

# first selection of competitive tickers from their vola benchmarks
def vola_filter(ETF_list, vola_list, baseline, start, end, filter_val):
    temp_df = pd.DataFrame()
    for i in range(1, len(ETF_list)):
        #try:
        temp_df, volatility = calc_vola(ETF_list[i], start, end)
        #except TypeError: # ticker dataframe is empty
            #print("no data present, too early/late or wrong ticker name")
            #continue
        
        # qualifiable tickers in 1st selection
        if (volatility <= (baseline + filter_val)):
            vola_list.append((ETF_list[i], volatility))
            df_list.append(temp_df)

        # to prevent the unlikely event of yahoo finance timeouts due to overly fast access
        time.sleep(0.2)


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


# this function not only calcs the momentum but also the normalized return, subject to change
def calc_momemtum(ticker, tick_df, main_df, start_date, end_date):
    global cpi
    global momentum_diff_dict
    # shouldn't need this
    # tick_df[f"Total_Return_{ticker}"] = tick_df[f"Total_Return_{ticker}"].replace(0, np.nan).ffill()

    # Merge CPI data (resample to match ETF data frequency)
    tick_df = tick_df.merge(cpi[["Inflation_YoY"]], left_on="Date", right_index=True, how="left")
    tick_df = tick_df.merge(cpi[["Inflation_MoM"]], left_on="Date", right_index=True, how="left")

    # Fill missing values (e.g., weekends/holidays)
    tick_df["Inflation_YoY"] = tick_df["Inflation_YoY"].ffill()
    tick_df["Inflation_MoM"] = tick_df["Inflation_MoM"].ffill()
    
    # lock to earliest available start date if start timeframe does not exist
    start_date = pd.to_datetime(start_date)
    if (tick_df["Date"].iloc[0] > start_date):
        start_date = tick_df["Date"].iloc[0]
    
    # lock to latest available end date if end timeframe does not exist
    end_date = pd.to_datetime(end_date)
    if (tick_df["Date"].iloc[-1] < end_date):
        end_date = tick_df["Date"].iloc[-1]

    # bitwise and all rows within the timeslot
    count_rows = tick_df[(tick_df['Date'] >= start_date) & (tick_df['Date'] <= end_date)].shape[0]

    if(count_rows < 248):
        print("not enough time for full momentum calculation!")
        return -1
    
    # Compute log returns to ensure compounding and adjust for inflation
    tick_df["Inflation_DoD"] = (1 + tick_df["Inflation_MoM"] / 100) ** (1 / 21) - 1
    tick_df[f'Inflation_Adjusted_Log_Return_{ticker}'] = np.log(1 + tick_df[f'Total_Return_{ticker}']) - \
                                                         np.log(1 + tick_df["Inflation_DoD"]/100)
    
    # variables for normalized return
    target_vola = 0.1 # subject to change
    norm_re_list = []
    
    # loop through key value pair in dict
    for type, month in momentum_diff_dict.items():

        end_index = tick_df[tick_df["Date"] <= end_date].index[-1]  # Gets the last available row before end_date

        # use timedelta to locate the available start date 
        lookback = end_date - pd.DateOffset(months=month)
        # if the current start date is not present, go back by one row and lock the next available date
        lookback_index = (tick_df["Date"] - lookback).abs().idxmin() # Gets the first available row after lookback
        #lookback_index = tick_df[tick_df["Date"] <= lookback].index[-1]

        # splice
        temp_df = tick_df.iloc[lookback_index:end_index + 1]
        
        # calculate the respective cumulative momentum using the cumulative momentum formula
        log_momentum = temp_df[f'Inflation_Adjusted_Log_Return_{ticker}'].sum()
        compounded_momentum = (np.exp(log_momentum) - 1)*100  # Convert back to percentage

        #calculates the normalized return
        type_mod = type.replace("momentum", "")
        # volaility nromalized return formula, divide momentum cuz its not percentage yet
        normalized_return = ((compounded_momentum/100)/tick_df[f"Vol_{type_mod}"].iloc[-1])*target_vola
        norm_re_list.append(normalized_return)

        # load value in main dataframe
        main_df.loc[main_df['ticker'] == ticker, type] = compounded_momentum

    # mean of different periods of normalized return
    norm_mean_re = mean(norm_re_list)
    main_df.loc[main_df['ticker'] == ticker, 'Norm_return'] = norm_mean_re

    if ticker == "GLD":
        save_df_csv(tick_df, ticker)


def main():
    #list of SPX sectors including SPX itself
    global df_list
    global cpi
    global momentum_diff_dict
    SP_INDEX = "^GSPC"
    sect_list = ["SPY", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    
    # list of bond ETFs, first one is baseline
    bond_list = ["AGG", "BKLN", "EMB", "LQD", "HYG", "MBB", "MUB", "TLT", "VCIT", "VTEB", "VCSH",\
                 "SGOV", "BSV", "IUSB", "IEF", "VGIT", "BIL", "BND", "JPST", "GOVT", "SHY", "JAAA",\
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
    end_ = "2009-01-01"
    start_adj = pd.to_datetime(end_) - pd.DateOffset(years=1)

    # a list for the volatility of stocks for ONE sectzzor, later integrated into volatility_frame
    volatility_list = [] 
    volatility_baseline = 0.0
    volatility_baseline_bond = 0.0
    volatility_baseline_com = 0.0
    m1 = get_money_supply(end_)

    #"""
    # read from FRED for CPI values for the US for the past year
    # the months = 14 could be a subject to change in the future since this is to buffer that this month
    # CPI hasn't come out yet and we need another month of values in order to do YtoY calculation
    cpi = web.DataReader("CPIAUCSL", "fred", pd.to_datetime(end_) - pd.DateOffset(months=15), end_)
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

    
    # calc SP volatility
    trash, volatility_baseline = calc_vola(sect_list[0], start_adj, end_)
    print("SPY volatility: ", volatility_baseline)
    # select attractive ETFS from SP sectors
    vola_filter(sect_list, volatility_list, volatility_baseline, start_adj, end_, 0.025)

    # calc bond vola
    trash, volatility_baseline_bond = calc_vola(bond_list[0], start_adj, end_)
    print(f"AGG volatiltiy: {volatility_baseline_bond}")
    # select bond ETFs
    vola_filter(bond_list, volatility_list, volatility_baseline_bond, start_adj, end_, 0.0175)

    # commodity ETFs
    trash, volatility_baseline_com = calc_vola(com_list[0], start_adj, end_)
    print(f"DBC volatiltiy: {volatility_baseline_com}")
    del trash
    vola_filter(com_list, volatility_list, volatility_baseline_com, start_adj, end_, 0.025)

    print("lower vola: ")
    for i in volatility_list:
        print(i)

    # unzip the list of tuple pairs into 2 lists
    list_tick, list_vola = zip(*volatility_list)
    for i in range(len(list_tick)):
        new_row = pd.DataFrame({
            'ticker' : [list_tick[i]],
            'vola' : [list_vola[i]]
        })
        main_frame = pd.concat([main_frame, new_row], ignore_index = True)

    for i in range(len(list_tick)):

        # momentum calculation using data spanning from a year ago to today
        # senarios: inflation YoY, 6Mo6M, MoM, WoW, DoD ---
        #                                                 |
        #                                                 V
        #           normalized return with or without inflation
        # 10 different senarios to test out when backtesting
        calc_momemtum(list_tick[i], df_list[i], main_frame, start_adj, end_)
 
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
    main_frame = main_frame[:8]
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

    main_frame = main_frame.drop(columns=[\
                                          '1MOmomentumScore', '3MOmomentumScore', '6MOmomentumScore',\
                                          '1YRmomentumScore'])

    # test o/p
    save_df_csv(main_frame, "main")

    return 0
# """
if __name__ == "__main__":
    main()