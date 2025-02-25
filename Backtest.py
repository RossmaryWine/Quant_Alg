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

def calc_vola(tick, start_date, end_date):
    # get SPY from yfinance 91-25
    ticker = yf.Ticker(tick)
    default_df = yf.download(tick, start_date, end=end_date)
    if default_df.empty:
        print(f"No data for {ticker}, skipping...")
        return 100

    # get the dividends of the ticker
    div_ = ticker.dividends

    #remove timezone from it being new york
    try:
        div_.index = div_.index.tz_localize(None)
    # this error doesn't occur unless there is nothing in the date column, AKA failed to retrieve data
    except AttributeError:
        print("failed to retrieve info! possibly due to ticker name or yf timing out")
        return 100

    # extract the dividends of specificed start to end date
    dividends_filtered = div_.loc[start_date:end_date]
    # total_dividends = dividends_filtered.sum()

    # Convert dividends to DataFrame
    dividends_df = dividends_filtered.to_frame().reset_index()
    dividends_df.columns = ["Date", "Dividends"] # set the columns

    # Merge with stock price data
    default_df = default_df.reset_index()

    # the ticker df defaults to a bunch of multi-indexes, to merge later with dividend df, have to convert 
    # the multi-indexes into normal ones, we go through all columns and join the pairs with _(chatGPT)
    default_df.columns = ['_'.join(col).strip() if isinstance(col, tuple)\
                        else col for col in default_df.columns]
    # reset the Date_ column after conjointing, there is a pair column for some reason that looks like ('Date', '')
    default_df.rename(columns={"Date_": "Date"}, inplace=True)

    # reset index again
    default_df = default_df.reset_index()
    # print(based.columns) debug statement

    # merge 2 dfs with Date column as the merge key and add matching dividends rows, if none dividend = NaN(left join)
    default_df = default_df.merge(dividends_df, on="Date", how="left")

    # Fill missing dividend values with 0
    default_df.fillna({"Dividends" : 0}, inplace=True)

    # use the total return formula on every single row: (Pend + D - Pstart)/Pstart
    # first row no value, replace with 0
    default_df[f"Total_Return_{tick}"] = [((default_df[f"Close_{tick}"][i] + default_df["Dividends"][i] - \
                                            default_df[f"Close_{tick}"][i - 1])/default_df[f"Close_{tick}"][i - 1]) \
                                            if i != 0 else 0 for i in range(len(default_df[f"Close_{tick}"]))]

    # use the volatility formula to get vola
    volatility = np.std(default_df[f"Total_Return_{tick}"].dropna()) * np.sqrt(252)
    
    return default_df, volatility


def vola_filter(ETF_list, vola_list, baseline, start, end, filter_val):
    temp_df = pd.DataFrame()
    for i in range(1, len(ETF_list)):

        temp_df, volatility = calc_vola(ETF_list[i], start, end)

        # print(f"{bond_list[i]}: {volatility}")
        if (volatility <= (baseline + filter_val)):
            # nomralize retrurn here
            vola_list.append((ETF_list[i], volatility))
            df_list.append(temp_df)

        # to prevent the unlikely event that yahoo finance shuts down overly fast access
        time.sleep(0.2)

def normalized_return(tick_df, main_df, vola, start_date, end_date):
    global momentum_diff_dict
    mean_list = []
    # for time in momentum_diff_dict.values():




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
    
    # loop through key value pair in dict
    for type, month in momentum_diff_dict.items():

        end_index = tick_df[tick_df["Date"] <= end_date].index[-1]  # Gets the last available row before end_date

        # use timedelta to locate the available start date 
        lookback = end_date - pd.DateOffset(months=month)
        # if the current start date is not present, go back by one row and lock the next available date
        lookback_index = (tick_df["Date"] - lookback).abs().idxmin() # Gets the first available row after lookback

        # calculate the respective cumulative momentum using the cumulative momentum formula

        temp_df = tick_df.iloc[lookback_index:end_index + 1]
        
        log_momentum = temp_df[f'Inflation_Adjusted_Log_Return_{ticker}'].sum()
        compounded_momentum = (np.exp(log_momentum) - 1)*100  # Convert back to percentage

        # load value in main dataframe
        main_df.loc[main_df['ticker'] == ticker, type] = compounded_momentum

        if ticker == "BKLN":
            save_df_csv(tick_df, ticker)
    
    # this code does not work due to the fact that percentileofscore() requires every ticker to be populated before running
    # otherwise it does not work. the score calculation step and the follow up is forced into main()
    """
    for type, month in momemtum_diff_dict.items():
        main_df.loc[main_df['ticker'] == ticker, f"{type}Score"] = \
        stats.percentileofscore(main_df[type], main_df.loc[main_df['ticker'] == ticker, type].values[0])
        print(stats.percentileofscore(main_df[type], main_df.loc[main_df['ticker'] == ticker, type].values[0]))
    """


def main():
    #list of SPX sectors including SPX itself
    global df_list
    global cpi
    global momentum_diff_dict
    SP_INDEX = "^GSPC"
    sect_list = ["SPY", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
    
    # list of bond ETFs, first one is baseline
    bond_list = ["AGG", "BKLN", "EMB", "LQD", "HYG", "MBB", "MUB", "TLT", "VCIT", "VTEB", "VCSH",\
                 "SGOV", "BSV", "IUSB", "IEF", "VGIT", "BIL"]

    # list of commodity ETFs
    com_list = ["DBC", "IBIT", "CPER", "CWB", "DBA", "DBB", "DBO", "GLD", "PALL", "PPLT", "SLV", \
                "U-U.TO", "UNG", "PDBC", "FTGC", "BCI"]

    # list of sectors on wiki
    # sect_list_name = ["Industrials", "Health Care", "Consumer Discretionary", "Information Technology", \
    #                   "Financials", "Consumer Staples", "Real Estate", "Communication Services", "Energy",
    #                   "Utilities", "Materials"]

    # the dataframe that stores morst of the cacluations for each of the ranked ETFs
    main_col = ['ticker', '1MOmomentum', '1MOmomentumScore', '3MOmomentum', '3MOmomentumScore', '6MOmomentum', \
                '6MOmomentumScore', '1YRmomentum', '1YRmomentumScore', 'HQM', 'vola']
    main_frame = pd.DataFrame(columns = main_col)

    start_ = "1991-01-01"
    end_ = str(date.today())
    start_adj = pd.to_datetime(end_) - pd.DateOffset(years=1)

    # a list for the volatility of stocks for ONE sector, later integrated into volatility_frame
    volatility_list = [] 
    volatility_baseline = 0.0
    volatility_baseline_bond = 0.0
    volatility_baseline_com = 0.0

    # read from FRED for CPI values for the US for the past year
    # the months = 14 could be a subject to change in the future since this is to buffer that this month
    # CPI hasn't come out yet and we need another month of values in order to do YtoY calculation
    cpi = web.DataReader("CPIAUCSL", "fred", pd.to_datetime(end_) - pd.DateOffset(months=14), end_)
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
    # cpi["Date"] = pd.to_datetime(cpi["Date"])
    # save_df_csv(cpi, "cpi2")

    # """
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

    # weighted average with both volatility and momentum
    # custom calculation that slightly boosts the ave of etfs that show resilience to inflation
    #main_frame['Weighted_HQM'] = \
    #(main_frame['HQM'] * (1 + cpi["Inflation_MoM"].mean() / 100)) / main_frame['HQM'].sum()
    main_frame['Weighted_HQM'] = main_frame['HQM'] / main_frame['HQM'].sum()

    # weighted ave for vola
    main_frame['Invert_vola'] = 1 / main_frame['vola'] 
    main_frame["Weighted_invert_vola"] = main_frame['Invert_vola'] / main_frame['Invert_vola'].sum()

    #combine the weighted ave for momentum and vola to get the true final weight on each etf
    main_frame['recap'] = main_frame['Weighted_HQM']*main_frame['Weighted_invert_vola']
    main_frame['true_weight'] = main_frame['recap'] /main_frame['recap'].sum()

    main_frame = main_frame.drop(columns=['Weighted_HQM', 'Invert_vola', 'Weighted_invert_vola', 'recap'])

    # test o/p
    save_df_csv(main_frame, "main")

    return 0
# """
if __name__ == "__main__":
    main()