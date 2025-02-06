import yfinance as yf
#import bloomberg if we can use API
import pandas as pd
import numpy as np
import time
import os

# Define the stock ticker symbol (e.g., "AAPL" for Apple)

#list of SPX sectors including SPX itself
sect_list = ["^GSPC", "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLK", "XLB", "XLRE", "XLU"]
# list of sectors on wiki
sect_list_real = ["Industrials", "Health Care", "Consumer Discretionary", "Information Technology", "Financials", "Consumer Staples", "Real Estate", "Communication Services", "Energy", "Utilities", "Materials"]

# Fetch sector data
stock = yf.Tickers(sect_list)

# Get historical market data from '91-'25
hist = stock.history(period="1mo")
output = stock.download(start="1991-01-01", end="2025-01-29", group_by="ticker")

test_frame = pd.DataFrame()
based = pd.DataFrame()
dividends_df = pd.DataFrame()
volatility_frame = pd.DataFrame()

# a list for the volatility of stocks for ONE sector, later integrated into volatility_frame
volatility_list = []

# output the data into an excel sheet
#output.to_csv("testop.csv")

# Get the list of S&P 500 companies and their sectors from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
table = pd.read_html(url)[0]  # First table contains stock tickers

# looping through all of the sectors and finding each company within the sector
for i in sect_list_real:
    # Filter for a specific sector (e.g., "Information Technology")
    sect_stocks = table[table["GICS Sector"] == i]["Symbol"].tolist()
    # append a column in the dataframe containing all companies within current sector
    test_frame[i] = pd.Series(sect_stocks) # the series is to make sure that the difference in row levels would not fuck up alignment

    # print(sect_stocks[:10])  # Display first 10 stocks in the sector
    # print("next")

# output the companies in each sector
# test_frame.to_csv("test.csv")

# get SPY from yfinance 91-25
spy = yf.Ticker('SPY')
based = yf.download('SPY', start="1991-01-01", end="2025-01-01")
if based.empty:
    print(f"No data for {i}, skipping...")

# get the dividends of SPY
div_SPY = spy.dividends

start_date = "1991-01-01"
end_date = "2025-01-01"

# extract the dividends of 19-25
dividends_filtered = div_SPY.loc[start_date:end_date]
# total_dividends = dividends_filtered.sum()

# Convert dividends to DataFrame
dividends_df = dividends_filtered.to_frame().reset_index()
dividends_df.columns = ["Date", "Dividends"] # set the columns

# Merge with stock price data
based = based.reset_index()

# print(based.index)     debugg statements, might need if more errors
# print(based.column)
# print(dividends_df.columns)

# the SPY df defaults to a bunch of multi-indexes, to merge later with dividend df, have to convert 
# the multi-indexes into normal ones, we go through all columns and join the pairs with _(chatGPT)
based.columns = ['_'.join(col).strip() if isinstance(col, tuple)\
                else col for col in based.columns]
# reset the Date_ column after conjointing, there is a pair column for some reason that looks like ('Date', '')
based.rename(columns={"Date_": "Date"}, inplace=True)

# reset index again
based = based.reset_index()
# print(based.columns) debug statement

# reset the date type of dividends as the SPY date is not locolized but dividends df are centralized towards New York
dividends_df["Date"] = dividends_df["Date"].dt.tz_localize(None)

# merge 2 dfs with Date column as the merge key and add matching dividends rows, if none dividend = NaN(left join)
based = based.merge(dividends_df, on="Date", how="left")

# Fill missing dividend values with 0
based["Dividends"].fillna(0, inplace=True)

# use the total return formula on every single row: (Pend + D - Pstart)/Pstart
ticker_str = 'SPY'
# first row no value, replace with 0
based["Total_Return_SPY"] = [((based[f"Close_{ticker_str}"][i] + based["Dividends"][i] - \
                            based[f"Close_{ticker_str}"][i - 1])/based[f"Close_{ticker_str}"][i - 1]) \
                            if i != 0 else 0 for i in range(len(based[f"Close_{ticker_str}"]))]

# use the volatility formula to get vola
volatility_baseline = np.std(based["Total_Return_SPY"].dropna()) * np.sqrt(252)

if os.path.exists("check@@.csv"):
    os.remove("check@@.csv")
    print("deleted prev version of check@@.csv")

based.to_csv("check@@.csv")
print("SPY volatility: ", volatility_baseline)

#find out how to get the specific stocks inside of each sector  

#create a different dataframe that store the voliatility of each stock(sector if applicable)
#loop thru sect_list and get the .pct_change() of each(and/or the individaul stocks), 
# and generate them into a new column in the new dataframe
# do .std() * sprt(252) to get hte volatility for the individal stocks, and then put the into a list
# try to importt the list into a new column in the dataframe 
#import to csv

# getting the sector volatility insetead of individual stocks
for i in range(1, len(sect_list)):
    sector = yf.Ticker(sect_list[i])
    ticker_div = sector.dividends
    

        
    data = yf.download(sect_list[i], start="2012-01-01", end="2025-01-01", progress=False)

        
    # make a new column in the dataframe called returns to store this value, if made then append
    data["Returns"] = data["Close"].pct_change() # return without dividends, return = rate of change difference
    volatility = np.std(data["Returns"].dropna()) * np.sqrt(252) # get the volotatily via normal distribution * sqrt of 252
    if (volatility <= volatility_baseline):
        volatility_list.append((sect_list[i], volatility))
    
    time.sleep(0.1)

for i in volatility_list:
    print(i)

# get the data for dividends for the alotted time periods first for SPY, and then every sector ETF
# check out the excel imported dataframe of SPY and determine the method of calculation for total returns
# copy and paste the code from before and change the method for calulating the total return of the ETFs

# Print historical data
# print(hist.head())