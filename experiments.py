import yfinance as yf
import matplotlib.pyplot as plt

# Download historical data for SPX and SPY
spx = yf.download("^GSPC", start="2023-01-01", end="2025-01-01")["Close"]
spy = yf.download("SPY", start="2023-01-01", end="2025-01-01")["Close"]

# Normalize to start at 100 for a fair comparison
spx_norm = (spx / spx.iloc[0])
spy_norm = (spy / spy.iloc[0])

# Plot
plt.figure(figsize=(10,5))
plt.plot(spx_norm, label="SPX (S&P 500 Index)", linestyle="dashed")
plt.plot(spy_norm, label="SPY (S&P 500 ETF)", linestyle="solid")
plt.legend()
plt.title("SPX vs SPY Performance")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.grid()
plt.show()

"""
# for all sectors
for j in sect_list_real:
    # format the ticker symbols for one sector to rid of NaN, spaces, make sure its in list and all string

    tickers = test_frame[j].dropna().astype(str).tolist()

    #loop through every ticker in the respective sector
    for i in tickers:

        # try maximum 3 times due to yfinance having a request restriction 
        for attempt in range(3):
            try:
                data = yf.download(i, start="2012-01-01", end="2025-01-01", progress=False)
                if data.empty:
                    print(f"No data for {i}, skipping...")
                else:
                    break  # Exit retry loop if download is successful
            except Exception as e:
                print(f"Error fetching {i}: {e}")
                time.sleep(2)  # Wait before retrying
        else:
            print(f"Failed to fetch {i} after 3 attempts.")
        
        # make a new column in the dataframe called returns to store this value, if made then append
        data["Returns"] = data["Close"].pct_change() # return without dividends, return = rate of change difference
        volatility = np.std(data["Returns"].dropna()) * np.sqrt(252) # get the volotatily via normal distribution * sqrt of 252
        if (volatility <= volatility_baseline):
            volatility_list.append((i, volatility))
        
        sector_df = pd.DataFrame(volatility_list, columns=["Stock", j]) 
        
    volatility_frame = pd.concat([volatility_frame, sector_df], axis=0, ignore_index=True)
    volatility_list.clear()

# volatility_frame.to_csv("valid_stock_for_vola.csv")
"""