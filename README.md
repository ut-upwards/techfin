# techfin : A package for technical indicators


## Indicators implemented in the package

1. Simple Moving Average
2. Exponential Moving Average
3. Bollinger Bands
4. Simple Relative Strength Index
5. Exponential Relative Strength Index
6. Moving Average Convergence Divergence
7. Money Flow Index

## Installation

- Make sure you have numpy, pandas, yfinance, matplotlib installed

- Run Following command to install the package
```
  pip install techfin
```

## How to import

- Run Following command to import
```
  import techfin

  # The following packages are necessary
  import pandas as pd
  import numpy as np
  import yfinance as yf
  import matplotlib.pylot as plt
```

## Example

- Run Following command to import
```
  data = yf.download('HDFC.NS', start=2021-01-01, end=2022-01-01, interval='1d', auto_adjust=True)
  tf.plot_Close(data) # to plot close price of data
  
  n = [10, 50, 100]
  data_SMA = techfin.SMA(data, n) # to calulate simple moving average
```


## Functions

| Sr. | Function                                 |Discription                                                 |
| --- | -----------------------------------------|-----------|
| 1.  | plot_Close (data)                        |Plot Close Price|
| 2.  | SMA (data, n)                            |Calculate Simple Moving Average|
| 3.  | plot_SMA (data, n)                       |Plot Simple Moving Average|
| 4.  | EMA (data, n)                            |Calculate Exponential Moving Average|
| 5.  | plot_EMA (data)                          |Plot Exponential Moving Average|
| 6.  | BollingerBands (data, n, factor)         |Calculate Bollinger Bands|
| 7.  | plot_BollingerBands (data)               |Plot Bollinger Bands|
| 8.  | Simple_RSI (data, n)                     |Calculate Simple RSI|
| 9.  | plot_S_RSI (data, overbought, oversold)  |Plot Simple RSI, highlighting overbought & oversold zones|
| 10. | Exponential_RSI (data, n)                |Calculate Exponential RSI|
| 11. | plot_E_RSI (data, overbought, oversold)  |Plot Exponential RSI, highlighting overbought & oversold zones|
| 12. | MACD (data, longEMA, shortEMA, signalEMA)|Calculate MACD indicator|
| 13. | plot_MACD (data)                         |Plot MACD indicator along with bars for Convergence/Divergence|
| 14. | Money_Flow_Index (data, n)               |Calculate Money Flow Index|
| 15. | plot_MFI (data, overbought, oversold) |Plot MFI, highlighting overbought & oversold zones|



## Screenshots
<p float="left">
  <img src="img/Close.jpg" width=350/>
  <img src="img/SMA.jpg" width=350/>
  <img src="img/EMA.jpg" width=350/>
  <img src="img/BB.jpg" width=350/>
</p>

<p float="right">
  <img src="img/MFI.jpg" width=350/>
  <img src="img/S_RSI.jpg" width=350/>
  <img src="img/E_RSI.jpg" width=350/>
  <img src="img/MACD.jpg" width=350/>
</p>
