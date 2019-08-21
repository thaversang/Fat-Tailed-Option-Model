#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 14:42:45 2019

@author: tom haversang
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dask import delayed
from time import time
from time.to_datetime import today
from fast_arrow import Client, Stock, OptionChain, Option,OptionMarketdata,StockMarketdata
class Robinhood_Data:

    """This code uses fast-arrow, a library developed by
    Weston Platter. It's excellent, and it showcases the power of
    the Robinhood API. You can find the API here...

    https://github.com/westonplatter/fast_arrow.

    While the brokerage has its limits, it's great for real time Options Market
    Analyses."""

    def __init__(self,username,password):
        self.client = Client(username=username, password=password)
        self.client.authenticate() # this part can use some finagling.




    def get_spot_price(self,symbol):
        stock1= Stock.fetch(self.client, symbol)
        stock = StockMarketdata.quote_by_instrument(self.client,_id =stock1['id'])

        return stock



    def _process_spot_price(self,spot_price_object):
        return (float(spot_price_object['bid_price'])+ float(spot_price_object['ask_price']))/2

    def __convert_to_float(self,x):
        try:
            return np.float(x)
        except:
            return x

    def get_options_robinhood(self,symbol,**exp):
        """Get Robinhood Options Chains for Puts and Calls.
        The code returns three objects. This code returns two pandas dataframes
        The first is Calls, the Second is Puts. The final output is the spot price.


        """

        try:
            stock = Stock.fetch(self.client, symbol)
            stock_id = stock["id"]
            option_chain = OptionChain.fetch(self.client, stock_id,symbol = symbol)
            eds = option_chain['expiration_dates']

            oc_id = option_chain["id"]

            spot = self.get_spot_price(symbol)

            spot_price = self._process_spot_price(spot)
            if exp:
                if exp['exp'] not in eds:
                    print('Expiry not a Valid Expiration,Here are the valid Expirations \n')
                    print(eds)
                    return np.nan,np.nan,np.nan

                expiry = exp['exp']
                eds = [expiry]
            else:
                    print('Expiry not a Valid Expiration,Here are the valid Expirations \n')
                    print(eds)
                    return np.nan,np.nan,np.nan

            ops = Option.in_chain(self.client, oc_id, expiration_dates=eds)
            ops = Option.mergein_marketdata_list(self.client, ops)
            df = pd.DataFrame(ops)
            df.index = np.arange(0,len(df))

            #calls = df.loc[df.type=='call']
            #calls = calls.sort_index()
            #puts = df.loc[df.type=='put']
            #puts = puts.sort_index()
            df['spot_price'] = spot_price
            #puts['spot_price'] = spot_price
            df = df.applymap(self.__convert_to_float)
            df['expiration_date'] = pd.to_datetime(df['expiration_date'].values)
            #puts  = puts.applymap(self.__convert_to_float)


            return df.fillna(0)

        except:
            return pd.DataFrame()

    def get_all_options_robinhood(self,symbol):
        """Here, we use a library called 'DASK' to parallelize our code to fetch the data. We can be clever
        and fetch it all at once, to HOPEFULLY, speed up data retrieval for our larger datasets,
        like SPY and AMZN, to name a couple.
        from dask import delayed"""


        df_list = []
        stock = Stock.fetch(self.client, symbol)
        stock_id = stock["id"]
        expiries = OptionChain.fetch(self.client, stock_id,symbol = symbol)['expiration_dates']

        for expiration_date in expiries:
            #print(expiration_date)
            y =  delayed(self.get_options_robinhood)(symbol,exp = expiration_date)
            df_list.append(y)

        ans = delayed(pd.concat)(df_list)
        df = ans.compute()
        return df.loc[df.type =='call'], df.loc[df.type=='put'],df.spot_price.iloc[0]
