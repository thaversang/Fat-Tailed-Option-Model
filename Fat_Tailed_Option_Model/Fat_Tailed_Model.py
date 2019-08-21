#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 19:17:33 2019

@author: tom haversang
"""
from Robinhood_Data_Datafetcher import Robinhood_Data
import pandas as pd
import numpy as np
from scipy.optimize import minimize
call_sample= pd.read_pickle('AAPL_Calls.pickle')
put_sample = pd.read_pickle('AAPL_Puts.pickle')
#call + put are from EOD August 2, 2019
spot_sample = 204.00
class Fat_Tailed_Option_Model:
    def __init__(self, symbol,call_option_dataframe,put_option_dataframe,evaluation_date,cutoff_thresh = 0.15):
        self.calls = call_option_dataframe
        self.puts = put_option_dataframe
        self._cutoff_thresh = cutoff_thresh
        self.eval_date = pd.to_datetime(evaluation_date)
        self.__transform_dataframes()
        self._get_otm_options()
        self._get_deep_otm_options()

    def _get_days_to_expiry(self,times):
        """returns the time to expiry in years"""
        return ((times - self.eval_date).values).astype(np.float)/((10**9*86400))/365

    def _get_otm_options(self):
        """fetches the out of the money options"""
        self.otm_puts = self.puts.loc[self.puts.tail_metric<0]
        self.otm_calls = self.calls.loc[self.calls.tail_metric<0]

    def _get_deep_otm_options(self):
        "fetches the deep out of the money puts and calls"
        self.deep_otm_puts = self.otm_puts.loc[self.otm_puts.tail_metric < self.otm_puts.tail_metric.quantile(self._cutoff_thresh)]
        self.deep_otm_calls = self.otm_calls.loc[self.otm_calls.tail_metric<self.otm_calls.tail_metric.quantile(self._cutoff_thresh)]
    def __apply_liquidity_filter(self):
        """this filter simply looks where someone is willing to sell.
        We can add more filters, like where there are bidders, and there is open interest.

        """
        return  self.calls.loc[self.calls.ask_size>0].copy(), self.puts.loc[self.puts.ask_size>0].copy()


    def __transform_dataframes(self):
        """make some changes....


        """
        self.calls,self.puts = self.__apply_liquidity_filter()
        self.calls.expiration_date = pd.to_datetime(self.calls.expiration_date)
        self.puts.expiration_date = pd.to_datetime(self.puts.expiration_date)
        self.calls['time_to_expiry'] = self._get_days_to_expiry(self.calls.expiration_date)
        self.puts['time_to_expiry'] = self._get_days_to_expiry(self.puts.expiration_date)
        self.calls['tail_metric'] = (self.calls.spot_price - self.calls.strike_price)/(np.sqrt(self.calls.time_to_expiry))
        self.calls = self.calls.sort_values(by='tail_metric')
        self.puts['tail_metric'] = -((self.puts.spot_price - self.puts.strike_price)/(np.sqrt(self.puts.time_to_expiry)))
        self.puts = self.puts.sort_values(by='tail_metric')
        column_names = ['strike_price','expiration_date','open_interest','last_trade_price','adjusted_mark_price','ask_price','ask_size','bid_price','bid_size',
                       'break_even_price','high_fill_rate_buy_price','high_fill_rate_sell_price',
                       'implied_volatility','spot_price','tail_metric','time_to_expiry']
        self.calls = self.calls.loc[:,column_names]
        self.puts = self.puts.loc[:,column_names]

    def _model_L_put(self,strike,alpha,market_price):
        M = ((1-(strike/spot))**(-alpha))/(1-alpha)


        t1 = spot/(1-alpha)
        t2 = spot*M
        t3 = strike*M
        t4 = strike*alpha/(1-alpha)
        t5 = strike/(1-alpha)

        num = t1 - t2 + t3 + t4 - t5
        denom = market_price
        return (num/denom)**(-1/alpha)
    def _pareto_call_price(self,spot,K,alpha,L):

        t1 = K-spot
        t2 = L**alpha
        t3 = (K/spot - 1)**(-alpha)

        num = t1*t2*t3
        denom = (alpha - 1)
        return num/denom

    def _model_L_call(self,spot,strike,alpha,market_price):
        M = -(((strike/spot- 1)**(-alpha)))/(1-alpha)


        ans = ((M/market_price)*((strike-spot)))**(-1/alpha)
        return np.real(ans)

    def _pareto_put_price(self,spot,K,alpha,L):
        strike = K
        num = -(L**alpha)*((spot - strike + (strike-spot)*((1-strike/spot)**-alpha)+strike*alpha))
        denom = alpha - 1

        return num/denom

    def _apply_model_to_all_put_option(self,alpha_input,L0):
        p = self.deep_otm_puts.copy()
        model_put_prices = p.apply(lambda col: self._pareto_put_price(spot = col['spot_price'],K = col['strike_price'],alpha = alpha_input,L = L0*np.sqrt(col['time_to_expiry'])),axis=1)

        return model_put_prices.values

    def _apply_model_to_all_call_option(self,alpha_input,L0):

        c = self.deep_otm_calls.copy()

        model_call_prices = c.apply(lambda col: self._pareto_call_price(spot = col['spot_price'],K = col['strike_price'],alpha = alpha_input,L = L0*np.sqrt(col['time_to_expiry'])),axis=1)
        return model_call_prices

    def _call_prices_error_function(self,inputs,price_column_name,weights = 'open_interest'):
        if weights == 'ones':
            weigths = 1.000
        else:
            weigths = self.deep_otm_calls[weights]
        alpha_input = inputs[0]
        L0_input = inputs[1]
        model_put_price = self._apply_model_to_all_call_option(alpha_input = alpha_input,L0 = L0_input)
        errors = ((model_put_price - self.deep_otm_calls[price_column_name])**2)*self.deep_otm_calls.open_interest
        return np.sum(errors)

    def fit_calls(self,price_column_name = 'last_trade_price',weights = 'open_interest'):
        func = lambda x: self._call_prices_error_function(x,price_column_name = price_column_name,weights = weights)
        alpha_call_fit,L0_call_fit =  minimize(func,[2,0.1],method ='Nelder-Mead').x
        self.deep_otm_calls['model_price'] = self._apply_model_to_all_call_option(alpha_call_fit,L0_call_fit)
        self.call_alpha = alpha_call_fit
        self.call_L0 = L0_call_fit


    def _put_prices_error_function(self,inputs,price_column_name,weights = 'ones'):
        if weights == 'ones':
            weigths = 1.000
        else:
            weights = self.deep_otm_puts[weights]
        alpha_input = inputs[0]
        L0_input = inputs[1]
        model_put_price = self._apply_model_to_all_put_option(alpha_input = alpha_input,L0 = L0_input)
        errors = ((model_put_price - self.deep_otm_puts[price_column_name])**2)*self.deep_otm_puts.open_interest
        return np.sum(errors)

    def fit_puts(self,price_column_name ='last_trade_price',weights = 'ones'):
        func = lambda x: self._put_prices_error_function(x,price_column_name = price_column_name,weights = weights)
        alpha_put_fit,L0_put_fit =  minimize(func,[2,0.1],method ='Nelder-Mead').x
        fit_prices = self._apply_model_to_all_put_option(alpha_put_fit,L0_put_fit)
        self.deep_otm_puts['model_price'] = fit_prices
        self.put_alpha = alpha_put_fit
        self.put_L0 = L0_put_fit

        #return fit_prices
