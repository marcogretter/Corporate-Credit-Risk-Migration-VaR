SystemError
"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 0: Discount Factors Bootstrap
"""

import numpy as np
import pandas as pd
import datetime as dt

from utilities.date_functions import (
    business_date_offset,
    year_frac_act_x,
    year_frac_30e_360
)
from typing import Iterable, Union, List, Union, Tuple

def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> List[float]:
    """
    Compute the zero rates from the discount factors.

    Parameters:
        dates (Union[List[float], pd.DatetimeIndex]): List of year fractions or dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        List[float]: List of zero rates.
    """

    effDates, effDf = dates, discount_factors
    # if the input are dates, it must be converted to year fractions, if it is already year fractions, nothing to do.  
    if isinstance(effDates, pd.DatetimeIndex): #the function checks if we gave 'calendar dates' and not year fractions
        # in the first case (where we gave calendar dates) it trasforms the dates in year fractions calculating
        # the difference between the ref date and all the other dates and it divides the result by 360 (Act/360).
        ref_date = effDates[0]
        effDates = np.array([(d - ref_date).days / 360.0 for d in effDates])
    
    effDates = effDates[1:] # we take every discount factor except the first since the first one is t=0
    effDf = discount_factors[1:]

    zero_rates = list(-np.log(effDf)/effDates)   # correct formula to convert a list of discounts into zero rates
    return zero_rates



def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    """
    Given a list of discount factors, return the discount factor at a given date by linear
    interpolation.

    REMARK: In our function, the discount factors are firstly transformed in zero rates, we interpolate the zero rates
    and in order to give back the requested discount factor we trasform the interpolated zero rate in discount factor.
    This is done because interpolating directly the discount factors may return "bad" interpolated discount factors.


    Parameters:
        reference_date (Union[dt.datetime, pd.Timestamp]): Reference date.
        interp_date (Union[dt.datetime, pd.Timestamp]): Date at which the discount factor is
            interpolated.
        dates (Union[List[dt.datetime], pd.DatetimeIndex]): List of dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        float: Discount factor at the interpolated date.
    """
    dfs = np.array(list(discount_factors), dtype=float)
    
    # if the data given are wrong, we stop immediately:
    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")
    
    # compute relevant yearfractions for available set of dates
    dates = pd.DatetimeIndex(dates)
    ref = pd.Timestamp(reference_date)
    # year fraction for the data for which we want to find the discount factor:
    t_star = year_frac_act_x(ref, pd.Timestamp(interp_date), 365)

    # compute year fractions for the other dates that we have:
    taus = np.array([year_frac_act_x(ref, d, 365) for d in dates], dtype=float)
    # convert discounts into zero rates
    z = np.zeros_like(dfs)
    for i in range(1, len(dfs)):  # we start from 1, we are not interested in t=0
        z[i] = -np.log(dfs[i]) / max(taus[i], 1e-12)
    # apply the interpolation on the target day:
    z_star = np.interp(t_star, taus, z)
    # convert zero rate into discount
    discount = float(np.exp(-z_star * t_star))
    return discount


def bootstrap(
    reference_date: dt.datetime,
    depo: pd.DataFrame,
    futures: pd.DataFrame,
    swaps: pd.DataFrame,
    shock: float = 0.0,
) -> pd.Series:
    """
    Bootstrap the discount factors from the given bid/ask market data. Deposit rates are used until
    the first future settlement date (included), futures rates are used until the 2y-swap settlement.

    Parameters:
        reference_date (dt.datetime): Reference date.
        depo (pd.DataFrame): Deposit rates.
        futures (pd.DataFrame): Futures rates.
        swaps (pd.DataFrame): Swaps rates.
        shock (Union[float, pd.Series]): Parallel shift to apply to the market rates, default to
            zero.

    Returns:
        pd.Series: Discount factors.
    """

    # initialize the list of dates and discounts
    termDates, discounts = [reference_date], [1.0]  # [1.0]= fattore di sconto a pronti, che è 1 perché oggi 1€ vale 1

    #### DEPOS
    
    # select the correct depos and their rates
    # We use the depos until we have the first future, so we select the dates of depos under the first date available
    # for futures (as explained at the incipit of this function)
    depoDates = depo.index[depo.index<=futures.index[0]].to_list() 
    # We use the rates of the dates found in the previous line of code. In order to get one rate only we do the mean
    # between the bid and ask rates
    depoRates = depo.loc[depoDates].mean(axis=1).values 

    # needed for the bumped bootstrap: if shock is a float, shift all the mkt data by that number, otherwise for each pillar its value
    depoRates = depoRates + (shock if isinstance(shock, float) else shock[depoDates].values)

    # We calculate the year frac for each depo date wr to the reference date
    depo_year_fracs = np.array([year_frac_act_x(reference_date, d, 360) for d in depoDates])

    # We calculate the depo discounts (see basicIR slides)
    depo_discounts = 1/(1+depoRates*depo_year_fracs)


    # we update the lists (that previously had only the initial values)
    termDates += depoDates
    discounts += list(depo_discounts)

    
    #### FUTURES

    # select the correct futures and their rates
    
    # As in the repos, we select the dates of the futures in which we are interested in.
    # We use the futures until the 2y swap. In our case the rule is to use the first 7 futures:
    futures_of_interest = futures.iloc[:7].copy()

    # We make the mean between the bid and ask prices for futures
    futures_Prices = futures_of_interest[['BID', 'ASK']].mean(axis=1).values
    futures_Prices = futures_Prices + (shock if isinstance(shock, float) else shock[futures_of_interest.index].values)

    # We convert the prices above in forward rates following the convention for futures:
    futures_Rates = (100 - futures_Prices) / 100
    
    i = 0

    for t, rowFut in futures_of_interest.iterrows(): 
        
        # for each futures, we take its dates
        t_start = rowFut['Settle']
        t_end = rowFut['Expiry']
        
        # We calculate the year frac between the two dates given above (ACT/360)
        tau = year_frac_act_x(t_start, t_end, 360)
        
        # We interpolate in order to find the real discount factor at the start of the futures:
        df_start = get_discount_factor_by_zero_rates_linear_interp(
            reference_date, t_start, termDates, discounts
        )
        
        # We calculate the real df at the end of the futures 
        df_end = df_start / (1 + futures_Rates[i] * tau)
        
        # Update the lists 
        termDates += [t_end]
        discounts += [df_end]  
        
        i += 1
   
    #### SWAPS

    # mean rates
    swapRates = (swaps[['BID', 'ASK']].mean(axis=1).values / 100) + (
       shock if isinstance(shock, float) else shock[swaps.index].values)

    # Initialize BPV
    swap_old = reference_date
    BPV = 0

    # We only use the first swap, whose period is covered by the futures, to calculate the BPV:
    date_1y = swaps.index[0]
    yf_1y = year_frac_30e_360(swap_old, date_1y)

    df_1y = get_discount_factor_by_zero_rates_linear_interp(reference_date, date_1y, termDates, discounts)

    BPV += df_1y * yf_1y
    swap_old = date_1y

    for idx, swapDate in enumerate(swaps.index[1:], start=1):
        rate = swapRates[idx]
        yf = year_frac_30e_360(swap_old, swapDate)
        
        # We calculate the DF using the BPV calculated previously
        df = (1.0 - rate * BPV) / (1.0 + rate * yf)
        
        # Update the terms 
        termDates.append(swapDate)
        discounts.append(df)
        BPV += df * yf
        swap_old = swapDate

    # Final results
    discount_factors = pd.Series(index=termDates, data=discounts).sort_index()
    zero = from_discount_factors_to_zero_rates(discount_factors.index, discount_factors.values)
    zero_rates = pd.Series(index=termDates[1:], data=zero)
    
    return discount_factors, zero_rates
