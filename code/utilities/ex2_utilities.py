"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 2: Corporate Bond Portfolio
"""

from typing import List, Union
import numpy as np
import pandas as pd
import datetime as dt
from utilities.ex1_utilities import (
    year_frac_act_x,
    year_frac_30e_360,
    get_discount_factor_by_zero_rates_linear_interp,
)
from utilities.ex0_utilities import (               # Error: tried to import business_date_offset function from utilities.ex1
    business_date_offset                            #        but it was actually in ex0_utilities
)


def bond_payment_dates(
    issue_date: Union[dt.date, pd.Timestamp], maturity: int, coupon_freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Calculate the payment dates of a bond.

    Parameters:
        issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
        maturity (int): Bond's maturity in years.
        coupon_freq (int): Coupon frequency in payments per years.

    Returns:
        Union[List[dt.date], List[pd.Timestamp]]: List of payment dates.
    """

    payment_dates = []
    counter = 1
    for _ in range(maturity * coupon_freq):
        payment_dt = business_date_offset(
            issue_date, month_offset=(12 // coupon_freq) * counter
        )
        payment_dates.append(payment_dt)            # added line

        counter += 1

    return payment_dates




def bond_cash_flows(
    ref_date: Union[dt.date, pd.Timestamp],
    issue_date: Union[dt.date, pd.Timestamp],
    maturity: int,
    coupon_rate: float,
    coupon_freq: int,
    notional: float = 1.0,
) -> pd.Series:
    """
    Calculate the cash flows of a bond.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
    maturity (int): Bond's maturity in years.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments per years.
    notional (float): Notional amount.

    Returns:
        pd.Series: Bond cash flows.
    """

    # Payment dates
    cash_flows_dates = bond_payment_dates(issue_date, maturity, coupon_freq)

    # Coupon payments
    dates = [ref_date] + cash_flows_dates
    cash_flows = pd.Series(
        data=[
             notional*coupon_rate*year_frac_30e_360(dates[i-1], dates[i])
             for i in range(1, len(dates))
        ],
        index=cash_flows_dates,
    )   
    # Notional payment
    cash_flows[cash_flows_dates[-1]] += notional

    return cash_flows


def defaultable_bond_dirty_price_from_intensity(
    ref_date: Union[dt.date, pd.Timestamp],
    issue_date: Union[dt.date, pd.Timestamp],
    maturity: int,
    coupon_rate: float,
    coupon_freq: int,
    recovery_rate: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond neglecting the recovery of the coupon payments.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
    maturity (int): Bond's maturity in years.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    recovery_rate (float): Recovery rate.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # Calculate the cash flows
    cash_flows = bond_cash_flows(
        ref_date, issue_date, maturity, coupon_rate, coupon_freq, notional
    )

    # Discount factors, I get only the DFs that I am interested in (the ones of the coupon dates and the maturity)
    cf_dates = pd.to_datetime(cash_flows.index)
    discount_factors.index = pd.to_datetime(discount_factors.index)
    interpolated_dfs = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, d, discount_factors.index, discount_factors) 
        for d in cf_dates
    ]
    discount_factors = pd.Series(data=interpolated_dfs, index=cash_flows.index)

    # Calculate the survival probabilities and default probabilities
    if isinstance(intensity, float):                 # if intensity is a float -> constant value (constant lambda on the whole period)
        survival_probs = np.exp(
            [
                -intensity * year_frac_act_x(ref_date, date, 365)
                for date in cash_flows.index
            ]
        )
        survival_probs = pd.Series(data=survival_probs, index=cash_flows.index)
    else:                                            # if instensity is not a float -> pd.Series -> piecewise constant
        dates = [ref_date] + list(cash_flows.index)
        dts = [year_frac_act_x(dates[i-1], dates[i], 365) for i in range(1, len(dates))]
        lambdas = intensity.reindex(cash_flows.index, method='bfill')
        exponent = (lambdas*dts).cumsum()
        survival_probs = np.exp(-exponent)
    
    default_probs = survival_probs.shift(1).fillna(1.0) - survival_probs
    
    # Calculate the dirty price
    pv_cash_flows = (cash_flows * discount_factors * survival_probs).sum()
    pv_recovery = (notional * recovery_rate * default_probs * discount_factors).sum()
    dirty_price = pv_cash_flows + pv_recovery
    return dirty_price


def defaultable_bond_dirty_price_from_z_spread(
    ref_date: Union[dt.date, pd.Timestamp],
    issue_date: Union[dt.date, pd.Timestamp],
    maturity: int,
    coupon_rate: float,
    coupon_freq: int,
    z_spread: float,
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond from the Z-spread.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    issue_date (Union[dt.date, pd.Timestamp]): Bond's issue date.
    maturity (int): Bond's maturity in years.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    z_spread (float): Z-spread.
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # Calculate the cash flows
    cash_flows = bond_cash_flows(
        ref_date, issue_date, maturity, coupon_rate, coupon_freq, notional
    )

    # Discount factors with z-spread
    cf_dates = pd.to_datetime(cash_flows.index)
    discount_factors.index = pd.to_datetime(discount_factors.index)
    interpolated_dfs = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, d, discount_factors.index, discount_factors) 
        for d in cf_dates
    ]
    discount_factors = pd.Series(data=interpolated_dfs, index=cash_flows.index)
    spread = np.exp(
            [
                -z_spread * year_frac_act_x(ref_date, date, 365)
                for date in cash_flows.index
            ]
        )
    spread = pd.Series(data=spread, index=cash_flows.index)

    # Calculate the dirty price
    dirty_price = (cash_flows * discount_factors * spread).sum()
    return dirty_price