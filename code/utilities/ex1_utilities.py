"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
from utilities.date_functions import (          #ERROR: date_functions is in the utilities 
    year_frac_act_x,
    date_series,
    year_frac_30e_360,
    schedule_year_fraction,
)
from utilities.ex0_utilities import (
    get_discount_factor_by_zero_rates_linear_interp,
)

from scipy.stats import norm

from typing import Union, List, Tuple


class SwapType(Enum):
    """
    Types of swaptions.
    """

    RECEIVER = "receiver"
    PAYER = "payer"


def swaption_price_calculator(
    S0: float,
    strike: float,
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    underlying_expiry: Union[dt.date, pd.Timestamp],
    sigma_black: float,
    freq: int,
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Return the swaption price defined by the input parameters.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Value date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Underlying forward starting swap expiry.
        sigma_black (float): Swaption implied volatility.
        freq (int): Number of times a year the fixed leg pays the coupon.
        discount_factors (pd.Series): Discount factors.
        swaption_type (SwapType): Swaption type, default to receiver.

    Returns:
        Union[float, Tuple[float, float]]: Swaption price (and possibly delta).
    """

    ttm = year_frac_act_x(ref_date, expiry, 365)
    d1 = 0.5 * sigma_black * np.sqrt(ttm) + (1/(sigma_black*np.sqrt(ttm))) * np.log(S0/strike)
    d2 = d1 - sigma_black * np.sqrt(ttm)

    fixed_leg_payment_dates = date_series(expiry, underlying_expiry, freq)

    # In order to get the bpv we have to delete the expiry from the dates in which we want to interpolate
    # the dfs since if we don't do so the first element of the dfs would be B(t0;t_n,t_n)=1, which is not correct
    # as it is explained in the formula of the bpv
    bpv = basis_point_value(fixed_leg_payment_dates[1:], discount_factors, expiry)

    df_settl = get_discount_factor_by_zero_rates_linear_interp(ref_date, expiry, discount_factors.index, discount_factors.values)

    # ERROR: THE ORIGINAL FUNCTION GIVEN BY THE TEACHERS INVERTED THE RECEIVER WITH THE PAYER FORMULA AND VICEVERSA
    if swaption_type == SwapType.RECEIVER:
        price = bpv * df_settl * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        delta = bpv * df_settl * (norm.cdf(d1) - 1)
    elif swaption_type == SwapType.PAYER:
        price = bpv * df_settl * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))
        delta = bpv * df_settl * norm.cdf(d1)
    else:
        raise ValueError("Invalid swaption type.")

    if compute_delta:
        return price, delta
    else:
        return price


def irs_proxy_duration( 
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: pd.Series,
) -> float:
    """
    Given the specifics of an interest rate swap (IRS), return its rate sensitivity calculated as
    the duration of a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Reference date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.

    Returns:
        (float): Swap duration.
    """
    schedule = [ref_date] + list(fixed_leg_payment_dates)
    
    # Year frac 
    year_fracs = schedule_year_fraction(schedule)

    # We calculate the (t_i - t0) in the numerator using the Act/365 convention:
    time_to_maturities = [year_frac_act_x(ref_date, date, 365) for date in fixed_leg_payment_dates]

    dfs = [
        get_discount_factor_by_zero_rates_linear_interp(
            discount_factors.index[0],
            date,
            discount_factors.index,
            discount_factors.values,
        )
        for date in fixed_leg_payment_dates
    ]
    # We calculate the coupons, the swap rate is annual, so we have to compute the correct value of the rate that we want
    cash_flows = [swap_rate * yf for yf in year_fracs]
    # to the final coupon we add also the notional
    cash_flows[-1] += 1.0

    # Duration formula
    numerator = sum(t * c * df for t, c, df in zip(time_to_maturities, cash_flows, dfs))
    denominator = sum(c * df for c, df in zip(cash_flows, dfs))

    return numerator / denominator


def basis_point_value(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    settlement_date: dt.datetime | None = None,
) -> float:
    """
    Given a swap fixed leg payment dates and the discount factors, return the basis point value.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        settlement_date (dt.datetime | None): Settlement date, default to None, i.e. to today.
            Needed in case of forward starting swaps.

    Returns:
        float: Basis point value.
    """
    if settlement_date is None:
        settlement_date = discount_factors.index[0]

    # Year fractions over the fixed leg accrual periods (from settlement -> first payment, etc.)
    schedule = [settlement_date] + fixed_leg_schedule # aggiungiamo settlement date se è diversa
    year_fracs = schedule_year_fraction(schedule)

    # Discount factors at each payment date
    dfs = [
        get_discount_factor_by_zero_rates_linear_interp(
            discount_factors.index[0],
            date,
            discount_factors.index,
            discount_factors.values,
        )
        for date in fixed_leg_schedule
    ]

    # For forward-start swaps, normalize the DF series by the DF at settlement
    df_settlement = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        settlement_date,
        discount_factors.index,
        discount_factors.values,
    )
    if df_settlement == 0:
        raise ZeroDivisionError("Discount factor at settlement date is zero.")
    dfs = [df / df_settlement for df in dfs]

    # BPV is the sum of (year_fraction * discount_factor) for each payment
    bpv = sum(yf * df for yf, df in zip(year_fracs, dfs))

    return bpv


def swap_par_rate(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: dt.datetime | None = None,
) -> float:
    
    # if fwd start fate is given I interpolate it, else it's 1
    if fwd_start_date is not None:
        discount_factor_t0 = get_discount_factor_by_zero_rates_linear_interp(
            discount_factors.index[0],
            fwd_start_date,
            discount_factors.index,
            discount_factors.values,
        )
    else:
        discount_factor_t0 = 1.0

    bpv = basis_point_value(fixed_leg_schedule, discount_factors, fwd_start_date)

    # I interpolate B(t, T_N)
    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )
    
    # Numerator:
    float_leg = 1.0 - (discount_factor_tN / discount_factor_t0)
    # Final Par swap rate:
    return float_leg / bpv


def swap_mtm(
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Given a swap rate, a fixed leg payment schedule and the discount factors, return the swap
    mark-to-market.

    Parameters:
        swap_rate (float): Swap rate.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        swap_type (SwapType): Swap type, either 'payer' or 'receiver', default to 'payer'.

    Returns:
        float: Swap mark-to-market.
    """

    # Single curve framework, returns price and basis point value
    bpv = basis_point_value(fixed_leg_schedule, discount_factors)
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0],
        fixed_leg_schedule[-1],
        discount_factors.index,
        discount_factors.values,
    )
    float_leg = 1.0 - P_term
    fixed_leg = swap_rate * bpv

    if swap_type == SwapType.PAYER:         #ERROR: it was receiver
        multiplier = 1
    elif swap_type == SwapType.RECEIVER:    #ERROR: it was payer
        multiplier = -1
    else:
        raise ValueError("Unknown swap type.")
    
    return multiplier * (float_leg - fixed_leg)
