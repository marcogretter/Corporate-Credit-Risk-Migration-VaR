# Corporate Credit Risk and Rating Migration Analysis in Python

This repository contains a Python implementation of corporate bond credit-risk
calibration and portfolio credit-risk simulation.

The project is divided into two main case studies:

1. calibration of the creditworthiness of a corporate issuer from bond prices;
2. Monte Carlo estimation of default and rating-migration risk for a portfolio of
   corporate zero-coupon bonds.

The analysis includes:

- Z-spread estimation;
- constant hazard-rate calibration;
- piecewise-constant hazard-rate bootstrapping;
- risk-neutral default probability estimation;
- comparison with historical transition probabilities;
- credit deterioration scenario analysis;
- rating migration modelling;
- one-factor dependence across issuers;
- default-only and migration-adjusted Value at Risk.

The reference valuation date is 15 February 2008.

## Project Overview

The first part of the project studies two fixed-rate corporate bonds issued by
the same company, referred to as Beta.

The second part studies a homogeneous portfolio of 100 corporate zero-coupon
bonds initially rated Investment Grade.

The objective is to connect:

- observed bond prices;
- credit spreads;
- default intensities;
- survival probabilities;
- historical rating transitions;
- portfolio loss distributions;
- credit Value at Risk.

## Case Study A — Corporate Bond Credit Calibration

Polimi Capital holds two fixed-rate bonds issued by Beta.

### One-Year Bond

- Position: long
- Face value: EUR 10 million
- Maturity: 1 year
- Annual coupon rate: 5.5%
- Coupon frequency: semi-annual
- Market dirty price: 100.00% of face value

### Two-Year Bond

- Position: long
- Face value: EUR 10 million
- Maturity: 2 years
- Annual coupon rate: 7.0%
- Coupon frequency: semi-annual
- Market dirty price: 103.00% of face value

The valuation uses the discount curve constructed in the preceding EURIBOR 3M
curve-bootstrap assignment.

## Risk-Free Bond Valuation

Before introducing credit risk, the contractual cash flows are discounted using
the reference interest-rate curve.

For a bond with coupon cash flows and principal repayment, the risk-free value
is computed as:

```text
risk_free_value =
    sum(discount_factor_i * coupon_cash_flow_i)
    + discount_factor_maturity * principal
```

The difference between this value and the observed market price reflects the
additional compensation required for credit and liquidity risk.

## Z-Spread

The Z-spread is the constant spread added to the risk-free zero-rate curve so
that the discounted contractual cash flows reproduce the observed dirty price.

For a candidate spread `z`, the model bond price is:

```text
bond_price(z) =
    sum(
        cash_flow_i
        * discount_factor_i
        * exp(-z * maturity_i)
    )
```

The calibrated Z-spread solves:

```text
bond_price(z_spread) = observed_dirty_price
```

The Z-spread is estimated separately for the one-year and two-year bonds.

## Interpretation of the Z-Spread

The Z-spread is a convenient summary measure of the yield compensation embedded
in the bond price.

However, it is not a structural default parameter.

It combines several effects, including:

- expected credit losses;
- liquidity compensation;
- risk premiums;
- market segmentation;
- model and curve differences.

Unlike a hazard-rate model, the Z-spread does not explicitly separate:

- survival cash flows;
- recovery payments;
- default timing.

For this reason, the Z-spread and default intensity should not be interpreted as
equivalent quantities.

## Constant Default Intensity

The first reduced-form credit model assumes that the default intensity is
constant through time.

For a constant intensity `lambda`, the survival probability at maturity `T` is:

```text
survival_probability(T) = exp(-lambda * T)
```

The cumulative default probability is:

```text
default_probability(T) =
    1 - survival_probability(T)
```

A recovery rate of 40% is assumed.

```text
recovery_rate = 0.40
loss_given_default = 0.60
```

The intensity is calibrated by matching the defaultable bond value to its
observed dirty price.

## Defaultable Bond Valuation

The bond value includes two components:

1. contractual payments received if the issuer survives;
2. recovery received if default occurs.

A generic discrete-time approximation is:

```text
defaultable_bond_value =
    present_value_of_survival_cash_flows
    + present_value_of_expected_recovery
```

The survival component is:

```text
survival_component =
    sum(
        discount_factor_i
        * contractual_cash_flow_i
        * survival_probability_i
    )
```

The recovery component depends on the assumed recovery convention.

The implementation follows the recovery convention provided by the assignment
code and applies it consistently across all calibrations.

## Separate Constant-Intensity Calibrations

A separate constant intensity is calibrated from each bond:

```text
lambda_1y = intensity implied by the one-year bond
lambda_2y = intensity implied by the two-year bond
```

The resulting intensities are not necessarily equal.

Possible reasons include:

- a non-flat credit term structure;
- different liquidity across maturities;
- different coupon structures;
- market noise;
- recovery-model assumptions;
- different risk premiums across maturities.

Calibrating each bond independently provides two maturity-specific summaries,
but it does not produce a single internally consistent credit curve.

## Piecewise-Constant Intensity Bootstrap

A more flexible specification assumes that the hazard rate is constant within
successive maturity intervals.

For the two-bond case:

```text
lambda(t) = lambda_1      for 0 < t <= 1 year
lambda(t) = lambda_2      for 1 < t <= 2 years
```

The first intensity is calibrated from the one-year bond.

The second intensity is then calibrated from the two-year bond while keeping the
first interval intensity fixed.

The corresponding survival probabilities are:

```text
survival_probability_1y =
    exp(-lambda_1 * 1)

survival_probability_2y =
    exp(
        -lambda_1 * 1
        -lambda_2 * 1
    )
```

The cumulative default probabilities are:

```text
default_probability_1y =
    1 - survival_probability_1y

default_probability_2y =
    1 - survival_probability_2y
```

## Why the Piecewise Bootstrap Is More Informative

The piecewise-constant model produces a single credit curve that is consistent
with both bond prices.

Compared with two separate constant-intensity calibrations, it:

- preserves information from the short-maturity bond;
- isolates the additional credit risk between years one and two;
- produces coherent survival probabilities across maturities;
- captures a non-flat credit term structure;
- avoids treating each bond as an unrelated calibration problem.

The result is generally a better representation of the issuer's term structure
of creditworthiness.

## Conditional Default Probability

The probability of default during the second year conditional on survival through
the first year is:

```text
conditional_default_probability_1y_to_2y =
    1
    - survival_probability_2y
      / survival_probability_1y
```

Under a piecewise-constant hazard rate, this is equivalent to:

```text
conditional_default_probability_1y_to_2y =
    1 - exp(-lambda_2 * 1)
```

This quantity isolates the credit risk associated specifically with the second
year.

## Credit Deterioration Scenario

The project introduces a scenario in which the market price of the two-year bond
falls from 103 to 97, while the price of the one-year bond remains unchanged.

```text
base_1y_price = 100
base_2y_price = 103

scenario_1y_price = 100
scenario_2y_price = 97
```

The unchanged one-year price implies that the short-term credit assessment is
approximately unchanged.

The lower two-year price indicates a deterioration in medium-term credit
expectations.

## Scenario Recalibration

Under the scenario:

1. retain the first-year intensity calibrated from the unchanged one-year bond;
2. recalibrate the second-year intensity using the new two-year bond price;
3. recompute survival probabilities;
4. recompute cumulative and conditional default probabilities.

The expected qualitative result is:

```text
scenario_lambda_1 approximately equals base_lambda_1

scenario_lambda_2 greater than base_lambda_2
```

Consequently:

```text
scenario_default_probability_2y
    greater than
base_default_probability_2y
```

The scenario therefore concentrates the deterioration in the forward credit risk
between years one and two.

## Risk-Neutral and Historical Default Probabilities

The probabilities inferred from bond prices are risk-neutral probabilities.

They are consistent with market prices and include compensation for bearing
credit risk.

Historical transition matrices instead provide physical or real-world
probabilities estimated from observed rating migrations and defaults.

The two probability measures serve different purposes.

### Risk-Neutral Probabilities

Used for:

- pricing;
- market-consistent valuation;
- credit spread calibration;
- derivative valuation.

### Historical Probabilities

Used for:

- risk measurement;
- scenario analysis;
- economic capital;
- portfolio loss forecasting.

Risk-neutral default probabilities are often higher than historical probabilities
because market prices include risk premiums, liquidity effects, and investor risk
aversion.

## Rating Transition Matrix

The project uses a one-year transition matrix with three states:

- Investment Grade;
- High Yield;
- Default.

A generic transition matrix has the structure:

```text
                To IG     To HY     To Default
From IG         p_IG_IG   p_IG_HY   p_IG_D
From HY         p_HY_IG   p_HY_HY   p_HY_D
From Default    0         0         1
```

Each row must sum to one.

```text
sum(probabilities_in_each_row) = 1
```

The Default state is absorbing.

## Historical One-Year Default Probability

For an issuer initially in a given rating state, the one-year historical default
probability is read directly from the corresponding transition-matrix row.

For an Investment Grade issuer:

```text
historical_default_probability_1y =
    transition_probability_IG_to_Default
```

## Historical Two-Year Default Probability

The two-year transition matrix is obtained by multiplying the one-year transition
matrix by itself:

```text
transition_matrix_2y =
    transition_matrix_1y
    @ transition_matrix_1y
```

The two-year historical default probability is then read from the relevant entry
of the two-year matrix.

## Historical Conditional Default Probability

The historical probability of default during the second year conditional on
survival through the first year is:

```text
conditional_default_probability =
    probability_of_default_by_2y_minus_default_by_1y
    / probability_of_survival_to_1y
```

Equivalent implementation:

```text
conditional_default_probability =
    (
        default_probability_2y
        - default_probability_1y
    )
    / (
        1 - default_probability_1y
    )
```

This quantity is compared with the market-implied conditional default probability
under the credit deterioration scenario.

## Case Study B — Credit Portfolio Model

The second part of the project considers a portfolio of 100 corporate zero-coupon
bonds.

Each bond:

- is issued by a different corporation;
- is initially rated Investment Grade;
- has a maturity of two years;
- pays EUR 1 million at maturity;
- has a recovery rate of 40%.

The portfolio face value is:

```text
portfolio_face_value =
    100 * EUR 1 million
```

The analysis incorporates both:

- default risk;
- rating migration risk.

## Initial Portfolio Mark-to-Market

At the valuation date, all issuers are Investment Grade.

The value of each bond is determined from:

- the risk-free discount curve;
- Investment Grade credit risk;
- expected recovery;
- the transition probabilities used by the model.

The initial portfolio value is:

```text
portfolio_value =
    number_of_bonds
    * value_of_one_IG_bond
```

Because the portfolio is homogeneous at inception, all individual bonds have the
same initial value.

## One-Year Forward Bond Values

At the one-year risk horizon, each surviving issuer may be:

- still Investment Grade;
- downgraded to High Yield.

The bond then has one year of residual maturity.

The project computes the corresponding forward values:

```text
forward_price_if_IG =
    value_at_year_1_of_a_one-year_residual_IG_bond

forward_price_if_HY =
    value_at_year_1_of_a_one-year_residual_HY_bond
```

A downgraded High Yield bond should generally have a lower forward value than an
Investment Grade bond because of its higher credit risk.

```text
forward_price_if_HY
    less than
forward_price_if_IG
```

If the issuer defaults, the portfolio receives the recovery value specified by
the model.

## Migration Loss

Migration risk captures losses that occur even when the issuer does not default.

For an issuer downgraded from Investment Grade to High Yield:

```text
migration_loss =
    forward_price_if_IG
    - forward_price_if_HY
```

This loss can be economically important because downgrades are more frequent
than defaults.

## One-Factor Firm Model

Dependence across issuers is introduced through a common systematic factor.

For issuer `i`, a latent credit variable can be represented as:

```text
credit_variable_i =
    sqrt(rho) * systematic_factor
    + sqrt(1 - rho) * idiosyncratic_factor_i
```

where:

```text
systematic_factor follows a standard normal distribution

idiosyncratic_factor_i follows a standard normal distribution
```

The systematic factor is shared by all issuers.

The idiosyncratic factors are independent across issuers and independent of the
systematic factor.

## Rating Thresholds

Rating thresholds are obtained from the cumulative transition probabilities of
an initially Investment Grade issuer.

For example:

```text
default_threshold =
    inverse_normal_cdf(probability_IG_to_Default)

high_yield_threshold =
    inverse_normal_cdf(
        probability_IG_to_Default
        + probability_IG_to_HY
    )
```

The simulated credit variable determines the issuer's state at the risk horizon.

A generic classification rule is:

```text
if credit_variable <= default_threshold:
    state = Default

elif credit_variable <= high_yield_threshold:
    state = High Yield

else:
    state = Investment Grade
```

The ordering may be adapted to the precise latent-variable convention used in
the implementation, but it must reproduce the transition probabilities when
`rho = 0`.

## Correlation Scenarios

The Monte Carlo analysis is repeated for three asset-correlation levels:

```text
rho = 0.00
rho = 0.25
rho = 0.50
```

The number of Monte Carlo scenarios is:

```text
number_of_simulations = 1_000_000
```

The scenarios illustrate how common systematic risk affects the tail of the
portfolio loss distribution.

## Independent-Issuer Case

When:

```text
rho = 0
```

the issuers are conditionally and unconditionally independent.

The number of defaults and downgrades is relatively concentrated around its
expected value.

Diversification is strongest in this case.

## Positive-Correlation Cases

When correlation increases, issuers are more exposed to the same systematic
shock.

This creates a greater probability of scenarios with:

- many simultaneous defaults;
- many simultaneous downgrades;
- large portfolio losses;
- heavier loss-distribution tails.

Correlation may have a limited effect on expected losses but a substantial
effect on high-quantile risk measures such as 99% VaR.

## Monte Carlo Workflow

For each correlation level:

1. generate one systematic factor per scenario;
2. generate one idiosyncratic factor per issuer and scenario;
3. combine them into latent credit variables;
4. assign each issuer to IG, HY, or Default;
5. count defaults and downgrades;
6. calculate the horizon portfolio value;
7. calculate the portfolio loss;
8. estimate expected event counts;
9. estimate 99% VaR.

## Efficient Simulation

A simulation with one million scenarios and 100 issuers can require substantial
memory.

The implementation can therefore process scenarios in batches.

A representative workflow is:

```text
for each simulation batch:
    simulate systematic factors
    simulate idiosyncratic factors
    classify issuer states
    compute portfolio losses
    store or aggregate required outputs
```

Batch processing reduces peak memory usage without changing the statistical
model.

## Average Number of Defaults

For every correlation level, the simulation reports:

```text
average_number_of_defaults =
    mean(default_count_per_scenario)
```

Since correlation changes dependence but not the marginal default probability,
the average number of defaults should remain approximately stable across
correlation scenarios.

Material differences may indicate:

- insufficient simulations;
- incorrect threshold construction;
- implementation errors;
- inconsistent random-variable scaling.

## Average Number of Downgrades

Similarly:

```text
average_number_of_downgrades =
    mean(downgrade_count_per_scenario)
```

The average should be approximately consistent with:

```text
number_of_issuers
    * probability_IG_to_HY
```

and should not change materially with correlation.

## Default-Only Loss

The first portfolio loss measure considers only defaults.

A generic implementation is:

```text
default_only_loss =
    number_of_defaults
    * loss_per_default
```

where:

```text
loss_per_default =
    non_default_reference_value
    - recovery_value
```

Downgraded issuers are treated as if they retained their non-default benchmark
value.

## Default-and-Migration Loss

The second loss measure includes both defaults and downgrades.

```text
total_credit_loss =
    default_loss
    + migration_loss
```

At issuer level:

```text
if state == Investment Grade:
    horizon_value = forward_price_if_IG

if state == High Yield:
    horizon_value = forward_price_if_HY

if state == Default:
    horizon_value = recovery_value
```

The portfolio loss is then:

```text
portfolio_loss =
    reference_portfolio_value_at_horizon
    - simulated_portfolio_value_at_horizon
```

## Value at Risk

The 99% Value at Risk is obtained from the simulated loss distribution.

```text
var_99 =
    quantile(portfolio_losses, 0.99)
```

The project computes two VaR measures for each correlation level:

```text
default_only_var_99

default_and_migration_var_99
```

This allows the relevance of migration risk to be assessed directly.

## Expected VaR Ranking

The expected qualitative ranking is:

```text
VaR with defaults and migrations
    greater than or equal to
VaR with defaults only
```

For the same loss definition:

```text
VaR at rho = 0.50
    generally greater than
VaR at rho = 0.25
    generally greater than
VaR at rho = 0.00
```

Small deviations may occur because of finite simulation error or discrete loss
distributions, but the overall economic pattern should reflect increasing tail
concentration as correlation rises.

## Why Migration Risk Matters

Default is a severe but relatively rare event.

A downgrade from Investment Grade to High Yield is less severe but generally
more frequent.

Consequently, migration risk can materially affect:

- expected portfolio losses;
- the shape of the loss distribution;
- portfolio VaR;
- economic capital;
- mark-to-market volatility.

Ignoring migration risk may substantially underestimate the risk of a bond
portfolio, especially at short horizons.

## Why Correlation Matters

Correlation determines the degree of diversification in the portfolio.

Low correlation implies that issuer-specific shocks offset each other across a
large portfolio.

High correlation increases the probability that many issuers deteriorate in the
same scenario.

As a result, higher correlation generally:

- leaves marginal event probabilities unchanged;
- leaves expected event counts approximately unchanged;
- increases loss dispersion;
- increases extreme portfolio losses;
- raises high-confidence VaR.

This distinction between expected loss and unexpected loss is central to credit
portfolio modelling.

## Risk-Neutral Calibration vs Portfolio Risk

The two case studies use credit probabilities for different purposes.

### Bond Calibration

Bond prices imply risk-neutral probabilities used for market-consistent pricing.

### Portfolio Simulation

The transition matrix provides historical probabilities used for loss
forecasting and risk measurement.

The two probability sets should not be substituted mechanically because they
belong to different probability measures.

## Numerical Validation

The implementation should include the following checks.

### Bond Repricing

The calibrated Z-spreads and intensities should reproduce the observed dirty
prices within numerical tolerance.

```text
repricing_error =
    model_price - observed_price
```

### Survival Monotonicity

Survival probabilities should not increase with maturity.

```text
survival_probability_2y
    less than or equal to
survival_probability_1y
```

### Valid Probabilities

All default and survival probabilities must lie between zero and one.

```text
0 <= probability <= 1
```

### Positive Intensities

Calibrated hazard rates should normally be non-negative.

A negative intensity would indicate inconsistent prices, an unsuitable recovery
assumption, or an implementation problem.

### Transition-Matrix Rows

Each row of the transition matrix must sum to one.

### Marginal Monte Carlo Frequencies

For every correlation level, simulated unconditional state frequencies should
match the transition probabilities.

### Correlation Invariance of Expected Counts

Average defaults and downgrades should remain approximately constant across
correlation scenarios.

### VaR Convergence

The estimated 99% VaR should be checked for stability across:

- different random seeds;
- different batch sizes;
- increasing simulation counts.

### Scenario Consistency

The fall in the two-year bond price should primarily increase the second-period
hazard rate and the two-year default probability.

## Suggested Repository Structure

```text
corporate-credit-risk-migration-var/
|
|-- README.md
|-- requirements.txt
|
|-- src/
|   |-- market_data.py
|   |-- bond_cashflows.py
|   |-- bond_pricing.py
|   |-- z_spread.py
|   |-- hazard_calibration.py
|   |-- survival_probabilities.py
|   |-- transition_matrix.py
|   |-- forward_bond_pricing.py
|   |-- one_factor_model.py
|   |-- portfolio_simulation.py
|   |-- credit_var.py
|   `-- validation.py
|
|-- notebooks/
|   `-- corporate_credit_risk_analysis.ipynb
|
|-- scripts/
|   `-- run_analysis.py
|
|-- data/
|   `-- README.md
|
|-- results/
|   |-- bond_calibration.csv
|   |-- default_probabilities.csv
|   |-- scenario_analysis.csv
|   |-- simulation_summary.csv
|   |-- credit_var_results.csv
|   `-- figures/
|       |-- survival_curves.png
|       |-- hazard_rates.png
|       |-- loss_distributions.png
|       `-- var_by_correlation.png
|
`-- report/
    `-- assignment_report.pdf
```

The file and folder names can be adapted to the structure of the actual Python
implementation.

## Requirements

A representative Python environment may include:

```text
numpy
pandas
scipy
matplotlib
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

A possible execution command is:

```bash
python scripts/run_analysis.py
```

Alternatively, the complete workflow can be executed from:

```text
notebooks/corporate_credit_risk_analysis.ipynb
```

## Main Outputs

The project reports:

- Z-spread of the one-year bond;
- Z-spread of the two-year bond;
- constant intensity implied by each bond;
- one-year market-implied default probability;
- two-year market-implied default probability;
- piecewise-constant hazard rates;
- conditional default probability between years one and two;
- recalibrated intensities under the bond-price scenario;
- historical one-year and two-year default probabilities;
- historical conditional second-year default probability;
- initial portfolio mark-to-market;
- one-year forward IG bond value;
- one-year forward HY bond value;
- average number of defaults;
- average number of downgrades;
- 99% default-only VaR;
- 99% default-and-migration VaR;
- comparison across correlation levels.

## Technologies

- Python
- NumPy
- pandas
- SciPy
- Matplotlib
- Corporate bond pricing
- Reduced-form credit models
- Hazard-rate calibration
- Rating transition matrices
- Monte Carlo simulation
- Credit Value at Risk

## Data

The project uses:

- the EURIBOR discount curve constructed in the preceding assignment;
- contractual data for the two Beta bonds;
- a one-year rating transition matrix;
- credit-state valuation inputs;
- the recovery-rate assumption.

Course-provided or proprietary data should not be included in a public repository
unless redistribution is explicitly permitted.

When the original data cannot be published, the `data` folder should contain a
description of:

- expected files;
- required columns;
- rating-state labels;
- units;
- date formats;
- recovery conventions.

## Academic Context

This project was developed as part of the Risk Management section of the
Financial Engineering course at Politecnico di Milano.

The repository presents the Python implementation, credit calibration,
scenario analysis, rating-migration model, and Monte Carlo portfolio risk
assessment developed for the assignment.
