# Solar and Wind Energy Forecasting

A Power Purchase Agreement (PPA) is a long-term, contract between an Independent Power Producer (IPP) and an oﬀ-taker, usually an energy intensive organisation or a utility company. PPAs are hedging tools by many organisations, as they oﬀer an opportunity for energy buyers to achieve price certainty and at the same time meet their sustainability objectives.

This tool is aimed for PPA analysts to support decisions for early stages of PPA negosiation with clients.
In order to support this negosiation, this tool produces a report that estimates client´s consumption summary based on baseload and peakload (daily and yearly), as well as a price summary, based on input costs for wind and solar per MWh, both total and yearly (within a chosen period).

As data limitation through the negosiation process is an issue, three different levels of estimation are offered, depending on the type of data provided.  The data provided is divided in two different groups: generation and consumption. Generation data can be provided from a prior negosiation in the case this is not available at the beginning. 
Also, the tool takes into account the fact that the client may only want a certain percentage of its electricity demand covered by PPAs or/and using only one type of technology (wind/solar).

The minimum data provided (for LvL. 1 forecast) is average HH client consumption, to make the minimum degree of estimation. In this case, a rough estimation can be achieved with data from energy providers from a prior negosiation.

This repository contains code for forecasting solar and wind energy generation using stochastic models.

## Wind Energy Forecasting


The `wind_forecast` function models wind energy generation using a Bayesian approach with PyMC3. It involves multiple distributions and switch points to accurately capture the variability in wind energy generation. The forecasting is performed using Markov Chain Monte Carlo (MCMC) methods to generate multiple traces, providing an ensemble forecast.

**Key Features:**
- Uses Bayesian inference for robust modeling
- Multiple beta distributions to capture different phases of wind patterns
- Switch points to identify changes in wind regimes
- Generates multiple traces using MCMC for ensemble forecasting

### Model Description

The model is constructed using the following steps:
1. **Data Normalization:** The input wind data is normalized to ensure it fits within a standard range.
2. **Beta Distributions:** Six beta distributions are used to model different phases of wind energy patterns.
3. **Switch Points:** Discrete uniform distributions are used to identify switch points where wind patterns change.
4. **Bayesian Inference:** PyMC3 is used to construct and sample from the Bayesian model.
5. **MCMC Sampling:** MCMC methods are used to generate multiple traces for forecasting.

### Detailed Code Explanation
The wind forecasting function is split into several steps to ensure accurate modeling:

1. **Normalization:** The wind data is normalized to fit within a [0,1] range.
2. **Model Setup:** Six beta distributions (`points_1` to `points_6`) are defined to capture different phases of wind patterns.
3. **Switch Points:** Discrete uniform distributions (`tau1` to `tau5`) are used to identify points where the wind pattern changes.
4. **Likelihood Definition:** The likelihood is defined using Poisson distribution with the switch function to accommodate different phases.
5. **Sampling:** MCMC sampling is performed using PyMC3 to generate multiple traces, providing an ensemble forecast.

### Usage Example
```python
import pandas as pd

# Load your wind data
wind_data = pd.read_csv('wind_data.csv', index_col=0)

# Generate the wind forecast
dates_list, traces_df = wind_forecast(wind_data['Wind'])

# Display the forecasted data
print(traces_df)
```

## Solar Energy Forecasting

The `solar_forecast` function models solar energy generation using a stochastic approach. It takes into account seasonal variations, decay rates, and natural variability to generate a realistic forecast of solar energy generation.

**Key Features:**
- Seasonal variation modeled using a sinusoidal curve
- Decay rate to simulate degradation over time
- Gaussian noise to account for natural variability
- Forecasts solar generation during periods of daylight only

**Usage Example:**
```python
import pandas as pd

# Load your solar data
solar_data = pd.read_csv('solar_data.csv', index_col=0)

# Generate the solar forecast
simulated_solar = solar_forecast(solar_data['Solar'])

# Display the simulated data
print(simulated_solar)
