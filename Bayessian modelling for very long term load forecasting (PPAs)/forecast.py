import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.stats as st
from pymc3.math import switch
import pymc3 as pm
from math import pi
import colorednoise as cn


def wind_forecast(wind: pd.Series, n_traces: int = 20) -> (list, pd.DataFrame):
    """
    Forecasts wind data using a Bayesian model with multiple change points and Beta distributions.

    Parameters:
        wind (pd.Series): Time series data of wind measurements.
        n_traces (int): Number of posterior predictive samples to generate for each segment.

    Returns:
        dates_list (list): List of change point indices.
        traces_df (pd.DataFrame): DataFrame containing the posterior predictive samples.
    """

    # Drop any missing values from the wind data
    wind = wind.dropna()

    # Normalize the data to the range [0, 1]
    wind_min = wind.min()
    wind_max = wind.max()
    one_year = (wind - wind_min) / (wind_max - wind_min) + 0.001
    observed = one_year.T.values[0]

    # Create an array of indices for the observed data
    N = np.arange(0, len(observed))

    # Define initial limits for the change points
    lims = [1251, 4200, 7240, 10100, 13100, 16000, int(len(observed) - 1)]

    with pm.Model() as model:
        # Define priors and likelihoods for six different Beta distributions
        alphas_1 = pm.Uniform("alphas_1", lower=0, upper=2)
        betas_1 = pm.Uniform("betas_1", lower=0, upper=10)
        points_1 = pm.Beta("gen_1", alpha=alphas_1, beta=betas_1)

        alphas_2 = pm.Uniform("alphas_2", lower=0, upper=2)
        betas_2 = pm.Uniform("betas_2", lower=0, upper=10)
        points_2 = pm.Beta("gen_2", alpha=alphas_2, beta=betas_2)

        alphas_3 = pm.Uniform("alphas_3", lower=0, upper=2)
        betas_3 = pm.Uniform("betas_3", lower=0, upper=10)
        points_3 = pm.Beta("gen_3", alpha=alphas_3, beta=betas_3)

        alphas_4 = pm.Uniform("alphas_4", lower=0, upper=2)
        betas_4 = pm.Uniform("betas_4", lower=0, upper=10)
        points_4 = pm.Beta("gen_4", alpha=alphas_4, beta=betas_4)

        alphas_5 = pm.Uniform("alphas_5", lower=0, upper=2)
        betas_5 = pm.Uniform("betas_5", lower=0, upper=10)
        points_5 = pm.Beta("gen_5", alpha=alphas_5, beta=betas_5)

        alphas_6 = pm.Uniform("alphas_6", lower=0, upper=2)
        betas_6 = pm.Uniform("betas_6", lower=0, upper=10)
        points_6 = pm.Beta("gen_6", alpha=alphas_6, beta=betas_6)

        # Define priors for the change points
        tau1 = pm.DiscreteUniform("tau1", lower=lims[0], upper=lims[1])
        tau2 = pm.DiscreteUniform("tau2", lower=lims[1], upper=lims[2])
        tau3 = pm.DiscreteUniform("tau3", lower=lims[2], upper=lims[3])
        tau4 = pm.DiscreteUniform("tau4", lower=lims[3], upper=lims[4])
        tau5 = pm.DiscreteUniform("tau5", lower=lims[4], upper=lims[5])
        # tau6 = pm.DiscreteUniform("tau6", lower=lims[5], upper=lims[6])

        # Define the switchpoint model to change the Beta distributions at change points
        mu1 = switch(tau1 >= N, points_1, points_2)
        mu2 = switch(tau2 >= N, mu1, points_3)
        mu3 = switch(tau3 >= N, mu2, points_4)
        mu4 = switch(tau4 >= N, mu3, points_5)
        mu5 = switch(tau5 >= N, mu4, points_6)

        # Define the Poisson likelihood for the observed data
        gen = pm.Poisson("obs", mu5, observed=observed)

    # Sample from the posterior distribution
    with model:
        tr = pm.sample(5000, tune=5000, njobs=1, cores=16)

    # Calculate the mean values of the change points
    t1 = int(tr["tau1"].mean())
    t2 = int(tr["tau2"].mean())
    t3 = int(tr["tau3"].mean())
    t4 = int(tr["tau4"].mean())
    t5 = int(tr["tau5"].mean())

    # Create a list of change point indices
    dates_list = [0, t1, t2, t3, t4, t5, int(len(observed) - 1)]

    traces_all = []

    for t in range(len(dates_list) - 1):
        print(f"\nPerforming {n_traces} traces for wind estimation.")
        print(
            f"Estimating from {one_year.iloc[dates_list[t]].name} to {one_year.iloc[dates_list[t+1]].name}\n"
        )

        # Segment the data between change points
        one_period = one_year.iloc[dates_list[t] : dates_list[t + 1]]
        observed = one_period.T.values[0]
        ndata = len(observed)

        # Define a new model for each segment
        with pm.Model() as model:
            # Define priors for the cluster sizes and Beta distributions
            p = pm.Dirichlet("p", a=np.array([1.0, 1.0, 1.0]), shape=3)
            alphas = pm.Uniform("alphas", lower=0, upper=10, shape=3)
            betas = pm.Uniform("betas", lower=0, upper=10, shape=3)

            # Define the categorical distribution for the clusters
            category = pm.Categorical("category", p=p, shape=ndata)

            # Define the likelihood for each observed value
            points = pm.Beta(
                "gen",
                alpha=alphas[category],
                beta=betas[category],
                observed=observed,
            )

        # Fit the model using Metropolis and ElemwiseCategorical samplers
        with model:
            step1 = pm.Metropolis(vars=[p, alphas, betas])
            step2 = pm.ElemwiseCategorical(vars=[category], values=[0, 1, 2])
            tr = pm.sample(2000, step=[step1, step2], tune=1000, cores=2)

        # Generate posterior predictive samples
        with model:
            ppc_trace = pm.sample_posterior_predictive(tr, 1000)

        # Adjust samples to be within the range of observed data
        for i in range(ppc_trace["gen"].shape[0]):
            for j in range(ppc_trace["gen"].shape[1]):
                ppc_trace["gen"][i][j] = min(
                    max(ppc_trace["gen"][i][j], observed.min()), observed.max()
                )

        one_trace = ppc_trace["gen"][-n_traces:]
        one_trace = np.reshape(
            one_trace, (one_trace.shape[1], one_trace.shape[0])
        )
        traces_all += one_trace.tolist()

        print(np.array(traces_all).shape)
        print()

    traces_all = np.array(traces_all + [[0] * n_traces])
    traces_df = pd.DataFrame(traces_all, index=one_year.index)

    return dates_list, traces_df


def solar_forecast(solar: pd.Series) -> pd.DataFrame:
    """
    Forecasts solar energy generation using a stochastic model.

    Parameters:
        solar (pd.Series): Time series data of solar energy measurements.

    Returns:
        simulated_df (pd.DataFrame): DataFrame containing the simulated solar energy generation.
    """

    # Copy the input data and convert the index to datetime
    train = solar.copy()
    train.index = pd.to_datetime(train.index, format="%Y-%m-%d %H:%M:%S")

    # Extract the solar energy data as a numpy array
    train_array = solar.T.values[0]

    # Compute a seasonal curve by resampling and smoothing the data
    seasonal_curve = (
        train.resample("D").max().rolling(window=30, center=True).mean().dropna().T.values[0]
    )

    # Generate a sinusoidal curve to model the seasonal variation
    days = 365
    t = np.linspace(0, pi, days)
    seasonal = seasonal_curve.max() * np.sin(t)

    # Adjust the sinusoidal curve to ensure a minimum value during winter
    for i in range(len(seasonal)):
        if seasonal[i] < seasonal_curve.min():
            seasonal[i] = int(seasonal_curve.min())

    # Add a decay rate of 1% to model the degradation over time
    decay_rate = 0.01
    decay = 1 - np.linspace(0, decay_rate, len(train_array))

    # Initialize an array to store the simulated solar generation
    zeros = np.zeros(int(len(train_array)))

    # Model solar generation during periods of daylight only
    division = 20
    k = 0
    for i in range(0, len(zeros) - 24, 48):
        t = np.linspace(0, pi, division)
        # Generate solar energy data peaking at midday
        zeros[i + 24 - int(division / 2) : i + 24 + int(division / 2)] = seasonal[k] * np.sin(t)
        k += 1

    # Add Gaussian noise to simulate natural variability
    beta = 0.9
    red_noise = cn.powerlaw_psd_gaussian(beta, len(zeros))
    zeros = zeros * (1 + red_noise * 0.3)

    # Apply decay to the simulated data
    for i in range(len(zeros)):
        zeros[i] = zeros[i] * decay[i]

    # Save the simulated data in a DataFrame
    simulated_df = pd.DataFrame(zeros, index=train.index, columns=["Simulation"])

    return simulated_df