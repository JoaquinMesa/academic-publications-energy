import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from datetime import datetime
from collections import namedtuple
from numpy import concatenate
import DB
from API_ELEXON import Period_ELEXON
from Parameters import name, ip, user, password


def getTriadParams():
    # Extraction from database of the parameters for the EMA and LSTM
    dbTriad = DB.ODBC_DB(name, ip, user, password)
    LSTM_ModelParameters = dbTriad.getdf("LSTM_ModelParameters")
    last = pd.DataFrame(LSTM_ModelParameters.iloc[-1]).T

    Name_editor = last["Editor"].values[0]
    niter = last["NIterations"].values[0]
    alpha_soft = last["AlphaSoft"].values[0]
    alpha_hard = last["AlphaHard"].values[0]
    lag = last["Lag"].values[0]
    perc_soft = last["PercSoft"].values[0]
    perc_hard = last["PercHard"].values[0]
    start = datetime.strptime(str(last["StartIni"].values[0].date()), '%Y-%m-%d')
    end = datetime.strptime(str(last["StartEnd"].values[0].date()), '%Y-%m-%d')

    TriadParams = namedtuple("TriadParams", "niter alpha_soft alpha_hard lag perc_soft perc_hard start end")
    tp = TriadParams(niter, alpha_soft, alpha_hard, lag, perc_soft, perc_hard, start, end)
    return tp


def TRIAD_calc(predictors, niter):
    # Machine learning algorithm for prediction of demand for forecast day, by settlement period
    dbTriad = DB.ODBC_DB(name, ip, user, password)
    data = dbTriad.getdf("UK_INDO_DemandData")

    if data.empty:
        print("The UK_INDO_DemandData data table is empty, please check that the Training data has been loaded properly")
        exit()

    data = data[['Wind', "NDF", "TSDF", "Solar", "R1"]]
    pred_values = predictors.values.astype("float32")
    values = data.values.astype("float32")

    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(values)
    pred_values = scaler.fit_transform(pred_values)

    train_X, train_y = values[:, :-1], values[:, -1]
    test_X = pred_values[:, :-1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    model = Sequential()
    model.add(LSTM(40, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    model.fit(train_X, train_y, epochs=niter, batch_size=48*6, verbose=2, shuffle=False)

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat_all = concatenate((test_X, yhat), axis=1)

    inv_yhat_all_sc = scaler.inverse_transform(inv_yhat_all)
    inv_yhat = inv_yhat_all_sc[:, -1].reshape((len(inv_yhat_all_sc), 1))

    return inv_yhat


def change_data_interval(genD, interval):
    # Function to find the maximum demand per day and return that
    M, N = genD.shape
    new_daily = []

    for index in range(0, M, interval):
        one_day_data = np.array(genD[index:index+interval, :])
        max_row = one_day_data[np.argmax(one_day_data[:, -1])]
        new_daily.append(max_row)

    return new_daily


def demandForecastToDF(datestamp, inv_yhat):
    # Convert demand forecast into a dataframe with datestamp, weekday, and forecast
    DateStamp_new = [pd.Timestamp(date) for date in datestamp]
    weekday = [date.weekday() for date in DateStamp_new]

    weekday_df = pd.DataFrame(weekday, columns=["Weekdays"])
    df = pd.DataFrame(inv_yhat, columns=["PeakDemandForecast"])
    dates_df = pd.DataFrame(DateStamp_new, columns=["DateStamp"])

    return pd.concat([weekday_df, dates_df, df], axis=1)


def update_PeakDemandForecast(datestamp, inv_yhat, TriadParams):
    # Update peak demand from the forecast provided (inv_yhat) into DB and append to historic values
    RunDate = datestamp[1]
    dbTriad = DB.ODBC_DB(name, ip, user, password)

    HistoricPeakDemands = dbTriad.getdf("Weekdays_LSTMPrediction")

    if HistoricPeakDemands.empty:
        print("No historic peak demands, calculating now based on start end dates in parameters")
        HistoricPeakDemands = calcHistoricPeakDemands(TriadParams)

    if datetime.strptime(RunDate, '%Y-%m-%d').weekday() >= 5:
        print("Runday is a weekend, skipping the insert of peak demand forecast into database")
        return HistoricPeakDemands

    DemandForecast = demandForecastToDF(datestamp, inv_yhat)
    PeakDemandForecast = change_data_interval(DemandForecast.values, 48)
    PeakDemandForecast = pd.DataFrame(PeakDemandForecast, columns=["Weekdays", "DateStamp", "PeakDemandForecast"])

    dbTriad.executesql(f"DELETE FROM dbo.Weekdays_LSTMPrediction WHERE DateStamp = '{RunDate}'")
    dbTriad.fastinsert("dbo.Weekdays_LSTMPrediction", PeakDemandForecast)

    ExistingValues = HistoricPeakDemands.loc[HistoricPeakDemands['DateStamp'] != pd.Timestamp(RunDate)]
    HistoricPeakDemands = ExistingValues.append(PeakDemandForecast, sort=False)
    return HistoricPeakDemands[["Weekdays", "DateStamp", "PeakDemandForecast"]]


def load_PeakDemandForecast(datestamp, inv_yhat):
    # Load peak demand forecasts into DB
    DemandForecast = demandForecastToDF(datestamp, inv_yhat)
    PeakDemandForecast = change_data_interval(DemandForecast.values, 48)
    PeakDemandForecast = pd.DataFrame(PeakDemandForecast, columns=["Weekdays", "DateStamp", "PeakDemandForecast"])

    PeakDemandForecast = PeakDemandForecast[(PeakDemandForecast.Weekdays != 5) & (PeakDemandForecast.Weekdays != 6)]

    dbTriad = DB.ODBC_DB(name, ip, user, password)
    dbTriad.fastinsert("dbo.Weekdays_LSTMPrediction", PeakDemandForecast)

    return PeakDemandForecast


def calcHistoricPeakDemands(TriadParams):
    # Get data for initiation period, so that EMA can be calculated
    ElexonForecasts = Period_ELEXON(TriadParams.start, TriadParams.end)
    predictors = ElexonForecasts[['Wind', "NDF", "TSDF", "Solar", "NDF"]]
    datestamp = ElexonForecasts["DateStamp"].values
    inv_yhat = TRIAD_calc(predictors, TriadParams.niter)
    return load_PeakDemandForecast(datestamp, inv_yhat)


def EMA(y, alpha, lag, factor):
    # Calculate Exponential Moving Average (EMA)
    EMA = np.zeros(len(y))
    for j in range(1, lag):
        MA = np.mean(y[0:j, -1])
        EMA[j] = (alpha * y[j, -1] + (1 - alpha) * MA) * factor
    for i in range(lag, len(y)):
        MA = np.mean(y[(i - lag):i, -1])
        EMA[i] = (alpha * y[i - 1, -1] + (1 - alpha) * MA) * factor
    return EMA


def writeTriadPrediction(output, RunDate):
    Output = pd.DataFrame(data=output, columns=['DateStamp', 'PeriodID', 'SoftEMA', 'HardEMA',
                                                'LSTMDemandForecast', 'SoftSignals', 'HardSignals'])

    Output['DateStamp'] = pd.to_datetime(Output.DateStamp)
    Output['PeriodID'] = Output['PeriodID'].astype(int)
    Output[['SoftEMA', 'HardEMA', 'LSTMDemandForecast']] = Output[['SoftEMA', 'HardEMA', 'LSTMDemandForecast']].astype(float)
    Output[['SoftSignals', 'HardSignals']] = Output[['SoftSignals', 'HardSignals']].astype(int)

    dbTriad = DB.ODBC_DB(name, ip, user, password)
    dbTriad.executesql(f"DELETE FROM dbo.TriadPrediction WHERE datestamp = '{RunDate}'")
    dbTriad.fastinsert("dbo.TriadPrediction", Output)
