import numpy as np
import traceback
import logging

from Parameters import RunDate, loglevel, logFilePath
import TRIADfunctions as tf
from API_ELEXON import Period_ELEXON

def setup_logging():
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {loglevel}')

    logging.basicConfig(filename=logFilePath, level=numeric_level, format='%(asctime)s - %(levelname)s:\t%(message)s \n')
    logging.info(f'Starting Triad model for RunDate: {RunDate}')
    logging.debug('Debug logging is on')
    logging.warning('Warning logging is on')

def main():
    try:
        # Get elexon data for RunDate specified in config file
        ElexonForecasts = Period_ELEXON(RunDate, RunDate)
        logging.debug('Retrieved data from API Elexon')

        # Main LSTM machine learning algorithm - predict the forecast demand for the RunDate by Settlement date
        predictors = ElexonForecasts[['Wind', "NDF", "TSDF", "Solar", "NDF"]]
        TriadParams = tf.getTriadParams()
        logging.debug('Triad Paramaters loaded')

        inv_yhat = tf.TRIAD_calc(predictors, TriadParams.niter)
        logging.info('LSTM model calculated')

        # Faff around with reformatting
        datestamp = ElexonForecasts[['DateStamp', 'SP']]
        date_period_values = datestamp.values
        dates_rep = datestamp["DateStamp"]

        # The inv_yhat array is re-arranged into a matrix that includes the periods in column 0
        date_values = np.reshape(date_period_values[:, 0], (-1, 1))
        period_values = np.reshape(date_period_values[:, 1], (-1, 1))

        # Peak demand per day is loaded into the DB, and historic peaks are loaded back to Python
        y = tf.update_PeakDemandForecast(dates_rep, inv_yhat, TriadParams)
        y = np.asarray(y)
        logging.debug('Loaded and retrieved peak demand per day')

        # EMAs are calculated from the peak demands per day, one for each value of the filter, soft and hard, respectively
        try:
            assert len(y) >= TriadParams.lag
        except AssertionError:
            logging.error("The lag must be less than or equal to the length of the loaded EMA data vector")
            print("ERROR: The lag must be less than or equal to the length of the loaded EMA data vector")
            print("Please choose a lower value for the lag and try again")
            return

        EMAsoft = tf.EMA(y, TriadParams.alpha_soft, TriadParams.lag, TriadParams.perc_soft)
        EMAhard = tf.EMA(y, TriadParams.alpha_hard, TriadParams.lag, TriadParams.perc_hard)

        logging.debug('EMA calculated')

        # Faff around with reformatting
        EMAsoft_ext = np.repeat(EMAsoft, 48)
        EMAhard_ext = np.repeat(EMAhard, 48)

        EMAsoft_ext = np.reshape(EMAsoft_ext, (-1, 1))
        EMAhard_ext = np.reshape(EMAhard_ext, (-1, 1))

        measure_hard = EMAhard_ext[-len(inv_yhat):]
        measure_soft = EMAsoft_ext[-len(inv_yhat):]

        EMA_signals_soft_ext = np.zeros(len(inv_yhat))
        EMA_signals_hard_ext = np.zeros(len(inv_yhat))

        # If the demand forecast is above the EMAs then it is a Triad Warning
        for i in range(len(inv_yhat)):
            if measure_soft[i] < inv_yhat[i]:
                EMA_signals_soft_ext[i] = 1
            if measure_hard[i] < inv_yhat[i]:
                EMA_signals_hard_ext[i] = 1

        # Faff around with reformatting data, create a single output dataframe to write to DB
        EMA_signals_soft_ext = np.reshape(EMA_signals_soft_ext, (-1, 1))
        EMA_signals_hard_ext = np.reshape(EMA_signals_hard_ext, (-1, 1))

        measure_soft = np.reshape(measure_soft, (-1, 1))
        measure_hard = np.reshape(measure_hard, (-1, 1))
        date_period_values = np.reshape(date_period_values, (-1, 1))

        output = np.concatenate(
            (date_values, period_values, measure_soft, measure_hard, inv_yhat, EMA_signals_soft_ext, EMA_signals_hard_ext),
            axis=1
        )

        logging.debug('Completed calculations')

        # Write results to DB
        tf.writeTriadPrediction(output, RunDate)
        logging.info('Completed load results to Triad DB - run has completed successfully \n\n')

    except Exception as inst:
        print(inst)
        print(traceback.format_exc(3))
        logging.error(str(inst))
        logging.error(traceback.format_exc(3) + ' \n\n')

if __name__ == "__main__":
    setup_logging()
    main()