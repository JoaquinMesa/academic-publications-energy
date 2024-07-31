import DB
import pandas as pd
from datetime import datetime
import configparser

# Reading configuration from config.ini
config = configparser.ConfigParser()
config.read('config.ini')

now = datetime.now()

# LOGGING
loglevel = config['LOG']['LogLevel']
logFilePath = config['LOG']['filePath']

# DATABASE PARAMETERS
name = config['DBCREDENTIALS']['dbname']
ip = config['DBCREDENTIALS']['dbip']
user = config['DBCREDENTIALS']['dbuser']
password = config['DBCREDENTIALS']['dbpassword']

# Proxy
proxy = config['Proxy']['address']

# API
url = config['API']['apiurl']
windsolarreport = config['API']['windsolarreport']
demandreport = config['API']['demandreport']

# Run Date
rundateStr = config['EDITORANDLSTM']['rundate']

# Validating that the run date is in the correct format, replacing 'today' with today's date
try:
    TryDate = datetime.strptime(rundateStr, '%Y-%m-%d')
    RunDate = TryDate.strftime('%Y-%m-%d')
except ValueError:
    if rundateStr.lower() == 'today':
        RunDate = datetime.now().strftime('%Y-%m-%d')
    else:
        raise ValueError('RunDate input in config.ini not recognized. Please use "YYYY-MM-DD" format or "today".')