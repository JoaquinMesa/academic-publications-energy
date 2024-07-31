## Loads the historic UK_INDO_DemandData into the database
# used for the LSTM model training
# Only has to be done once per year

import pandas as pd
from API_ELEXON import Period_ELEXON
from datetime import datetime
import DB
from Parameters import name, ip, user, password

# Establish a database connection
dbTriad = DB.ODBC_DB(name, ip, user, password)

# Retrieve the existing data from the database table
trainingtable = dbTriad.getdf("UK_INDO_DemandData")

# If there is already data in the table, notify and exit the script
if not trainingtable.empty:
    print("There is already Training data in the DB. Please use or delete it from the DB")
    exit()

# Define the date ranges for data extraction
date_ranges = [
    ('2015-11-01', '2016-02-29'),
    ('2016-11-01', '2017-02-28'),
    ('2017-11-01', '2018-02-28')
]

# Load data for each date range
data_frames = []
for start, end in date_ranges:
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    period_data = Period_ELEXON(start_date, end_date)
    data_frames.append(period_data)

# Concatenate the data frames
period_loaded = pd.concat(data_frames, ignore_index=True)

# Add an index column to the data
index = [x for x in range(1, period_loaded.shape[0] + 1)]
index_df = pd.DataFrame(index, columns=['Index'])
period_loaded = pd.concat([index_df, period_loaded], axis=1)

# Define the SQL query for inserting data
query = """INSERT INTO dbo.UK_INDO_DemandData VALUES (?,?,?,?,?,?,?,?,?,?,?)"""

# Insert the data into the database
dbTriad.str_insert(query, period_loaded)
