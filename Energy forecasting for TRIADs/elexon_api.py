import pandas as pd  # Data manipulation and analysis
import numpy as np  # Numerical operations
from datetime import datetime  # Date and time handling
import logging  # Logging for tracking events and errors
import requests  # Making HTTP requests
import os  # Interacting with the operating system
import urllib  # URL handling
from urllib3 import request  # Making HTTP requests with urllib3
from lxml import objectify  # XML handling with lxml
from collections import OrderedDict  # Preserving the order of dictionary keys
from xml.dom import minidom  # XML parsing with minidom

# Import custom parameters from an external file
from Parameters import url as refer, windsolarreport, demandreport, proxy

# Definition of custom functions for API data extraction

def BMRS_GetXML(**kwargs):
    """
    Retrieves XML data from the BMRS API for the given parameters.
    
    **kwargs: Arbitrary keyword arguments representing API parameters.
    
    Returns:
        xml: The parsed XML object.
    """
    url = 'https://api.bmreports.com/BMRS/{report}/v1?APIKey=<<YOURAPIKEY>>&ServiceType=xml'.format(**kwargs)
    for key, value in kwargs.items():
        if key not in ['report']:  # 'report' is already included in the URL
            a = "&%s=%s" % (key, value)
            url = url + a
    print(url)
    xml = objectify.parse(urllib.request.urlopen(url))  # Parse the XML data from the URL
    return xml

def BMRS_Dataframe(**kwargs):
    """
    Processes the XML data into a DataFrame.
    
    **kwargs: Arbitrary keyword arguments representing API parameters.
    
    Returns:
        df: DataFrame containing the processed XML data.
    """
    result = None
    numerror = 0
    while result is None:
        try:
            tags = []
            output = []
            for root in BMRS_GetXML(**kwargs).findall("./responseBody/responseList/item/"):
                tags.append(root.tag)
                
            tag = OrderedDict((x, 1) for x in tags).keys()
            df = pd.DataFrame(columns=tag)
            
            for root in BMRS_GetXML(**kwargs).findall("./responseBody/responseList/item"):
                data = root.getchildren()
                output.append(data)
                
            df = pd.DataFrame(output, columns=tag)
            return df
        
        except Exception as exception:
            print(type(exception).__name__ )
            numerror = numerror + 1
            print(numerror)
            assert type(exception).__name__ == 'NameError'

def xml2df(data_xml):
    """
    Converts XML data into a DataFrame.
    
    data_xml: Parsed XML object.
    
    Returns:
        df_data: DataFrame containing the XML data.
    """
    data = []
    root = data_xml.getroot()
    for item in root.responseBody.responseList.item:
        el_data = {}
        for child in item.getchildren():
            el_data[child.tag] = child.text
        data.append(el_data)
        
    df_data = pd.DataFrame(data)
    
    return df_data

def get_weekdays(days_list):
    """
    Converts a list of dates into their respective weekdays.
    
    days_list: List of dates in 'YYYY-MM-DD' format.
    
    Returns:
        weekday: DataFrame containing the weekday for each date.
        DateStamp_new: DataFrame containing the formatted date strings.
        hour: DataFrame containing the time in 'HH:MM' format.
    """
    DateStamp_new = np.zeros(len(days_list)).tolist()
    weekday = np.zeros(len(days_list)).tolist()
    
    for i in range(len(days_list)):
        DateStamp_new[i] = datetime.strptime(str(days_list[i]), "%Y-%m-%d")  # Convert to datetime
        weekday[i] = DateStamp_new[i].weekday()
        DateStamp_new[i] = datetime.strftime(DateStamp_new[i],  "%Y-%m-%d")
    
    hour_format = pd.date_range(start="00:00", periods=len(days_list), freq="30T").strftime("%H:%M").tolist()
    
    DateStamp_new = pd.DataFrame((DateStamp_new), columns=["DateStamp"])
    weekday = pd.DataFrame(weekday, columns=["Weekday"])
    hour = pd.DataFrame(hour_format, columns=["Hour"])
    
    try:
        assert(len(weekday) == len(DateStamp_new) == len(hour))  # Ensure all lists have the same length
    except AssertionError:
        print("ERROR: Check assertion on line 108 in API_ELEXON.py")
        logging.error("Check assertion on line 108 in API_ELEXON.py")
        raise

    return weekday, DateStamp_new, hour

def Today_ELEXON(now):
    """
    Retrieves and processes ELEXON data for the specified date.
    
    now: The date for which data is to be retrieved, in 'YYYY-MM-DD' format.
    
    Returns:
        Output_data: DataFrame containing the combined data for renewables and day-ahead demand.
    """
    renewables_key = {'report': windsolarreport, 'SettlementDate': now, 'Period': '*', 'Data col name': 'WindAndSolar'}
    
    input_D1 = dict(renewables_key)
    del input_D1['Data col name']
    print(input_D1) 
    
    data_ren = xml2df(BMRS_GetXML(**input_D1))  # Collecting data using the function that grabs XML

    dayahead_key = {'report': demandreport, 'FromDate': now, 'ToDate': now, 'Data col name': 'DayAheadDemand'}
    
    input_D2 = dict(dayahead_key)
    del input_D2['Data col name']
    print(input_D2)  
    
    data_day = xml2df(BMRS_GetXML(**input_D2))  # Collecting data using the function that grabs XML
    
    data_ren = pd.DataFrame(data_ren)
    data_day = pd.DataFrame(data_day)
    data_ren = data_ren.loc[data_ren['processType'].str.lower() == 'day ahead']
    
    NDF = data_day[data_day.recordType == 'DANF'][['settlementDate', 'settlementPeriod', 'demand']].rename(index=str, columns={"settlementDate": "SD", "settlementPeriod": "SP", "demand": "NDF"})
    TSDF = data_day[data_day.recordType == 'DATF'][['settlementPeriod', 'demand']].rename(index=str, columns={"settlementPeriod": "SP", "demand": "TSDF"}).set_index(["SP"])

    days_list_ren = NDF["SD"].values.tolist()
    
    weekdays_ren, DateStamp_ren, hour_ren = get_weekdays(days_list_ren)
    
    weekdays_ren.index = NDF["SP"].values.tolist()
    DateStamp_ren.index = NDF["SP"].values.tolist()
    hour_ren.index = NDF["SP"].values.tolist()
    
    NDF = NDF.set_index(["SP"])
    
    dayahead1 = pd.concat([NDF["NDF"], TSDF], axis=1)
    dayahead = pd.concat([DateStamp_ren, hour_ren, weekdays_ren, dayahead1], axis=1).reset_index(level=['SP'])

    data_ren['settlementPeriod'] = data_ren['settlementPeriod'].values.astype("int")
    data_ren.sort_values(by=['settlementPeriod'], ascending=[True], inplace=True)

    Solar = data_ren.loc[data_ren['powerSystemResourceType'] == '"Solar"'][['settlementPeriod', 'quantity']].rename(index=str, columns={"settlementPeriod": "SP", "quantity": "Solar"}).set_index(["SP"])
    Onshore = data_ren.loc[data_ren['powerSystemResourceType'] == '"Wind Onshore"'][['settlementPeriod', 'quantity']].rename(index=str, columns={"settlementPeriod": "SP", "quantity": "WindOnshore"}).set_index(["SP"])
    Offshore = data_ren.loc[data_ren['powerSystemResourceType'] == '"Wind Offshore"'][['settlementPeriod', 'quantity']].rename(index=str, columns={"settlementPeriod": "SP", "quantity": "WindOffshore"}).set_index(["SP"])
    
    Renewables = pd.concat([Solar, Onshore, Offshore], axis=1)
    Wind = pd.DataFrame(Renewables["WindOnshore"].values.astype("float32") + Renewables["WindOffshore"].values.astype("float32"), columns=["Wind"])
    Renewables.reset_index(drop=True, inplace=True)
    dayahead.reset_index(drop=True, inplace=True)
    Output_data = pd.concat([dayahead, Renewables, Wind], axis=1)

    Output_data[['SP', 'Weekday']] = Output_data[['SP', 'Weekday']].astype(int)
    Output_data[['NDF', 'TSDF', 'Solar', 'WindOnshore', 'WindOffshore']] = Output_data[['NDF', 'TSDF', 'Solar', 'WindOnshore', 'WindOffshore']].astype(float)
    
    return Output_data

def Period_ELEXON(start, end):
    """
    Pulls data from Elexon API for a given date range. Can use the same start and end date to grab data for one day.

    Args:
        start (str): The start date in the format 'YYYY-MM-DD'.
        end (str): The end date in the format 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: DataFrame containing the combined renewables and day-ahead demand data.
    """
    index = pd.date_range(start, end).date
    
    # Dictionary for renewables data request
    renewables_key = {'report': windsolarreport, 'SettlementDate': start, 'Period': '*', 'Data col name': 'WindAndSolar'}
    input_D1 = dict(renewables_key)
    del input_D1['Data col name']
    print(input_D1)

    # Collecting initial renewables data
    data_ren = xml2df(BMRS_GetXML(**input_D1))

    # Loop through and add data for all extra days
    if start != end:
        cols = data_ren.columns
        data_ren = []
        for date in index:
            renewables_key = {'report': windsolarreport, 'SettlementDate': date, 'Period': '*', 'Data col name': 'WindAndSolar'}
            input_D1 = dict(renewables_key)
            del input_D1['Data col name']
            print(input_D1)
            
            # Collecting data for one iteration
            data_ren_oneiter = xml2df(BMRS_GetXML(**input_D1)).values.tolist()
            data_ren.extend(data_ren_oneiter)

        data_ren = pd.DataFrame(data_ren, columns=cols)

    # Dictionary for day-ahead demand data request
    dayahead_key = {'report': demandreport, 'FromDate': start, 'ToDate': start, 'Data col name': 'DayAheadDemand'}
    input_D2 = dict(dayahead_key)
    del input_D2['Data col name']
    print(input_D2)

    # Collecting initial day-ahead demand data
    data_day = xml2df(BMRS_GetXML(**input_D2))

    # Loop through and add data for all extra days
    if start != end:
        cols = data_day.columns
        data_day = []
        for date in index:
            dayahead_key = {'report': demandreport, 'FromDate': date, 'ToDate': date, 'Data col name': 'DayAheadDemand'}
            input_D2 = dict(dayahead_key)
            del input_D2['Data col name']
            print(input_D2)
            
            # Collecting data for one iteration
            data_day_oneiter = xml2df(BMRS_GetXML(**input_D2)).values.tolist()
            data_day.extend(data_day_oneiter)

        data_day = pd.DataFrame(data_day, columns=cols)

    # Processing renewables data (solar, wind)
    data_ren = data_ren.loc[data_ren['processType'].str.lower() == 'day ahead']
    data_ren['settlementPeriod'] = data_ren['settlementPeriod'].astype("int")
    data_ren['quantity'] = data_ren['quantity'].astype("float32")
    data_ren.sort_values(by=['settlementDate', 'settlementPeriod'], ascending=[True, True])

    Solar = data_ren.loc[data_ren['powerSystemResourceType'] == '"Solar"'][['settlementDate', 'settlementPeriod', 'quantity']].rename(columns={"settlementPeriod": "SP", "quantity": "Solar"})
    Onshore = data_ren.loc[data_ren['powerSystemResourceType'] == '"Wind Onshore"'][['settlementDate', 'settlementPeriod', 'quantity']].rename(columns={"settlementPeriod": "SP", "quantity": "WindOnshore"})
    Offshore = data_ren.loc[data_ren['powerSystemResourceType'] == '"Wind Offshore"'][['settlementDate', 'settlementPeriod', 'quantity']].rename(columns={"settlementPeriod": "SP", "quantity": "WindOffshore"})

    Renewables = pd.merge(Solar, Onshore, how='left', on=['SP', 'settlementDate'])
    Renewables = pd.merge(Renewables, Offshore, how='left', on=['SP', 'settlementDate'])
    Renewables['Wind'] = Renewables['WindOnshore'] + Renewables['WindOffshore']

    # Processing demand forecast (NDF and TSDF)
    NDF = data_day[data_day.recordType == 'DANF'][['settlementDate', 'settlementPeriod', 'demand']].rename(columns={"settlementDate": "SD", "settlementPeriod": "SP", "demand": "NDF"})
    TSDF = data_day[data_day.recordType == 'DATF'][['settlementDate', 'settlementPeriod', 'demand']].rename(columns={"settlementDate": "SD", "settlementPeriod": "SP", "demand": "TSDF"})

    days_list_ren = NDF["SD"].tolist()
    weekdays_ren, DateStamp_ren, hour_ren = get_weekdays(days_list_ren)

    dayahead1 = pd.merge(NDF, TSDF, how='left', on=['SD', 'SP'])
    dayahead = pd.concat([DateStamp_ren, hour_ren, weekdays_ren, dayahead1], axis=1)
    dayahead['SP'] = dayahead['SP'].astype("int")

    # Joining renewables and day-ahead data together, based on SP and Settlement Date
    Output_data = pd.merge(Renewables, dayahead, how='outer', left_on=['SP', 'settlementDate'], right_on=['SP', 'DateStamp'])
    Output_data[['Weekday']] = Output_data[['Weekday']].astype(int)
    Output_data[['NDF', 'TSDF', 'Solar', 'WindOnshore', 'WindOffshore', 'Wind']] = Output_data[['NDF', 'TSDF', 'Solar', 'WindOnshore', 'WindOffshore', 'Wind']].astype(float)

    return Output_data