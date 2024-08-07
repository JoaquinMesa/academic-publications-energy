import sqlalchemy as sa
import urllib
from pandas import DataFrame
import pandas as pd
#import cStringIO

class ODBC_DB(object):
    """Class for connecting to ODBC database and then selecting, inserting etc data"""

    def __init__(self, database, server, username, password):
        self.database = database
        self.server = server
        self.username = username
        self.password = password
        self.driver = '{ODBC Driver 11 for SQL Server}'

        self.connectionString = '''Driver=%s;
                                      Server=%s;
                                      DATABASE=%s;
                                      UID=%s;
                                      PWD=%s; 
                                      ''' % (self.driver, self.server, self.database, self.username, self.password)

        params = urllib.parse.quote_plus(self.connectionString)

        try:
            self.engine = sa.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
            self.metadata = sa.MetaData()
            self.engine.connect()
        except Exception as inst:
            print('Failed to connect to ' + self.database + ' database')
            #print(inst)
            raise
            # return self

    def gettableschema(self,tablename):
        # gets the table columns and data types through reflection
        table = sa.Table(tablename, self.metadata, autoload=True, autoload_with=self.engine)
        return table

    def selecttabledata(self,sqlstatement):
        # provide a sqlalchemy statement, returns a dataframe
        conn = self.engine.connect()
        results = conn.execute(sqlstatement)
        df = DataFrame(results.fetchall())
        df.columns = results.keys()
        return df


    def bulkinsert(self,data,table):
        # SQL BULK Insert option for large datasets, requires connection to SQL server local drive
        connection = self.engine.raw_connection()
        cursor = connection.cursor()

        data.to_csv('?', sep='\t', header=False, index=False)

        sql = """
        BULK INSERT myDb.dbo.SpikeData123
        FROM 'C:\\__tmp\\biTest.txt' WITH (
            FIELDTERMINATOR='\\t',
            ROWTERMINATOR='\\n'
            );
        """

        cursor.execute(sql)
        connection.commit()
        cursor.close()
        connection.close()

    def getdf(self, tablestr):

        try:
            data = pd.read_sql(tablestr, self.engine)
            return data
        except Exception as inst:
            print('Failed to get date from table: ' + tablestr)
            #print(inst)
            raise


    def fastinsert(self,table,dataframe):
        # fast insert, using execute many option, table is SQL table to insert to
        # dataframe is inserted data - note columns in dataframe must match SQL table
        try:
            connection = self.engine.raw_connection()
            cursor = connection.cursor()

            cursor.fast_executemany = True
            cursor.connection.autocommit = True

            MyCols = dataframe.columns.values
            sql = "INSERT INTO %s ([" % table + ",[".join(MyCols + "]") + ") VALUES (" \
                    + ("?," * MyCols.size)[:-1] + ")"

            params = [tuple(x) for x in dataframe.values]

            result = cursor.executemany(sql, params)
            cursor.close()
            connection.close()
            return result
        except Exception as inst:
            print("unable to insert values into " + table)
            #print(inst)
            raise

    def str_insert(self,query,dataframe):
        # fast insert, using execute many option, with manual definition of insert query
        # query is e.g. """INSERT INTO dbo.UK_INDO_DemandData VALUES (?,?,?,?,?,?,?,?,?,?,?)"""
        # dataframe is inserted data

        try:
            connection = self.engine.raw_connection()
            cursor = connection.cursor()

            cursor.fast_executemany = True
            cursor.connection.autocommit = True

            params = [tuple(x) for x in dataframe.values]

            result = cursor.executemany(query, params)
            cursor.close()
            connection.close()
            return result
        except Exception as inst:
            print("unable to insert data using query:  " + query)
            #print(inst)
            raise


    def insertdataframe(self,dataframe,table):
        # this is so slow it's not worth using
        # Check if table exists first, otherwise dataframe.to_sql will make the table
        try:
            self.gettableschema(table)
        except sa.exc.NoSuchTableError as Err:
            raise sa.exc.NoSuchTableError('Insert failed: The table %s was not found in the database' % table)

        # Nwow append data
        result = dataframe.to_sql(name=table, con=self.engine, if_exists='append', index=False)
        return result

    def executesql(self,procedurename,params=None):
        # Can be used to execute SQL e.g. select * from table
        # or can run a procedure with paramaters eg. executesql('MyProc',[10,4])
        # params can be a single integer or string, or a list, or a tuple
        if params == None:
            conn = self.engine.connect()
            result = conn.execute(procedurename)
        else:

            if isinstance(params,tuple) or isinstance(params,list):
                numparams = len(params)
            elif isinstance(params,int) or isinstance(params,str):
                numparams = 1
            else:
                raise ValueError('params must be tuple,list or single integer or string')

            conn = self.engine.connect()
            result = conn.execute(procedurename +
                                  (" ?," * ((numparams / len("?,")) + 1))[:-1],
                                  params)

        if result.returns_rows == True:
            if result.rowcount == 0:
                print('Procedure returned a row set with zero records')
                output = 0
            else:
                df = DataFrame(result.fetchall())
                df.columns = result.keys()
                output = df
        else:
            output =  result.rowcount

        conn.close()
        return output


