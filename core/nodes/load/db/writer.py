""" Executable node database writer class. Currently supports postgresql

Bulk loading of the dataframe class is written based on
Pandas-to-postgres
by the Growth Lab at Harvard's Center for International Development
https://github.com/cid-harvard/pandas-to-postgres

It writes a csv in-memory to a StringIO and chunks through it.

Supports
1. truncate-load,
2. data formmatting, NaN handling and casting data types
3. validation after load - load assurance through count verification
4. SQL templates based on jinja2, with support for looping

TODO: PostgresMetadataWriterNode and PostgresQueryMetadataNode are not very generic

templating is based on jinja2 and is taken from jinja2sql with few modification
and additional functions to our current requirements.

This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from io import StringIO

from sqlalchemy import create_engine, MetaData

from core.error import NodeConfigurationError, NodeDBError
from core.logmanager import get_logger
from core.nodes.common.sql.template2sql import apply_sql_template
from core.nodes.node import AbstructNode
import psycopg2 as pg
import pandas as pd
from sqlalchemy.schema import AddConstraint, DropConstraint
from sqlalchemy.exc import SQLAlchemyError

def create_file_object(df):
    """
    Writes pandas dataframe to an in-memory StringIO file object. Adapted from
    https://gist.github.com/mangecoeur/1fbd63d4758c2ba0c470#gistcomment-2086007
    Parameters
    ----------
    df: pandas DataFrame
    Returns
    -------
    file_object: StringIO
    """
    file_object = StringIO()
    df.to_csv(file_object, index=False)
    file_object.seek(0)
    return file_object

def df_generator(df, chunksize=10 ** 6):
    """
    Create a generator to iterate over chunks of a dataframe
    Parameters
    ----------
    df: pandas dataframe
        Data to iterate over
    chunksize: int
        Max number of rows to return in a chunk
    """
    rows = 0
    if not df.shape[0] % chunksize:
        n_chunks = max(df.shape[0] // chunksize, 1)
    else:
        n_chunks = (df.shape[0] // chunksize) + 1

    for i in range(n_chunks):
        yield df.iloc[rows : rows + chunksize]
        rows += chunksize

def cast_pandas(df, columns=None, copy_obj=None, **kwargs):
    """
    Pandas does not handle null values in integer or boolean fields out of the
    box, so cast fields that should be these types in the database to object
    fields and change np.nan to None
    Parameters
    ----------
    df: pandas DataFrame
        data frame with fields that are desired to be int or bool as float with
        np.nan that should correspond to None
    columns: list of SQLAlchemy Columns
        Columns to iterate through to determine data types
    copy_obj: BaseCopy or subclass
        instance of BaseCopy passed from the BaseCopy.data_formatting method where
        we can access BaseCopy.table_obj.columns
    Returns
    -------
    df: pandas DataFrame
        DataFrame with fields that correspond to Postgres int, bigint, and bool
        fields changed to objects with None values for null
    """

    logger = get_logger("cast_pandas")

    if columns is None and copy_obj is None:
        raise ValueError("One of columns or copy_obj must be supplied")

    columns = columns or copy_obj.table_obj.columns
    for col in columns:
        try:
            if str(col.type) in ["INTEGER", "BIGINT"]:
                df[col.name] = df[col.name].apply(
                    lambda x: None if pd.isna(x) else int(x), convert_dtype=False
                )
            elif str(col.type) == "BOOLEAN":
                df[col.name] = df[col.name].apply(
                    lambda x: None if pd.isna(x) else bool(x), convert_dtype=False
                )
        except KeyError:
            logger.warn(
                "Column {} not in DataFrame. Cannot coerce object type.".format(
                    col.name
                )
            )

    return df


class BaseCopy(object):
    """
    Parent class for all common attibutes and methods for copy objects
    """

    def __init__(
        self,
        conn=None,
        table_obj=None,
        sql_table=None,
        csv_chunksize=10 ** 6,
    ):
        """
        Parameters
        ----------
        conn: SQLAlchemy Connection
            Managed outside of the object
        table_obj: SQLAlchemy Table
            Model object for the destination SQL Table
        sql_table: string
            SQL table name
        csv_chunksize: int
            Max rows to keep in memory when generating CSV for COPY
        """

        self.rows = 0
        self.csv_chunksize = csv_chunksize
        self.conn = conn
        self.table_obj = table_obj
        self.sql_table = sql_table
        self.logger = get_logger(self.sql_table)

    def truncate(self):
        """TRUNCATE PostgreSQL table"""
        self.logger.info("Truncating {}".format(self.sql_table))
        self.conn.execute("TRUNCATE TABLE {};".format(self.sql_table))

    def analyze(self):
        """Run ANALYZE on PostgreSQL table"""
        self.logger.info("Analyzing {}".format(self.sql_table))
        self.conn.execute("ANALYZE {};".format(self.sql_table))

    def copy_from_file(self, file_object):
        """
        COPY to PostgreSQL table using StringIO CSV object
        Parameters
        ----------
        file_object: StringIO
            CSV formatted data to COPY from DataFrame to PostgreSQL
        """
        cur = self.conn.connection.cursor()
        file_object.seek(0)
        columns = file_object.readline()
        sql = "COPY {table} ({columns}) FROM STDIN WITH CSV".format(
            table=self.sql_table, columns=columns
        )
        cur.copy_expert(sql=sql, file=file_object)

    def data_formatting(self, df, functions=None, **kwargs):
        """
        Call each function in the functions list arg on the DataFrame and return
        Parameters
        ----------
        df: pandas DataFrame
            DataFrame to format
        functions: list of functions
            Functions to apply to df. each gets passed df, self as copy_obj, and all
            kwargs passed to data_formatting
        **kwargs
            kwargs to pass on to each function
        Returns
        -------
        df: pandas DataFrame
            formatted DataFrame
        """
        if functions is None:
            functions = []
        for f in functions:
            df = f(df, copy_obj=self, **kwargs)
        return df


class DataFrameCopy(BaseCopy):
    """
    Class for handling a standard case of iterating over a pandas DataFrame in chunks
    and COPYing to PostgreSQL via StringIO CSV
    """

    def __init__(
            self,
            df,
            conn=None,
            table_obj=None,
            sql_table=None,
            csv_chunksize=10 ** 6
    ):
        """
        Parameters
        ----------
        df: pandas DataFrame
            Data to copy to database table
        conn: SQlAlchemy Connection
            Managed outside of the object
        table_obj: SQLAlchemy model object
            Destination SQL Table
        csv_chunksize: int
            Max rows to keep in memory when generating CSV for COPY
        """
        super().__init__(conn, table_obj, sql_table, csv_chunksize)

        self.df = df
        self.rows = self.df.shape[0]

    def copy(self, should_truncate=False, functions=[cast_pandas]):
        self.df = self.data_formatting(self.df, functions=functions)
        with self.conn.begin():
            if should_truncate:
                self.truncate()

            self.logger.info("Creating generator for chunking dataframe")
            for chunk in df_generator(self.df, self.csv_chunksize):

                self.logger.info("Creating CSV in memory")
                fo = create_file_object(chunk)

                self.logger.info("Copying chunk to database")
                self.copy_from_file(fo)
                del fo

            self.logger.info("All chunks copied ({} rows)".format(self.rows))

class PostgresMetadataWriterNode(AbstructNode):
    """Write to a db"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("PostgresMetadataWriterNode")

        # sql to execute
        if 'sql' in self.parameter:
            self.sql = self.parameter['sql']
        else:
            self.logger.exception('SQL query not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'SQL query not provided "{0}"'.format(parameter))

        # sql type
        if 'sql_type' in self.parameter:
            self.sql_type = self.parameter['sql_type']
        else:
            self.logger.exception('SQL type not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'SQL type not provided "{0}"'.format(parameter))

        # if sql-type is template then sql_parameters must be there
        if self.sql_type == 'template':
            if 'sql_parameters' in self.parameter:
                self.sql_parameters = self.parameter['sql_parameters']
            else:
                self.logger.exception('SQL parameters not provided "{0}"'.format(parameter))
                raise NodeConfigurationError(
                    'SQL parameters not provided "{0}"'.format(parameter))

        if 'conn' in self.parameter:
            self.conn = self.parameter['conn']
            host = self.conn['parameter']['host']
            port = self.conn['parameter']['port']
            dbname = self.conn['parameter']['dbname']
            user = self.conn['parameter']['user']
            password = self.conn['parameter']['password']
            self.connection_string = f"host='{host}' port='{port}' dbname='{dbname}' user='{user}' password='{password}'"
        else:
            self.logger.exception('DB connection details not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'DB connection details not provided "{0}"'.format(parameter))

    def execute(self):
        # Create connection from parameters
        try:
            with pg.connect(self.connection_string) as connection:
                # Read the sql file
                qtf = open(self.sql, 'r')
                self.query_template = qtf.read()

                # resolve parameters
                self.sql_parameters_resolved = {}
                for k, v in self.sql_parameters.items():
                    self.sql_parameters_resolved[k] = str(self.getFromCache(v))

                # resolve sql
                query = apply_sql_template(self.query_template, self.sql_parameters_resolved)

                cur = connection.cursor()
                cur.execute(query)
                connection.commit()

                # Close the file
                qtf.close()
        except pg.Error as e:
            self.logger.exception('Connection error {0}'.format(e))
            raise NodeDBError(
                'Connection error {0}'.format(e))

class PostgresQueryMetadataNode(AbstructNode):
    """Query db"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("PostgresQueryMetadataNode")

        # sql to execute
        if 'sql' in self.parameter:
            self.sql = self.parameter['sql']
        else:
            self.logger.exception('SQL query not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'SQL query not provided "{0}"'.format(parameter))

        # sql type
        if 'sql_type' in self.parameter:
            self.sql_type = self.parameter['sql_type']
        else:
            self.logger.exception('SQL type not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'SQL type not provided "{0}"'.format(parameter))

        # select column
        if 'select_column' in self.parameter:
            self.select_column = self.parameter['select_column']
        else:
            self.logger.exception('Select columns not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'select columns not provided "{0}"'.format(parameter))

        # if sql-type is template then sql_parameters must be there
        if self.sql_type == 'template':
            if 'sql_parameters' in self.parameter:
                self.sql_parameters = self.parameter['sql_parameters']
            else:
                self.logger.exception('SQL parameters not provided "{0}"'.format(parameter))
                raise NodeConfigurationError(
                    'SQL parameters not provided "{0}"'.format(parameter))

        if 'conn' in self.parameter:
            self.conn = self.parameter['conn']
            host = self.conn['parameter']['host']
            port = self.conn['parameter']['port']
            dbname = self.conn['parameter']['dbname']
            user = self.conn['parameter']['user']
            password = self.conn['parameter']['password']
            self.connection_string = f"host='{host}' port='{port}' dbname='{dbname}' user='{user}' password='{password}'"
        else:
            self.logger.exception('DB connection details not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'DB connection details not provided "{0}"'.format(parameter))

        # Validate that output definition exists
        if self.output == None:
            self.logger.exception('Output can not be None')
            raise NodeConfigurationError(
                'Output can not be None')

    def execute(self):
        # Create connection from parameters
        try:
            with pg.connect(self.connection_string) as connection:
                # Read the sql file
                qtf = open(self.sql, 'r')
                self.query_template = qtf.read()

                # resolve parameters
                self.sql_parameters_resolved = {}
                for k, v in self.sql_parameters.items():
                    self.sql_parameters_resolved[k] = str(self.getFromCache(v))

                # resolve sql
                query =  apply_sql_template(self.query_template, self.sql_parameters_resolved)

                df = pd.read_sql(sql=query, con=connection)
                r =  df[self.select_column].values.tolist()[0]

                # Add dataframe out output
                self.addToCache(self.output, r)

                # Close the file
                qtf.close()
        except pg.Error as e:
            self.logger.exception('Connection error {0}'.format(e))
            raise NodeDBError(
                'Connection error {0}'.format(e))

class PostgresLoadDataFrameNode(AbstructNode):
    """Append all rows of a DF to a db"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("PostgresLoadDataFrameNode")

        if 'conn' in self.parameter:
            self.conn = self.parameter['conn']
            host = self.conn['parameter']['host']
            port = self.conn['parameter']['port']
            dbname = self.conn['parameter']['dbname']
            user = self.conn['parameter']['user']
            password = self.conn['parameter']['password']
            self.connection_string = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}'
        else:
            self.logger.exception('DB connection details not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'DB connection details not provided "{0}"'.format(parameter))

        # columns
        if 'columns' in self.parameter:
            self.columns = self.parameter['columns']

        if 'from_variable' in self.parameter:
            self.from_variable = self.parameter['from_variable']
        else:
            self.from_variable = None

        if 'add_column' in self.parameter:
            self.add_column = self.parameter['add_column']
        else:
            self.add_column = None

        if 'add_column_name' in self.parameter:
            self.add_column_name = self.parameter['add_column_name']
        else:
            self.add_column_name = None

        # db info
        if 'schema' in self.parameter:
            self.schema = self.parameter['schema']
        else:
            self.logger.exception('DB schema details not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'DB schema details not provided "{0}"'.format(parameter))

        if 'table' in self.parameter:
            self.table = self.parameter['table']
        else:
            self.logger.exception('DB table details not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'DB table details not provided "{0}"'.format(parameter))

        if self.input == None:
            raise NodeConfigurationError(
                'Input can not be None')

    def execute(self):
        if self.from_variable:
            self.columns = self.getFromCache(self.columns)

        # Create connection from parameters
        try:
            engine = create_engine(self.connection_string, echo=True)
            metadata = MetaData(schema=self.schema)
            metadata.reflect(bind=engine)
            table_model = metadata.tables[f'{self.schema}.{self.table}']

            df = self.getFromCache(self.input)

            if self.add_column:
                v =  self.getFromCache(self.add_column)
                df[self.add_column_name] = v

            if self.columns:
                df = df[self.columns]

            with engine.connect() as c:
                DataFrameCopy(df, conn=c, table_obj=table_model, sql_table=f'{self.schema}.{self.table}').copy()
        except SQLAlchemyError as e:
            self.logger.exception('DB error "{0}"'.format(e))