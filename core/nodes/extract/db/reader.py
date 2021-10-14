""" Executable node database reader class. Currently supports postgresql
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.error import NodeConfigurationError, NodeDBError
from core.logmanager import get_logger
from core.nodes.node import AbstructNode
import psycopg2 as pg
import pandas as pd
from pandas.io.sql import DatabaseError

class PostgresReaderNode(AbstructNode):
    """Read from a db"""
    def __init__(self, name, parameter, input, output):
        super().__init__(name, parameter, input, output)
        self.logger = get_logger("PostgresReaderNode")

        if 'sql' in self.parameter:
            self.sql = self.parameter['sql']
        else:
            self.logger.exception('SQL query not provided "{0}"'.format(parameter))
            raise NodeConfigurationError(
                'SQL query not provided "{0}"'.format(parameter))

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
                query = open(self.sql, 'r')
                try:
                    # Execute the SQL and get a dataframe
                    df = pd.read_sql_query(query.read(), connection)

                    # Add dataframe out output
                    self.addToCache(self.output, df)

                    # Close the file
                    query.close()
                except DatabaseError as e:
                    self.logger.exception('Query error {0}'.format(e))
                    raise NodeDBError(
                        'Query error {0}'.format(e))
        except pg.Error as e:
            self.logger.exception('Connection error {0}'.format(e))
            raise NodeDBError(
                'Connection error {0}'.format(e))
