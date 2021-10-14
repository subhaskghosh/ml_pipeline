""" Executable node database writer class. Currently supports postgresql
This script defines the class that can be used for defining a node in the DAG.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""
from core.error import NodeConfigurationError, NodeDBError
from core.logmanager import get_logger
from core.nodes.common.sql.template2sql import apply_sql_template
from core.nodes.node import AbstructNode
import psycopg2 as pg
import pandas as pd
from pandas.io.sql import DatabaseError

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

                cut_id = pd.read_sql(sql=query, con=connection)

                # Add dataframe out output
                self.addToCache(self.output, cut_id)

                # Close the file
                qtf.close()
        except pg.Error as e:
            self.logger.exception('Connection error {0}'.format(e))
            raise NodeDBError(
                'Connection error {0}'.format(e))