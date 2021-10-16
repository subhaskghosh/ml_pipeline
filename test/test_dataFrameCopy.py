from unittest import TestCase
from sqlalchemy import MetaData, create_engine

from core.nodes.load.db.writer import DataFrameCopy


class TestDataFrameCopy(TestCase):
    def test_copy(self):
        conn_string = 'postgresql+psycopg2://postgres:m0nd0d0r1@localhost:5432/test'
        engine = create_engine(conn_string, echo=True)
        metadata = MetaData(schema='dna_ml')
        metadata.reflect(bind=engine)
        table_model = metadata.tables['cluster_classification']
        with engine.connect() as c:
            DataFrameCopy(df, conn=c, table_obj=table_model).copy()
