from unittest import TestCase
import pprint
from core.builder import DAGBuilder


class TestDAGBuilder(TestCase):
    def test_get(self):
        db = DAGBuilder(path="./resources/dummy/dummy_clustering.yaml", param = {'run_date':'20210920', 'features': 'numeric_columns'})
        dag = db.get()
        r = dag.run()
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(r)
        #dag.show(path='./resources/dummy/')