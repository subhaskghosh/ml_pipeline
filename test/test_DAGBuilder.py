from unittest import TestCase

from core.builder import DAGBuilder


class TestDAGBuilder(TestCase):
    def test_get(self):
        db = DAGBuilder(path="./resources/dummy/dummy_clustering.yaml", param = {'run_date':'20210920'})
        dag = db.get()
        r = dag.run()
        #print(r.get_payload())
        #dag.show(path='./resources/dummy/')
        pass