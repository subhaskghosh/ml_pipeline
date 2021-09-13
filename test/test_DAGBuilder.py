from unittest import TestCase

from core.config import DAGBuilder


class TestDAGBuilder(TestCase):
    def test_get(self):
        db = DAGBuilder(path="./resources/dummy/dummy_clustering.yaml")
        dag = db.get()
        r = dag.run()
        print(r.get_payload())
        pass
