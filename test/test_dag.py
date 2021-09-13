from unittest import TestCase
from core.dag import Dag


class TestDag(TestCase):

    def test_add_vertex(self):
        self.d = Dag()
        self.d.add_vertex(0, 1, 2, 3, 4, 5, 6, 7)
        self.d.add_edge(0, 2)
        self.d.add_edge(1, 2)
        self.d.add_edge(2, 3)
        self.d.add_edge(3, 4)
        self.d.add_edge(4, 5)
        self.d.add_edge(5, 6)
        self.d.add_edge(5, 7)
        print(self.d.all_starts())
        print(self.d.all_terminals())
        self.d.show(options = {
            'arrowstyle': '->',
            'arrowsize': 20,
            "font_size": 8,
            "node_size": 200,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
            "node_shape": 's',
        })

    def test_add_edge(self):
        import random
        self.d = Dag()
        options = {
            'arrowstyle': '->',
            'arrowsize': 20,
            "font_size": 8,
            "node_size": 200,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 2,
            "width": 2,
            "node_shape": 's',
        }

        for i in range(0,100):
            self.d.add_vertex(i)

        for i in range(0,99):
            u = random.randint(0,i)
            v = random.randint(i+1,99)
            self.d.add_edge(u, v)

        self.d.show(options=options)


    def test_validate_vertex(self):
        import math
        import random
        self.d = Dag()

        a = random.randint(0,10)
        b = math.sqrt
        c = math.cos
        self.d.add_vertex(a)
        self.d.add_vertex(b)
        self.d.add_vertex(c)

        self.d.add_edge(a, b)
        self.d.add_edge(b, c)
        self.d.show()

    def test_has_path_to(self):
        self.fail()

    def test_vertex_size(self):
        self.fail()

    def test_edge_size(self):
        self.fail()

    def test_successors(self):
        self.fail()

    def test_predecessors(self):
        self.fail()

    def test_indegree(self):
        self.fail()

    def test_outdegree(self):
        self.fail()

    def test_endpoints(self):
        self.fail()

    def test_all_starts(self):
        self.fail()

    def test_all_terminals(self):
        self.fail()

    def test_show(self):
        self.fail()

    def test_run(self):
        self.fail()
