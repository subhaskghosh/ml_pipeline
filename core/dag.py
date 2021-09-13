""" Directed Acyclic Graph that executes.
This script defines the class that can be used for building a DAG with nodes.
based on https://github.com/xianghuzhao/paradag, but using networkx and
added executable type for nodes.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

from core.error import *
import networkx as nx

from core.executor import Executor
from core.nodes.node import AbstructNode
from core.processor import Processor
from core.selector import Selector


def call_method(instance, method, *args, **kwargs):
    try:
        func = getattr(instance, method)
    except AttributeError:
        return None

    if not callable(func):
        return None

    return func(*args, **kwargs)

class Dag(object):
    """Provides functionality to create a DAG"""

    def __init__(self):
        self.selector = Selector()
        self.graph = nx.DiGraph()
        self.processor = Processor()
        self.executor = Executor()

    def add_vertex(self, *vertices):
        '''Add one or more vertices'''
        for v in vertices:
            self.graph.add_node(v)

    def vertices(self):
        '''Get the vertices list'''
        return self.graph.nodes()

    def add_edge(self, u, *vs):
        '''Add edge(s) from one vertex to others'''
        self.validate_vertex(u, *vs)
        for v_to in vs:
            if self.has_path_to(v_to, u):  # pylint: disable=arguments-out-of-order
                raise DAGCycleError(
                    'Cycle if add edge from "{0}" to "{1}"'.format(u, v_to))
            self.graph.add_edge(u, v_to)

    def edges(self):
        '''Get the edges list'''
        return self.graph.edges()

    def validate_vertex(self, *vertices):
        for vtx in vertices:
            if vtx not in self.graph.nodes:
                raise DAGVertexNotFoundError(
                    'Vertex "{0}" does not belong to DAG'.format(vtx))

    def has_path_to(self, u, v):
        if u == v:
            return True
        for vtx in self.graph.successors(u):
            if self.has_path_to(vtx, v):
                return True
        return False

    def vertex_size(self):
        '''Get the number of vertices'''
        return len(self.graph.nodes)

    def edge_size(self):
        '''Get the number of edges'''
        return len(self.graph.edges)

    def successors(self, v):
        '''Get the successors of the specified vertex'''
        self.validate_vertex(v)
        return self.graph.successors(v)

    def predecessors(self, v):
        '''Get the predecessors of the specified vertex'''
        self.validate_vertex(v)
        return self.graph.predecessors(v)

    def indegree(self, v):
        '''Get the indegree of the specified vertex'''
        return len(list(self.predecessors(v)))

    def outdegree(self, v):
        '''Get the outdegree of the specified vertex'''
        return len(list(self.successors(v)))

    def endpoints(self, degree_callback):
        endpoints = set()
        for vtx in self.graph.nodes():
            if degree_callback(vtx) == 0:
                endpoints.add(vtx)
        return endpoints

    def all_starts(self):
        '''Get all the starting vertices'''
        return self.endpoints(self.indegree)

    def all_terminals(self):
        '''Get all the terminating vertices'''
        return self.endpoints(self.outdegree)

    def show(self, options=None):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10), dpi=80)

        if not options:
            self.options = {
                'arrowstyle': '->',
                'arrowsize': 20,
                "font_size": 11,
                "node_size": 10000,
                "node_color": "white",
                "edgecolors": "black",
                "linewidths": 2,
                "width": 2,
                "node_shape": 's',
            }
        else:
            self.options = options

        pos = self.hierarchy_pos()

        nx.draw(self.graph, pos, with_labels=False, arrows=True, **self.options)

        lpos = {}
        tpos = {}
        labels = {}
        elabels = {}

        for v in self.vertices():
            if not isinstance(v,AbstructNode):
                labels[v] = str(v)
            else:
                labels[v] = v.name

        for v in self.vertices():
            elabels[v] = str(v)

        for k, v in pos.items():
            lpos[k] = (v[0]+0.04,v[1])
            tpos[k] = (v[0],v[1]+0.06)

        nx.draw_networkx_labels(self.graph, tpos, labels, **self.options)
        nx.draw_networkx_labels(self.graph, lpos, elabels, **self.options)

        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

    def process_vertices(self, vertices_to_run, vertices_running, processor, executor):
        def execute_func(param):
            return call_method(executor, 'execute', param)

        vertices_with_param = [(vtx, call_method(executor, 'param', vtx))
                               for vtx in vertices_to_run]
        try:
            return processor.process(vertices_with_param, execute_func)
        except VertexExecutionError:
            call_method(executor, 'abort', vertices_running)
            call_method(processor, 'abort')
            raise

    def run(self):
        '''Run tasks according to DAG'''
        indegree_dict = {}
        for vtx in self.vertices():
            indegree_dict[vtx] = self.indegree(vtx)

        vertices_final = []
        vertices_running = set()
        vertices_zero_indegree = self.all_starts()

        while vertices_zero_indegree:
            vertices_idle = vertices_zero_indegree - vertices_running
            vertices_to_run = self.selector.select(vertices_running, vertices_idle)
            call_method(self.executor, 'report_start', vertices_to_run)

            vertices_running |= set(vertices_to_run)
            call_method(self.executor, 'report_running', vertices_running)

            processed_results = self.process_vertices(
                vertices_to_run, vertices_running, self.processor, self.executor)
            call_method(self.executor, 'report_finish', processed_results)

            vertices_processed = [result[0] for result in processed_results]
            vertices_running -= set(vertices_processed)

            vertices_final += vertices_processed
            vertices_zero_indegree -= set(vertices_processed)

            for vtx, result in processed_results:
                for v_to in self.successors(vtx):
                    call_method(self.executor, 'deliver', v_to, result)

                    indegree_dict[v_to] -= 1
                    if indegree_dict[v_to] == 0:
                        vertices_zero_indegree.add(v_to)

        return result

    def hierarchy_pos(self, root=None, width=1., vert_gap=0.2, vert_loc=0.0, xcenter=0.5):
        if len(list(self.all_starts())) > 1:
            from networkx.drawing.nx_pydot import graphviz_layout
            return graphviz_layout(self.graph, prog="dot")

        self.root = list(self.all_starts())[0]

        if not nx.is_tree(self.graph):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0.0, xcenter=0.5, pos=None, parent=None):

            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)
            if len(children) != 0:
                dx = width / len(children)
                nextx = xcenter - width / 2 - dx / 2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                         vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                         pos=pos, parent=root)
            return pos

        return _hierarchy_pos(self.graph, self.root, width, vert_gap, vert_loc, xcenter)