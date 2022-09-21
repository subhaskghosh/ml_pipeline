""" DAG next node selection method.. Since DAG and processor is sequential -
FullSelector is meaningless and is kept for future
"""
__author__ = "Subhas K. Ghosh"
__version__ = "1.0"

class Selector(object):
    '''A selector just selects the next idle vertex'''

    def select(self, _, idle):
        '''Select the next idle vertex'''
        return [next(iter(idle))]

class FullSelector(object):
    '''A selector selects all the idle vertices'''

    def select(self, _, idle):
        '''Select all the idle vertices'''
        return list(idle)