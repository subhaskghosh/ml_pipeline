""" DAG next node selection method.
Written by Subhas K Ghosh (subhas.k.ghosh@gmail.com).
(c) Copyright , All Rights Reserved.
Table generation:
(c) Copyright Subhas K Ghosh, 2021.
"""

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