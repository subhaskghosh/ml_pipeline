""" DAG cache - where variable can be stored and retrived. Think the was variables
in a cell of note book becomes available to subsequent cells.
"""
__author__ = "Subhas K. Ghosh"
__version__ = "1.0"

from core.error import CacheViolationError
from core.logmanager import get_logger


class AbstructCache(object):
    def __init__(self):
        pass

    def update(self,k, v):
        pass

    def get(self,k):
        pass

    def delete(self,k):
        pass

class SimpleCache(AbstructCache):
    def __init__(self):
        '''In the simple form - its just a dict'''
        super().__init__()
        self.logger = get_logger("SimpleCache")
        self.store = {}

    def update(self,k, v):
        self.store[k] = v

    def get(self,k):
        if k in self.store:
            return self.store[k]
        else:
            self.logger.exception('Key "{0}" does not exits!'.format(k))
            raise CacheViolationError(
                'Key "{0}" does not exits!'.format(k))

    def delete(self,k):
        if k in self.store:
            return self.store.pop(k, None)
        else:
            self.logger.exception('Key "{0}" does not exits!'.format(k))
            raise CacheViolationError(
                'Key "{0}" does not exits!'.format(k))