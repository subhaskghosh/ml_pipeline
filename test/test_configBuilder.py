from unittest import TestCase

from core.config import ConfigBuilder


class TestConfigBuilder(TestCase):
    def test_add_section(self):
        c = ConfigBuilder(path="./resources/dummy/dummy_clustering.yaml")
        c.show()

    def test_add_properties(self):
        self.fail()
