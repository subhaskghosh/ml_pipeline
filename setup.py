from setuptools import setup
from pathlib import Path

source_root = Path(".")

# Read the requirements
with (source_root / "requirements.txt").open(encoding="utf8") as f:
    requirements = f.readlines()

setup(
    name='ml_pipeline',
    version='1.0',
    packages=['api', 'api.cli', 'core', 'core.nodes', 'core.nodes.ml', 'core.nodes.ml.teaft',
              'core.nodes.ml.classifier', 'core.nodes.ml.clustering', 'core.nodes.ml.processing',
              'core.nodes.ml.transformer', 'core.nodes.info', 'core.nodes.load', 'core.nodes.load.db',
              'core.nodes.load.file', 'core.nodes.common', 'core.nodes.common.sql', 'core.nodes.extract',
              'core.nodes.extract.db', 'core.nodes.extract.file', 'core.nodes.transform', 'test'],
    url='',
    python_requires=">=3.7",
    install_requires=requirements,
    license='Copyright GTM.ai, 2021',
    author='Subhas K Ghosh',
    author_email='subhas.ghosh@salesdna.ai',
    description='ML Pipeline DAG creation and execution'
)
