import argparse
import json
import os
import sys
import pprint
from core.builder import DAGBuilder

parser = argparse.ArgumentParser(description='The ML Pipeline DAG Executor')

parser.add_argument('--config_path', type=str, default=None, help='Path of the YAML configuration template')
parser.add_argument('--config_parameters', type=str, default=None, help='Path of the configuration run parameters')

args = parser.parse_args()

templete_path = args.config_path
parameter_path = args.config_parameters

if not os.path.exists(templete_path):
    print(f'The path {templete_path} does not exist')
    sys.exit()
else:
    if not os.path.exists(parameter_path):
        print(f'The path {parameter_path} does not exist')
        sys.exit()
    else:
        with open(parameter_path) as json_file:
            param = json.load(json_file)
            db = DAGBuilder(path=templete_path, param = param)
            dag = db.get()
            r = dag.run()
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(r)
