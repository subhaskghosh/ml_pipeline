import argparse
import json
import os
import sys
import pprint
from core.builder import DAGBuilder
from typing import Any, List, Optional
from pathlib import Path

def parse_args(args: Optional[List[Any]] = None) -> argparse.Namespace:
    """Parse the command line arguments for the `ml_pipeline`
    Args:
      args: List of input arguments. (Default value=None).
    Returns:
      Namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='The ML Pipeline DAG Executor'
    )

    # YAML template configuration path

    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help='Path of the YAML configuration template'
    )

    # We can take a set of variables that gets resolved into YAML
    # Pass them as a JSON file
    parser.add_argument(
        '--config_parameters',
        type=str,
        default=None,
        help='Path of the configuration run parameters'
    )

    return parser.parse_args(args)

def main(args: Optional[List[Any]] = None) -> None:
    """Run the `ml_pipeline`.
    Args:
      args: Arguments for the programme (Default value=None).
    """

    # Parse the arguments
    parsed_args = parse_args(args)
    kwargs = vars(parsed_args)

    templete_path = Path(kwargs.pop("config_path"))
    parameter_path = Path(kwargs.pop("config_parameters"))


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

if __name__ == "__main__":
   main(sys.argv[1:])