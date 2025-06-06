import argparse
import logging

from ludwig import automl
from ludwig.utils import defaults
from pandas import read_csv

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Render a Ludwig config')
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to the dataset file',
        required=True)
    parser.add_argument(
        '--output_feature',
        type=int,
        help='Name for the output feature',
        required=True)
    parser.add_argument(
        '--output',
        type=str,
        help='Path for the output file',
        required=True)
    parser.add_argument(
        '--renderconfig',
        action='store_true',
        help='Render the config',
        required=False,
        default=False)
    args = parser.parse_args()

    # get the output feature name
    df = read_csv(args.dataset, nrows=2, sep=None, engine='python')
    names = df.columns.tolist()
    target = names[args.output_feature - 1]

    args_init = ["--dataset", args.dataset,
                 "--target", target,
                 "--output", args.output]
    automl.cli_init_config(args_init)

    if args.renderconfig:
        args_render = ["--config", args.output, "--output", args.output]
        defaults.cli_render_config(args_render)


if __name__ == "__main__":
    main()
