import argparse
from src.processing import DatasetBuilder
from src.config import load_config

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="configs/filters.yaml")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config_path)

    builder = DatasetBuilder(config=cfg)
    builder.build(args.input_dir, args.output_dir, debug=args.debug)


if __name__ == "__main__":
    main()