import argparse
from src.processing import build_dataset
from src.config import load_config

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    build_dataset(args.input_dir, args.output_dir, cfg, args.debug)

if __name__ == "__main__":
    main()