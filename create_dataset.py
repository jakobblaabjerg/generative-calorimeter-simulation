import argparse
from src.data_processing import DataProcessor

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="configs/filters.yaml")
    args = parser.parse_args()

    processor = DataProcessor(
        cfg=args.config_file,
        output_dir=args.output_dir,
        input_dir=args.input_dir
    )

    processor.build_dataset()


if __name__ == "__main__":
    main()