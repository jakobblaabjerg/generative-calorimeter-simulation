import argparse
from src.trainer import run_eval
from src.config import load_config
import os
import json 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--save_dir", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(file_path=args.cfg_file)

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = os.path.dirname(os.path.abspath(args.cfg_file))

    os.makedirs(save_dir, exist_ok=True)

    metrics = run_eval(
        cfg=cfg, 
        split=args.split, 
        num_samples=args.num_samples
        )

    metrics["split"] = args.split

    save_path = os.path.join(save_dir, f"metrics_{args.split}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()
