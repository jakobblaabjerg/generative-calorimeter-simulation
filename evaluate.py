import argparse
from src.trainer import run_eval
from src.config import load_config
import os
import json 

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--cfg_sampling", type=str, default="configs/sampling.yanml")
    parser.add_argument("--num_mc_samples", type=int, default=1)
    parser.add_argument("--data_dir", type=int, required=True)
    parser.add_argument("--save_dir", type=int, default=None)
    args = parser.parse_args()

    cfg_sampling = load_config(file_path=args.cfg_sampling)

    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = os.path.dirname(os.path.abspath(args.cfg_file))

    os.makedirs(save_dir, exist_ok=True)

    metrics = run_eval(
        model_dir=args.model_dir, 
        cfg_sampling=cfg_sampling, 
        data_dir=args.data_dir, 
        num_mc_samples=args.num_mc_samples, 
        seed=123,
        )

    save_path = os.path.join(save_dir, f"metrics.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()
