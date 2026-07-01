from src.config import override_config, load_config
from src.training.sampling import run_generation

import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--config_filters", type=str, default="configs/filters.yaml")
    parser.add_argument("--config_sampling", type=str, default="configs/sampling.yaml")
    
    parser.add_argument("--phi", type=float, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--e_inc", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=None)

    args = parser.parse_args()

    overrides = {
        "data_loader.phi": args.phi,
        "data_loader.theta": args.theta,
        "data_loader.e_inc": args.e_inc,
        "data_loader.num_samples": args.num_samples,
    }

    cfg_filters = load_config(args.config_filters)
    cfg_sampling = load_config(args.config_sampling)
    override_config(cfg_sampling, overrides)

    run_generation(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        cfg_filters=cfg_filters,
        cfg_sampling=cfg_sampling, 
    )

if __name__ == "__main__":
    main()