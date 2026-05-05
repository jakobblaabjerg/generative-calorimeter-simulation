from src.config import load_config, get_search_space
from src.trainer import run_sweep
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--space", type=str, default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg_base = load_config(f"configs/base_{args.model}.yaml")

    # optional encoder
    if args.encoder and args.model == "cfm":
        cfg_encoder = load_config(f"configs/{args.encoder}_encoder.yaml")
        cfg_base.model.encoder = cfg_encoder

    search_space = get_search_space(
        load_dir="configs", 
        model=args.model, 
        encoder=args.encoder, 
        selected_space=args.space
        )    

    run_sweep(
        cfg=cfg_base, 
        search_space=search_space, 
        num_trials=args.trials, 
        num_samples=args.samples, 
        debug=args.debug
        )


if __name__ == "__main__":
    main()

