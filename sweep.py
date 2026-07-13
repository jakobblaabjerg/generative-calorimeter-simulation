from src.config import load_config, get_search_space
from src.training.sweep import run_sweep
from src.config import override_config
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--space", type=str, default=None)
    parser.add_argument("--mc_samples", type=int, default=1)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)    
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_files", type=int, default=None)    

    args = parser.parse_args()

    model_dir = (
        f"{args.model[:3]}/{args.model}"
        if args.model.startswith("mdn")
        else args.model
    )

    cfg_base = load_config(
        f"configs/{args.dataset}/models/{model_dir}/base_{args.model}.yaml"
    )

    overrides = {
        "trainer.patience": args.patience,
        "trainer.epochs": args.epochs,
        "data_loader.load_dir": args.data_dir,
        "data_loader.num_files": args.num_files,
        "logger.log_dir": args.log_dir
    }

    override_config(cfg_base, overrides)


    # optional encoder
    if args.encoder and args.model == "cfm":
        cfg_encoder = load_config(f"configs/{args.dataset}/models/{args.model}/{args.encoder}_encoder.yaml")
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
        num_mc_samples=args.mc_samples, 
        )


if __name__ == "__main__":
    main()

