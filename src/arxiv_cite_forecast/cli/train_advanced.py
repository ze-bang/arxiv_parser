import argparse
from pathlib import Path

import pandas as pd

from ..models.advanced import train_multihorizon_model, AdvancedArtifacts
from ..models.baseline import save_artifacts


def main():
    p = argparse.ArgumentParser(description="Train advanced multi-horizon model")
    p.add_argument("--data", required=True, help="Parquet dataset path")
    p.add_argument("--text-encoder", choices=["tfidf", "transformer", "hybrid"], 
                   default="transformer", help="Text encoder type")
    p.add_argument("--use-graph-features", action="store_true", 
                   help="Enable graph-based author features")
    p.add_argument("--horizons", nargs="+", type=int, default=[12],
                   help="Citation horizons to predict (months)")
    p.add_argument("--model-out", required=True, help="Output model path")
    p.add_argument("--config-out", help="Output config JSON path")
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    print(f"Loaded {len(df)} samples")
    
    try:
        artifacts, metrics = train_multihorizon_model(
            df,
            text_encoder_type=args.text_encoder,
            use_graph_features=args.use_graph_features,
            horizons=args.horizons
        )
        
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        save_artifacts(artifacts, args.model_out)
        
        print("Training metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        if args.config_out:
            import json
            Path(args.config_out).parent.mkdir(parents=True, exist_ok=True)
            config = {
                'model_type': 'advanced',
                'text_encoder': args.text_encoder,
                'use_graph_features': args.use_graph_features,
                'horizons': args.horizons,
                'metrics': metrics
            }
            with open(args.config_out, 'w') as f:
                json.dump(config, f, indent=2)
        
        print(f"Model saved to: {args.model_out}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
