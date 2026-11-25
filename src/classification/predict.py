"""Predict a candy type from a single image with deployed ConvNeXt-Tiny."""
import argparse
import json
from pathlib import Path
from src.pipeline import ACSPipeline


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    args = parser.parse_args(argv)
    model = ACSPipeline(device=args.device)
    print(json.dumps(model.predict_image(args.image), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
