"""Run frozen ConvNeXt evaluation with independent validation or final test data.

python -m src.evaluation.classification --split val
python -m src.evaluation.classification --split test
"""
from __future__ import annotations
import argparse
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.classification.train import check_splits
from src.classification.transforms import build_transform
from src.classification.convnext_adapter import load_convnext_model, read_checkpoint
from src.config import catalog, load_config, path_from_root
from src.evaluation.reporting import save_report


def run(split: str, device: str | None = None):
    if split not in ('val', 'test'):
        raise ValueError('Split must be val or test')
    cfg = load_config('classification')
    sets = check_splits(path_from_root(cfg['dataset']['processed']), cfg)
    ds = sets[split]
    if not ds.samples:
        raise ValueError(f'No photographs in {split} split')
    checkpoint = path_from_root(cfg['checkpoint'])
    bundle = read_checkpoint(checkpoint)
    names, _ = catalog()
    if names != bundle['classes']:
        raise ValueError('Checkpoint and catalog mismatch')
    device_name = device or cfg['training']['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device_name)
    model = load_convnext_model(checkpoint, dev, class_names=names)
    ds.transform = build_transform(bundle.get('preprocessing', cfg['preprocessing']), training=False)
    loader = DataLoader(ds, batch_size=int(cfg['training']['batch_size']), shuffle=False,
                        num_workers=int(cfg['training']['num_workers']))
    criterion = nn.CrossEntropyLoss(reduction='sum')
    truth, prediction = [], []
    total_loss = 0.
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(dev))
            total_loss += float(criterion(logits, targets.to(dev)).item())
            truth.extend(targets.tolist())
            prediction.extend(logits.argmax(dim=1).cpu().tolist())
    output = path_from_root('results/evaluation/classification')/split
    result = save_report(truth, prediction, names, output, split=split,
                         files=[p for p, _ in ds.samples])
    result.update(loss=total_loss/len(truth), checkpoint=str(checkpoint))
    (output/'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'{split}: accuracy={result["accuracy"]:.4f}, macro_F1={result["macro_f1"]:.4f}, n={len(truth)}')
    print('Report directory:', output)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    run(args.split, args.device)


if __name__ == '__main__':
    main()
