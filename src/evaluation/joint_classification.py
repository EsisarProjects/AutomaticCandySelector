"""Evaluate both frozen classifiers jointly on isolated test photographs.

ConvNeXt predicts the type; the required color classifier predicts the variant
using THAT predicted type (not the ground truth). Complete catalog performance.

 python -m src.evaluation.joint_classification --split val
 python -m src.evaluation.joint_classification --split test
"""
from __future__ import annotations
import argparse
import csv
import json
import numpy as np
from PIL import Image, ImageOps
from src.classification.train import check_splits
from src.config import catalog, load_config, path_from_root
from src.evaluation.variant_metrics import UNKNOWN, save_variant_report
from src.evaluation.reporting import save_report
from src.pipeline import ACSPipeline


def run(split: str, device: str = 'auto'):
    if split not in ('val', 'test'):
        raise ValueError('Split must be val or test')
    cfg=load_config('classification')
    sets=check_splits(path_from_root(cfg['dataset']['processed']),cfg)
    ds=sets[split]
    if not ds.samples:
        raise ValueError(f'No images for {split}')
    classifier=ACSPipeline(device=device)
    names,variants=catalog()
    type_index={name:i for i,name in enumerate(names)}
    truth,pred,truth_type,pred_type,files=[],[],[],[],[]
    detailed=[]
    for path, label in ds.samples:
        expected_type=names[label]
        expected_variant=path.parent.parent.name
        with Image.open(path) as im:
            rgb=np.asarray(ImageOps.exif_transpose(im).convert('RGB'), dtype=np.uint8)
        result=classifier.predict_rgb(rgb)
        prediction_type=result['type_predit']
        prediction_variant=result['couleur_predite']
        predicted_full=(f'{prediction_type}/{prediction_variant}'
                        if prediction_variant in variants.get(prediction_type,[]) else UNKNOWN)
        expected_full=f'{expected_type}/{expected_variant}'
        truth.append(expected_full); pred.append(predicted_full)
        truth_type.append(type_index[expected_type]); pred_type.append(type_index[prediction_type])
        files.append(str(path))
        detailed.append({'image':str(path),'true_type':expected_type,'predicted_type':prediction_type,
                         'true_variant':expected_variant,'predicted_variant':prediction_variant or UNKNOWN,
                         'type_correct':expected_type==prediction_type,
                         'joint_correct':expected_full==predicted_full,
                         'type_confidence':result['confiance_type'],
                         'color_confidence':result['score_couleur']})
    root=path_from_root(f'results/evaluation/joint_classification/{split}')
    variant_metrics=save_variant_report(truth,pred,root/'variant_classes',split=split,files=files)
    type_metrics=save_report(truth_type,pred_type,names,root/'type_6_classes',split=split,files=files)
    good_type=[r for r in detailed if r['type_correct']]
    results={
        'split':split,'images':len(detailed),'type_accuracy':type_metrics['accuracy'],
        'type_macro_f1':type_metrics['macro_f1'],
        'joint_variant_accuracy':variant_metrics['accuracy'],
        'joint_variant_macro_f1':variant_metrics['macro_f1'],
        'variant_unknown_rate':variant_metrics['unknown_rate'],
        'variant_accuracy_given_correct_type':(sum(r['joint_correct'] for r in good_type)/len(good_type)
                                               if good_type else None),
        'uses_predicted_type_for_variant':True,
        'catalog_variants':sum(map(len,variants.values())),
    }
    root.mkdir(parents=True,exist_ok=True)
    (root/'metrics.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    with (root/'predictions.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(detailed[0])); w.writeheader(); w.writerows(detailed)
    print(f'JOINT [{split}] type={results["type_accuracy"]:.4f} '
          f'{results["catalog_variants"]}-variant={results["joint_variant_accuracy"]:.4f} '
          f'unknown_rate={results["variant_unknown_rate"]:.3f}')
    return results


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--split',choices=('val','test'),required=True)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    args=parser.parse_args(argv)
    run(args.split,args.device)

if __name__ == '__main__':
    main()
