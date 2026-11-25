"""Evaluate YOLO + ConvNeXt + required variant recognition in real scenes.

Requires manual per-object ground-truth CSV with columns:
 image,x1,y1,x2,y2,type,variant

The YOLO training labels remain single-class bonbon. This additional table is
for evaluation only and MUST NOT be exposed to the inference models.

 python -m src.evaluation.end_to_end --split val
 python -m src.evaluation.end_to_end --split test
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from src.config import catalog, load_config, path_from_root
from src.evaluation.variant_metrics import UNKNOWN, save_variant_report

IMAGE_EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}


def overlap_iou(a,b):
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3]))
    intersection=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(0,float(a[2])-float(a[0]))*max(0,float(a[3])-float(a[1]))
    area_b=max(0,float(b[2])-float(b[0]))*max(0,float(b[3])-float(b[1]))
    denominator=area_a+area_b-intersection
    return intersection/denominator if denominator>0 else 0.0


def match_boxes(ground_truth, predictions, threshold:float=0.5):
    """Greedy one-to-one matching ranked by IoU, independent of class labels."""
    candidates=[]
    for gi,gt in enumerate(ground_truth):
        for pi,pred in enumerate(predictions):
            iou=overlap_iou((gt['x1'],gt['y1'],gt['x2'],gt['y2']),pred['bbox_xyxy'])
            if iou>=threshold:
                candidates.append((iou,gi,pi))
    candidates.sort(reverse=True)
    matched=[]; used_gt=set(); used_pred=set()
    for iou,gi,pi in candidates:
        if gi not in used_gt and pi not in used_pred:
            matched.append((gi,pi,iou))
            used_gt.add(gi); used_pred.add(pi)
    return matched


def summarize_scene(ground_truth,predictions,*,iou_threshold:float=0.5):
    pairs=match_boxes(ground_truth,predictions,iou_threshold)
    matched_gt={gi:(pi,iou) for gi,pi,iou in pairs}
    type_correct=variant_correct=0
    evaluations=[]
    for gi,gt in enumerate(ground_truth):
        expected=f"{gt['type']}/{gt['variant']}"
        if gi in matched_gt:
            pi,iou=matched_gt[gi]
            pr=predictions[pi]
            is_type=pr['type_predit']==gt['type']
            is_variant=is_type and pr['couleur_predite']==gt['variant']
            type_correct+=int(is_type)
            variant_correct+=int(is_variant)
            predicted=(f"{pr['type_predit']}/{pr['couleur_predite']}"
                       if pr.get('couleur_predite') else UNKNOWN)
            evaluations.append({'true_variant':expected,'predicted_variant':predicted,
                                'matched':True,'type_correct':is_type,
                                'joint_correct':is_variant,'iou':iou})
        else:
            evaluations.append({'true_variant':expected,'predicted_variant':UNKNOWN,
                                'matched':False,'type_correct':False,
                                'joint_correct':False,'iou':None})
    return {'gt':len(ground_truth),'predictions':len(predictions),'matched':len(pairs),
            'false_positives':len(predictions)-len(pairs),
            'missed':len(ground_truth)-len(pairs),
            'type_correct_on_matched':type_correct,
            'fully_correct':variant_correct,
            'scene_exact':len(pairs)==len(ground_truth)==len(predictions) and
                          variant_correct==len(ground_truth),
            'objects':evaluations}


def read_reference(csv_path:Path,images_dir:Path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'Missing labeled scene annotations: {csv_path}')
    classes,variants=catalog()
    table=defaultdict(list)
    with csv_path.open(newline='',encoding='utf8-sig') as file:
        reader=csv.DictReader(file)
        required={'image','x1','y1','x2','y2','type','variant'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Ground truth CSV must contain columns {sorted(required)}')
        for row in reader:
            name=row['image'].strip()
            if Path(name).name!=name:
                raise ValueError('Use image FILE NAMES (no directory paths) in reference CSV')
            typ=row['type'].strip(); variant=row['variant'].strip()
            if typ not in classes or variant not in variants[typ]:
                raise ValueError(f'Invalid catalog annotation: {typ}/{variant}')
            box={key:float(row[key]) for key in ('x1','y1','x2','y2')}
            if not all(np.isfinite(v) for v in box.values()) or box['x2']<=box['x1'] or box['y2']<=box['y1']:
                raise ValueError(f'Invalid box for {name}: {box}')
            table[name].append({**box,'type':typ,'variant':variant})
    available={p.name for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()}
    if not available:
        raise ValueError(f'No scene images in {images_dir}')
    unknown=set(table)-available
    if unknown:
        raise ValueError(f'Annotations reference missing images: {sorted(unknown)[:3]}')
    # Every image in images_dir is evaluated, including negative scenes with zero boxes.
    return table


def run(split:str,device:str='auto'):
    if split not in ('val','test'):
        raise ValueError('Split must be val/test')
    detection=load_config('detection'); eval_settings=detection['evaluation']
    images_dir=path_from_root(eval_settings['images'][split]); reference=path_from_root(eval_settings['ground_truth'][split])
    truth=read_reference(reference,images_dir)
    from ultralytics import YOLO
    from src.pipeline import ACSPipeline,resolve_device
    from src.detection.integrated_pipeline import run_stages
    device=str(resolve_device(device))
    detector_path=path_from_root(detection['checkpoint'])
    if not detector_path.is_file():
        raise FileNotFoundError(f'Missing YOLO checkpoint: {detector_path}')
    detector=YOLO(str(detector_path)); classifier=ACSPipeline(device=device)
    infer=detection['inference']; threshold=float(eval_settings['matching_iou'])
    all_reports=[]; truth_variants=[]; predicted_variants=[]; durations=[]
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS or not image.is_file(): continue
        with Image.open(image) as source:
            rgb=np.asarray(ImageOps.exif_transpose(source).convert('RGB'),dtype=np.uint8)
        t0=time.perf_counter()
        results,_=run_stages(rgb,detector=detector,classifier=classifier,
            detector_conf=float(infer['conf']),imgsz=int(infer['image_size']),
            device=device,iou=float(infer['iou']),padding=float(infer['padding']),
            type_warning_threshold=float(infer['type_warning_threshold']))
        elapsed=time.perf_counter()-t0
        report=summarize_scene(truth.get(image.name,[]),results,iou_threshold=threshold)
        durations.append(elapsed); report.update(image=image.name,time_seconds=elapsed)
        all_reports.append(report)
        for item in report['objects']:
            truth_variants.append(item['true_variant']); predicted_variants.append(item['predicted_variant'])
    total_gt=sum(x['gt'] for x in all_reports)
    total_preds=sum(x['predictions'] for x in all_reports)
    total_match=sum(x['matched'] for x in all_reports)
    full=sum(x['fully_correct'] for x in all_reports)
    def f1(precision,recall):
        return 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0
    detection_precision=total_match/total_preds if total_preds else 0.
    detection_recall=total_match/total_gt if total_gt else 0.
    end_precision=full/total_preds if total_preds else 0.
    end_recall=full/total_gt if total_gt else 0.
    output=path_from_root(f'results/evaluation/end_to_end/{split}'); output.mkdir(parents=True,exist_ok=True)
    summary={'split':split,'images':len(all_reports),'reference_objects':total_gt,
        'predicted_objects':total_preds,'matched_objects':total_match,
        'fully_correct_objects':full,
        'type_accuracy_on_matched':sum(x['type_correct_on_matched'] for x in all_reports)/total_match if total_match else 0.,
        'variant_accuracy_on_matched':full/total_match if total_match else 0.,
        'detection_precision_at_configured_threshold':detection_precision,
        'detection_recall_at_configured_threshold':detection_recall,
        'detection_f1_at_configured_threshold':f1(detection_precision,detection_recall),
        'end_to_end_precision':end_precision,'end_to_end_recall':end_recall,
        'end_to_end_f1':f1(end_precision,end_recall),
        'exact_scene_accuracy':sum(x['scene_exact'] for x in all_reports)/len(all_reports),
        'mean_processing_seconds':float(np.mean(durations)),
        'p95_processing_seconds':float(np.percentile(durations,95)),
        'matching_iou':threshold}
    (output/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (output/'per_scene.json').write_text(json.dumps(all_reports,indent=2),encoding='utf8')
    if truth_variants:
        save_variant_report(truth_variants,predicted_variants,output/'object_22_classes',
                            split=split)
    print(f'END-TO-END [{split}]: {summary}')
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument('--split',required=True,choices=('val','test'))
    parser.add_argument('--device',default='auto',choices=('auto','cpu','cuda'))
    args=parser.parse_args(argv); run(args.split,args.device)

if __name__=='__main__':
    main()
