"""Functional benchmark of the ACTUAL persistent Python worker used by Java.

This validates Java <-> Python backend protocol and emitted artifacts, but does
not automate Swing clicks, layout, or human usability tests. See docs/EVALUATION_PROTOCOL.md.

 python -m src.evaluation.application --source path/to/test_image_or_folder
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from src.config import path_from_root
from src.detection.integrated_pipeline import gather_images


def parse_worker_output(stdout: str, expected:int):
    lines=[line.strip() for line in stdout.splitlines() if line.strip()]
    ready=[line for line in lines if line=='READY']
    annotated=[line[len('RESULT:'):] for line in lines if line.startswith('RESULT:')]
    errors=[line for line in lines if line.startswith('ERROR:')]
    done=sum(line=='DONE' for line in lines)
    if len(ready)!=1 or done!=expected:
        raise RuntimeError(f'Worker protocol mismatch: READY={len(ready)}, DONE={done}, expected={expected}; errors={errors}')
    if errors:
        raise RuntimeError(f'Worker reported errors: {errors}')
    if len(annotated)!=expected:
        raise RuntimeError(f'Worker returned {len(annotated)} images for {expected} requests')
    return annotated


def run(source:Path,timeout_seconds:float=600.0):
    files=gather_images(source.expanduser().resolve())
    root=path_from_root('.')
    payload=''.join(str(path.resolve())+'\n' for path in files)
    start=time.perf_counter()
    completed=subprocess.run([sys.executable,'worker.py'],input=payload,text=True,
                             capture_output=True,cwd=root,timeout=timeout_seconds)
    elapsed=time.perf_counter()-start
    if completed.returncode != 0:
        raise RuntimeError(f'ACS worker exit code {completed.returncode}: '
                           f'{completed.stdout[-2000:]} {completed.stderr[-1000:]}')
    outputs=parse_worker_output(completed.stdout,len(files))
    items=[]
    for image, pathstr in zip(files,outputs):
        annotated=Path(pathstr)
        if not annotated.is_file():
            raise FileNotFoundError(f'Worker did not generate output: {annotated}')
        stem=annotated.stem.removesuffix('_annote')
        prediction_file=annotated.with_name(stem+'_predictions.json')
        if not prediction_file.is_file():
            raise FileNotFoundError(f'Worker JSON missing: {prediction_file}')
        data=json.loads(prediction_file.read_text(encoding='utf8'))
        if data.get('nombre_detections')!=len(data.get('detections',[])):
            raise ValueError(f'Worker JSON is inconsistent for {image}')
        for item in data['detections']:
            if not item.get('type_predit') or not item.get('couleur_predite'):
                # A difficult image may produce a declared unknown. Count it
                # explicitly; don't report it as a successful classification.
                pass
        items.append({'image':str(image),'annotated':str(annotated),
                      'json':str(prediction_file),'detections':data['nombre_detections'],
                      'unresolved_variants':sum(not row.get('couleur_predite')
                                                for row in data['detections'])})
    output=path_from_root('results/evaluation/application');output.mkdir(parents=True,exist_ok=True)
    summary={'images':len(files),'all_images_returned':len(items)==len(files),
             'worker_protocol_passed':True,'json_validation_passed':True,
             'unresolved_variant_count':sum(item['unresolved_variants'] for item in items),
             'total_wall_seconds_including_model_loading':elapsed,
             'mean_wall_seconds_per_image_including_startup':elapsed/len(files),
             'note':'This measures worker/backend, NOT interactive Swing usability or per-image UI latency.',
             'items':items}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Worker/application backend benchmark:',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--timeout-seconds',default=600.,type=float)
    args=parser.parse_args(argv);run(args.source,args.timeout_seconds)

if __name__=='__main__':
    main()
