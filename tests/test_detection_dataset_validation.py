import pytest
from pathlib import Path
from PIL import Image
from src.detection.train import inspect_dataset


def test_yolo_annotation_validation(tmp_path):
    for split in ('train', 'val', 'test'):
        (tmp_path / split / 'images').mkdir(parents=True)
        (tmp_path / split / 'labels').mkdir(parents=True)
        Image.new('RGB', (16, 16)).save(tmp_path / split / 'images' / '01.jpg')
        (tmp_path / split / 'labels' / '01.txt').write_text('0 0.5 0.5 0.5 0.5\n')
    cfg = tmp_path / 'data.yaml'
    cfg.write_text('train: train/images\nval: val/images\ntest: test/images\nnames: [bonbon]\nnc: 1\n')
    assert inspect_dataset(cfg)['images'] == {'train':1,'val':1,'test':1}
    # L'ancien export Roboflow utilise aussi 'bonbom' pour la même classe 0.
    cfg.write_text('train: train/images\nval: val/images\ntest: test/images\nnames: [bonbom]\nnc: 1\n')
    assert inspect_dataset(cfg)['classes'] == ['bonbom']
    (tmp_path / 'val' / 'labels' / '01.txt').write_text('1 0.5 0.5 0.5 0.5\n')
    with pytest.raises(ValueError, match='Invalid bonbon class'):
        inspect_dataset(cfg)
