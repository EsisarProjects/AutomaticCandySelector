from PIL import Image
from src.data.merge_ourson_red import merge


def test_copy_previous_red_photos_without_destroying_sources(tmp_path):
    root = tmp_path / 'Classification'
    first = root / 'Train_Val' / 'ourson' / 'rougefaible'
    second = root / 'Train_Val' / 'ourson' / 'rougefort'
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    Image.new('RGB', (12, 12), (255, 0, 0)).save(first / 'IMG_001.jpg')
    Image.new('RGB', (12, 12), (200, 0, 0)).save(second / 'IMG_001.jpg')
    assert merge(root, apply=False)[0] == 2
    assert not (root / 'Train_Val' / 'ourson' / 'rouge').exists()
    assert merge(root, apply=True)[0] == 2
    assert len(list((root / 'Train_Val' / 'ourson' / 'rouge').glob('*.jpg'))) == 2
    assert (first / 'IMG_001.jpg').exists()
    assert merge(root, apply=True)[0] == 0
