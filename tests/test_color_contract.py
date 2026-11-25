import numpy as np
from src.config import catalog
from src.color.color_classifier import fit_color_models, predict_color


def test_v2_color_bundle_respects_full_catalog():
    _, variants = catalog()
    hue_by_color = {'bleu': 110, 'jaune': 30, 'orange': 15, 'rouge': 0,
                    'vert': 60, 'noir': 5, 'rose': 160, 'blanc': 5,
                    'rougefaible': 1, 'rougefort': 2}
    rows=[]
    for typ, colors in variants.items():
        for color in colors:
            rows.append({'type': typ, 'variant': color,
                         'features': {'status': 'detecte', 'hue_mean': hue_by_color[color],
                                      's_median': 180, 'v_median': 185}})
    bundle=fit_color_models(rows, variants, n_neighbors=1)
    assert len(bundle['variants']) == 6
    assert len(bundle['models']) == 4  # 2 single-variant types are catalog rules
    out=predict_color(bundle,'crocodile',rows[0]['features'])
    assert out['prediction'] == 'bleu'
