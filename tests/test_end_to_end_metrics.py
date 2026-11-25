from src.evaluation.end_to_end import overlap_iou, match_boxes, summarize_scene


def test_iou_and_one_to_one_matching():
    gt=[{'x1':0,'y1':0,'x2':10,'y2':10,'type':'ourson','variant':'rouge'},
        {'x1':20,'y1':20,'x2':30,'y2':30,'type':'crocodile','variant':'bleu'}]
    pred=[{'bbox_xyxy':[0,0,10,10],'type_predit':'ourson','couleur_predite':'rouge'},
          {'bbox_xyxy':[20,20,30,30],'type_predit':'crocodile','couleur_predite':'vert'},
          {'bbox_xyxy':[20,20,30,30],'type_predit':'crocodile','couleur_predite':'bleu'}]
    report=summarize_scene(gt,pred)
    assert overlap_iou([0,0,10,10],[0,0,10,10])==1
    assert report['matched']==2
    assert report['false_positives']==1
    assert report['fully_correct']==2  # one of two duplicate boxes can be matched
    assert report['scene_exact'] is False  # extra false-positive detection


def test_missed_object_counts_as_error():
    gt=[{'x1':0,'y1':0,'x2':10,'y2':10,'type':'ourson','variant':'rouge'}]
    report=summarize_scene(gt,[])
    assert report['missed']==1
    assert report['fully_correct']==0
    assert report['objects'][0]['predicted_variant']=='unknown'
