from src.evaluation.variant_metrics import save_variant_report, variant_names


def test_perfect_22_variant_report_is_not_penalized_by_unknown_class(tmp_path):
    labels=variant_names()
    result=save_variant_report(labels,labels,tmp_path/'perfect',split='test')
    assert result['accuracy']==1.0
    assert result['macro_f1']==1.0
    assert result['unknown_rate']==0.0


def test_abstentions_count_as_incorrect(tmp_path):
    labels=variant_names()
    pred=labels.copy(); pred[0]='unknown'
    result=save_variant_report(labels,pred,tmp_path/'abstained',split='val')
    assert result['accuracy']<1.0
    assert result['unknown_predictions']==1
