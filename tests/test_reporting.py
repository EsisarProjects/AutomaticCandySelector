import json
from pathlib import Path
from src.evaluation.reporting import save_history, save_report


def test_report_outputs(tmp_path: Path):
    output = tmp_path/'val'
    result = save_report([0, 1, 1, 0], [0, 1, 0, 0], ['a', 'b'], output,
                         split='val', files=['1.jpg', '2.jpg', '3.jpg', '4.jpg'])
    assert result['accuracy'] == .75
    assert result['samples'] == 4
    assert result['confusion_matrix'] == [[2, 0], [1, 1]]
    assert json.loads((output/'metrics.json').read_text())['split'] == 'val'
    for filename in ['confusion_matrix.png', 'confusion_matrix.csv',
                     'per_class_metrics.csv', 'predictions.csv']:
        assert (output/filename).is_file()


def test_history_outputs(tmp_path: Path):
    history = [dict(epoch=1, train_loss=2., val_loss=2.5, train_accuracy=.2, val_accuracy=.1),
               dict(epoch=2, train_loss=1., val_loss=1.3, train_accuracy=.6, val_accuracy=.5)]
    save_history(history, tmp_path)
    for filename in ['history.csv', 'loss_curve.png', 'accuracy_curve.png']:
        assert (tmp_path/filename).is_file()
