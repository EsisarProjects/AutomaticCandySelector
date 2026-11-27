"""The independent type evaluator is importable and rejects invalid splits."""
import pytest
from src.evaluation.classification import evaluate_split


def test_independent_evaluation_rejects_training_split():
    with pytest.raises(ValueError, match='Only val and held-out test'):
        evaluate_split('train')
