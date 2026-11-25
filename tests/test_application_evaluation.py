import pytest
from src.evaluation.application import parse_worker_output


def test_worker_protocol_accepts_valid_results():
    value='READY\nINFO:ok\nRESULT:/tmp/image_annote.jpg\nDONE\n'
    assert parse_worker_output(value,1)==['/tmp/image_annote.jpg']


def test_worker_protocol_rejects_missing_result():
    with pytest.raises(RuntimeError):
        parse_worker_output('READY\nDONE\n',1)
