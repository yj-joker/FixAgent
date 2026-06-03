from services.manual_graph_ingest import classify_error


def test_classify_error():
    assert classify_error(TimeoutError("timeout")) == "failed_retryable"
    assert classify_error(ConnectionError("refused")) == "failed_retryable"
    assert classify_error(ValueError("no schema sections")) == "failed_permanent"
