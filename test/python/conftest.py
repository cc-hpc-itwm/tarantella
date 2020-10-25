import pytest
import logging
import os

@pytest.fixture(scope="session")
def tarantella_framework():
  os.environ['TF_CUDNN_DETERMINISTIC']='1'

  import tarantella
  tarantella.init()

  logging.getLogger().info("init tarantella")
  yield tarantella  # provide the fixture value
  logging.getLogger().info("teardown tarantella")
