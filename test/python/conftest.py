import pytest
import logging
import os

import tensorflow as tf

@pytest.fixture(scope="session", autouse = True)
def setup_tests():
  os.environ['TF_CUDNN_DETERMINISTIC']='1'

def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "tfversion(version): test to run only on specific tf versions"
    )

def pytest_runtest_setup(item):
    supported_versions = [mark.args[0] for mark in item.iter_markers(name="tfversion")]
    if supported_versions:
      supportedv = None
      for v in supported_versions:
        if tf.__version__.startswith(v):
          supportedv = v
      if not supportedv:
        pytest.skip("Test does not support TF{}".format(tf.__version__))
