import pytest
import logging
import os

import tensorflow as tf
import tarantella.utilities.tf_version as utils

@pytest.fixture(scope="session", autouse = True)
def setup_tests():
  os.environ['TF_CUDNN_DETERMINISTIC']='1'

def pytest_configure(config):
  # register additional markers
  config.addinivalue_line(
      "markers", "tfversion(version): run test only on specific tf versions")
  config.addinivalue_line(
      "markers", "min_tfversion(version): run test only when TF version is at least the one specified")
  config.addinivalue_line(
      "markers", "max_tfversion(version): run test only when TF version is at most the one specified")

def get_marker_values(item, marker_name):
  # Example:
  # @pytest.mark.`marker_name`('2.2')
  # @pytest.mark.`marker_name`(['2.3', '2.4'])
  supported_versions = list()
  version_markers = [mark.args[0] for mark in item.iter_markers(name = marker_name)]
  for versions in version_markers:
    if not isinstance(versions, list):
      versions = [versions]
    supported_versions += versions
  return supported_versions

def check_tfversion_marker(item):
  supported_versions = get_marker_values(item, 'tfversion')
  if supported_versions:
    return utils.tf_version_in_list(supported_versions)
  else: # marker not specified
    return True

def check_min_tfversion_marker(item):
  supported_versions = get_marker_values(item, 'min_tfversion')
  if len(supported_versions) > 1:
    raise ValueError(f"Minimum version specified incorrectly as `{supported_versions}`, "
                      "it should be a single version number")
  if supported_versions:
    return utils.tf_version_above_equal(supported_versions[0])
  else: # marker not specified
    return True

def check_max_tfversion_marker(item):
  supported_versions = get_marker_values(item, 'max_tfversion')
  if len(supported_versions) > 1:
    raise ValueError(f"Maximum version specified incorrectly as `{supported_versions}`, "
                      "it should be a single version number")
  if supported_versions:
    return utils.tf_version_below_equal(supported_versions[0])
  else: # marker not specified
    return True

def pytest_runtest_setup(item):
  if not (check_tfversion_marker(item) and \
          check_min_tfversion_marker(item) and \
          check_max_tfversion_marker(item)):
    pytest.skip("Test does not support TF{}".format(tf.__version__))
