import tensorflow as tf

def tf_version_has_minor(version_string):
  vlist = version_string.split('.')
  return len(vlist) >= 3

def current_version():
  return tf.__version__

def version_as_list(version_string):
  return [int(v) for v in version_string.split('.')]

def tf_version_equal(version_string):
  if tf_version_has_minor(version_string):
    return version_string == current_version()
  return current_version().startswith(version_string)

def tf_version_below_equal(version_string):
  vlist = version_as_list(version_string)
  tf_vlist = version_as_list(current_version())
  return all([tf_version <= max_version for max_version, tf_version in zip(vlist, tf_vlist)])

def tf_version_above_equal(version_string):
  vlist = version_as_list(version_string)
  tf_vlist = version_as_list(current_version())
  return all([tf_version >= min_version for min_version, tf_version in zip(vlist, tf_vlist)])

def tf_version_in_range(start_version, end_version):
  return tf_version_above_equal(start_version) and tf_version_below_equal(end_version)

def tf_version_in_list(supported_versions):
  assert isinstance(supported_versions, list)
  return any([tf_version_equal(v) for v in supported_versions])

