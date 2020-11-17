from abc import ABCMeta, abstractmethod
import os
import stat
import tempfile

def make_executable(filename):
  os.chmod(filename, os.stat(filename).st_mode | stat.S_IXUSR)

class TemporaryFileWrapper(metaclass = ABCMeta):
  def __init__(self, dir = None, is_executable = False):
    self.is_executable = is_executable
    self.dir = dir

  def __enter__(self):
    self.file_handle, self.filename = tempfile.mkstemp(dir = self.dir)

    with os.fdopen(self.file_handle, 'w') as f:
      contents = self.get_initial_contents()
      f.write(str(contents))

    if self.is_executable:
      make_executable(self.filename)

  def __exit__(self, *args):
    os.remove(self.filename)

  @abstractmethod
  def get_initial_contents(self):
    raise NotImplementedError

  @property
  def name(self):
    return self.filename


class HostFile(TemporaryFileWrapper):
  def __init__(self, nodes, devices_per_node):
    super().__init__(is_executable = False)

    if not isinstance(nodes, list) or len(nodes) == 0:
      raise LogicError("[create_nodes_file] Empty list of nodes provided")
    if devices_per_node is None or devices_per_node <= 0:
      raise LogicError("[create_nodes_file] Incorrect number of `devices_per_node`")
    self.nodes = sorted(nodes)
    self.devices_per_node = devices_per_node

  def get_initial_contents(self):
    contents = ""
    for node in self.nodes:
      contents += '\n'.join([node] * self.devices_per_node) + '\n'
    return contents

class GPIScriptFile(TemporaryFileWrapper):
  def __init__(self, header, environment, command, dir):
    super().__init__(dir = dir, is_executable = True)
    self.contents = [header,
                     environment,
                     command]

  def get_initial_contents(self):
    return '\n'.join(self.contents)
