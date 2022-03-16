import enum

class ParallelStrategy(enum.Flag):
  NONE = enum.auto()
  DATA = enum.auto()
  PIPELINING = enum.auto()
  ALL = DATA | PIPELINING
