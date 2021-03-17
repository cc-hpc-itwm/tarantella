import GPICommLib

class Barrier():
  def __init__(self):
    self.barrier = GPICommLib.Barrier()

  def synchronize(self):
    self.barrier.blocking_barrier_all_ranks()
