
class ConnectionInfo:
  def __init__(self, rank_pair, size_in_bytes):
    self.rank0, self.rank1 = rank_pair
    self.size_in_bytes = size_in_bytes

  def contains_rank(self, rank):
    return rank in [self.rank0, self.rank1]

  def get_other_rank(self, rank):
    if not self.contains_rank(rank):
      raise ValueError(f"[ConnectionInfo][get_other_rank] Requested rank {rank} does not belong to"
                       f"this connection ({self.rank0},{self.rank1}).")
    return self.rank0 if rank == self.rank1 else self.rank1

  def get_size_in_bytes(self):
    return self.size_in_bytes
