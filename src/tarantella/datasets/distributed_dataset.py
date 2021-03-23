import tensorflow as tf
from tensorflow.python.data.ops import dataset_ops as ds

from tarantella import logger
import tarantella.datasets.dataset_helpers as ds_helpers

class DistributedDataset:
  def __init__(self, dataset, num_ranks, rank, shuffle_seed = 42):
    self.num_ranks = num_ranks
    self.rank = rank
    self.shuffle_seed = shuffle_seed

    self.dataset = dataset
    self.base_dataset, self.dataset_transformations = \
           ds_helpers.gen_dataset_transformations(dataset)
    self.batching_info = ds_helpers.get_batching_info(self.dataset_transformations)
    self.diff_micro_batch = False
    self.micro_batch_size = None

  def distribute_dataset_across_ranks(self, user_micro_batch_size = None, is_training = True):
    dataset = self.base_dataset

    # Batched datsets:
    # re-apply dataset transformations identically, except for batching & shuffling
    for index, (transf, ds_kwargs) in enumerate(self.dataset_transformations):
      # shuffle operation
      if isinstance(transf(dataset, **ds_kwargs), ds.ShuffleDataset):
        dataset = self.shuffle_with_seed(dataset, ds_kwargs)

      # batch operation (i.e., `batch` or `padded_batch`)
      elif self.batching_info.is_last_batching_transformation(index):
        batch_size = self.batching_info.batch_size
        if user_micro_batch_size:
          micro_batch_size = user_micro_batch_size
          if micro_batch_size * self.num_ranks != batch_size:
            raise ValueError("[DistributedDataset] micro batch size ({}) is not consistent \
with batch size ({}) on number of devices used ({}).".format(micro_batch_size, batch_size,
                                                            self.num_ranks))
        else:
          micro_batch_size = self.get_microbatch_size(batch_size)
        if is_training:
          dataset = self.distributed_batch(dataset,
                                           batch_size = batch_size,
                                           micro_batch_size = micro_batch_size)
        else:
          # FIXME: distribute batch for `evaluate` and `predict`
          dataset = self.batching_info.apply(dataset, new_batch_size = micro_batch_size)

      # other operations
      else:
        dataset = transf(dataset, **ds_kwargs)

    # Unbatched datasets
    if self.batching_info.is_batched == False:
      if is_training == False:    # outside `fit`
        if user_micro_batch_size:
          dataset = self.batching_info.apply(dataset, new_batch_size = micro_batch_size)
        else:
          dataset = self.batching_info.apply(dataset, new_batch_size = 1)

      if is_training == True:     # inside `fit`
        if user_micro_batch_size:
          micro_batch_size = user_micro_batch_size
          batch_size = micro_batch_size * self.num_ranks
          dataset = self.distributed_batch(dataset,
                                          batch_size = batch_size,
                                          micro_batch_size = micro_batch_size)
        else:
          raise ValueError("[DistributedDataset] Unbatched datasets without tnt_micro_batch_size are not supported")

    return dataset

  def shuffle_with_seed(self, dataset, ds_kwargs):
    if not 'seed' in ds_kwargs or ds_kwargs['seed'] is None:
      logger.warn("Shuffling with fixed shuffle seed {}.".format(self.shuffle_seed))
      ds_kwargs['seed'] = self.shuffle_seed
    else:
      logger.debug("Shuffling with shuffle seed {}.".format(ds_kwargs['seed']))
    return dataset.shuffle(**ds_kwargs)

  def distributed_batch(self, dataset, batch_size, micro_batch_size):
    if self.batching_info.drop_remainder == True:
      dataset = self.batching_info.apply(dataset, new_batch_size = batch_size)
      dataset = dataset.unbatch()
    else:
      num_samples = ds_helpers.get_num_samples(dataset)
      if num_samples == tf.data.experimental.INFINITE_CARDINALITY:
        raise ValueError("[DistributedDataset] Infinite dataset provided")
      #get large batch, check is num_sample is multiple of new_batch_size
      new_batch_size = batch_size
#       print("dist num_samples is : new_batch_size is ",num_samples," ",new_batch_size)
      if tf.version.VERSION >= "2.2.0":
        #Always pad for version greater than 2.2.0
        dataset = ds_helpers.pad_dataset(dataset,new_batch_size,self.num_ranks,num_samples)
      else:
        if num_samples % new_batch_size != 0:
          num_samples_multiple = (num_samples // new_batch_size) * new_batch_size
          logger.warn("Number of samples ({}) is not a multiple of batch size.\
 Removing the last incomplete batch from the dataset. Now dataset has {} samples.".format(num_samples,num_samples_multiple))
          dataset = dataset.take(num_samples_multiple)
    
    if self.diff_micro_batch:
      print("use dataset.window(),micro_batch_size is",self.micro_batch_size)
      dataset = dataset.skip(self.rank)
      dataset = dataset.window(size = self.micro_batch_size,shift = batch_size,stride = self.num_ranks,drop_remainder = False)
      def create_seqeunce_ds(a,b):
        return tf.data.Dataset.zip((a.batch(self.micro_batch_size, drop_remainder=False),b.batch(self.micro_batch_size, drop_remainder=False)))
      dataset = dataset.flat_map(create_seqeunce_ds)
    else:
      print("here")
      dataset = dataset.shard(num_shards=self.num_ranks, index = self.rank)
      dataset = self.batching_info.apply(dataset, new_batch_size = self.micro_batch_size)
    
    logger.info("Using batch size = {}, micro batch size = {}.".format(
                batch_size, micro_batch_size))
    return dataset

  def get_microbatch_size(self, batch_size):
    if batch_size is None or batch_size == 0:
      raise ValueError("[DistributedDataset]Incorrectly defined batch size")
    
    self.micro_batch_size = int(batch_size // self.num_ranks)
    
    remain_element = batch_size % self.num_ranks
    
    if remain_element != 0:
      logger.debug("Batch size ({}) is a not multiple of the number of ranks {}.".format(
                 batch_size, self.num_ranks))
      self.diff_micro_batch = True
      if self.rank + 1 <= remain_element:
        self.micro_batch_size = self.micro_batch_size + 1
      
    logger.debug("Rank {} has micro batch {}.".format(
                 self.rank, self.micro_batch_size))
    return self.micro_batch_size
