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
    
    self.special_global_batch_size = None
    self.special_my_size = None
    self.normal_factor = None

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

  def pad_dataset(self,dataset,batch_size,comm_size,num_samples):
    real_batch_size = int(batch_size//comm_size) * comm_size
    #num_samples = 23 new_batch_size = 3*3 = 9
    #num_padded = 9-(23 - 2 * 9) = 4
    #take last 2*9 - 5 element as rest_dataset, rest_dataset has 14 elements
    #first batch rest_dataset with batch_size 9,rest_dataset would contain 2 part. First with 9 element, second with 5    elements
    #Then padded batch with batch_size 2 without passing padded shape,
    #All dimensions of all components are padded to the maximum size in the batch.
    #Now the size of rest_dataset is 1*2*9,with 2 unbatch,the size of rest_dataset is 18 with padded 0
    #concat two dataset, now the size of dataset is 9 + 18 = 27,which is multiple of new_batch_size.
    #batch the dataset with micro_batchsize 3,the shape of dataset is 9*3,then shard with 3 nodes,each node has dataset with shape 3*3
    if num_samples % real_batch_size != 0:
      num_padded = num_samples - int(num_samples // real_batch_size)*real_batch_size
      self.special_global_batch_size = real_batch_size
      num_padded = real_batch_size - num_padded
    
      zero_elem = num_padded//self.num_ranks
      module = num_padded%self.num_ranks
      if module >= self.rank + 1:
        zero_elem = zero_elem + 1
      #get my true micro_size that is not zero.
      self.special_my_size = self.micro_batch_size - zero_elem
    
      rest_dataset = dataset.take(2*real_batch_size - num_padded)
      logger.info("Dataset is padded with {} elements.".format(
                num_padded))
      rest_dataset = rest_dataset.batch(real_batch_size,drop_remainder=False)

      #If padded_shape is unset, all dimensions of all components are padded to the maximum size in the batch.
      rest_dataset = rest_dataset.padded_batch(2)
      rest_dataset = rest_dataset.unbatch()
      rest_dataset = rest_dataset.unbatch()

      ##take previous and concat togther with rest_dataset
      rest_dataset = rest_dataset.skip(2*real_batch_size - num_padded)
      dataset = rest_dataset.concatenate(dataset)
    return dataset

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
        dataset = self.pad_dataset(dataset,new_batch_size,self.num_ranks,num_samples)
      else:
        if num_samples % new_batch_size != 0:
          num_samples_multiple = (num_samples // new_batch_size) * new_batch_size
          logger.warn("Number of samples ({}) is not a multiple of batch size.\
 Removing the last incomplete batch from the dataset. Now dataset has {} samples.".format(num_samples,num_samples_multiple))
          dataset = dataset.take(num_samples_multiple)
    
    if self.diff_micro_batch:
      print("use dataset.window(),micro_batch_size is",self.micro_batch_size)
      #get scaling factor for diff micro_batch_size
      self.normal_factor = self.micro_batch_size * self.num_ranks / batch_size
      ##Assume that we have 3 rank with global batch size 7. And the dataset has 14 elements.(num_sample must be multiple of batch size here)
      ##Rank 0 has micro_batch_size 3, rank 1 and 2 has  micro_batch_size 2.
      ##So rank 0 need 1,4,7, 8,11,14th elements in dataset. rank 1 needs 2,5, 9,15th elements in dataset.
      ##rank 2 need 3,6, 10,13th elements in datasets.
      ##with windows,rank 1 and rank 2 should skip 1 and 2 element repectively.
      ##As for rank 0, size = 3, shift = 7, stride = 3. And it start from the first element. So it can first take size = 3 element with stride 3 so it get 1,4,7. Then it shift to the 1 + 7 = 8th elements, And again take take size = 3 element with stride 3 so it get 8,11,14.
      if self.rank != 0:
        dataset = dataset.skip(self.rank)
      dataset = dataset.window(size = self.micro_batch_size,shift = batch_size,stride = self.num_ranks,drop_remainder = False)
      def create_seqeunce_ds(*param):
        #For case that param has only one dataset
        if len(param) == 1 and not isinstance(param[0],tuple):
          return self.batching_info.apply(param[0], new_batch_size = self.micro_batch_size)
        result = []
        #Param contain multiple element
        for pa in param:
          #if a element is a tuple, each element in tuple is a dataset
          if isinstance(pa,tuple):
            temp = []
            for p in pa:
              temp.append(self.batching_info.apply(p, new_batch_size = self.micro_batch_size))
            result.append(tuple(temp))
          ##otherwise, this is a dataset
          else:
            result.append(self.batching_info.apply(pa, new_batch_size = self.micro_batch_size))
        #return result after zip.
        return tf.data.Dataset.zip(tuple(result))
    
      dataset = dataset.interleave(create_seqeunce_ds,cycle_length = 8,num_parallel_calls = 8)
    else:
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
  ##generate a callback if it necessary
  def generate_callback_if_have(self):
    if self.normal_factor == None and self.special_global_batch_size == None:
      return None
    if self.normal_factor == None:
      self.normal_factor = 1.0
      return ds_helpers.ScalingFactorScheduler(self.normal_factor,
                                     self.special_my_size * self.num_ranks / self.special_global_batch_size,
                                     0)
    else:
      return ds_helpers.ScalingFactorScheduler(self.normal_factor)