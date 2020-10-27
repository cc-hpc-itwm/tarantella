import numpy as np
import logging
import tensorflow as tf

def compare_datasets(partition_0_dataset, dataset_reference):
  it_ref = iter(dataset_reference)
  it_part0 = iter(partition_0_dataset)
  for index, (partition0_sample_list, ref_sample_list) in \
              enumerate(zip(it_part0, it_ref)):
    assert(len(partition0_sample_list) == len(ref_sample_list))
    for partition0_sample, ref_sample in zip(partition0_sample_list, ref_sample_list):
      ds_ref = ref_sample.numpy().flatten()

      ds_partition0 = np.array([])
      for key in sorted(partition0_sample.keys()):
        elem = partition0_sample[key].numpy()
        if not 'fake' in key:
          ds_partition0 = np.append(ds_partition0, elem.flatten())
      logging.debug("[id %d] Ref: %d, Part0: %d", index, len(ds_ref), len(ds_partition0))
      logging.debug( ds_ref[0:10])
      logging.debug( ds_partition0[0:10])
      
      assert np.allclose(ds_partition0, ds_ref, atol=1e-8)

  try:
    next(it_part0)
    assert False, "Partition dataset is not fully consumed"
  except StopIteration:
    pass

  try:
    next(it_ref)
    assert False, "Reference dataset is not fully consumed"
  except StopIteration:
    pass
  logging.info("Datasets are identical")