
from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt

import copy
import numpy as np
import os
import pytest
import shutil

@pytest.fixture(scope="function")
def setup_save_path(request):
  barrier = tnt.Barrier()
  barrier.execute()
  # save logs in a shared directory accessible to all ranks
  save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_callbacks")
  if tnt.is_master_rank():
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
  barrier.execute()
  yield save_dir

  # clean up
  barrier.execute()
  if tnt.is_master_rank():
    shutil.rmtree(save_dir, ignore_errors=True)


def gen_model_runners(model_config: base_runner.ModelConfig):
  tnt_model_runner = base_runner.generate_tnt_model_runner(model_config)
  reference_model_runner = base_runner.TrainingRunner(model_config.model_generator())
  return tnt_model_runner, reference_model_runner

def train_val_dataset_generator():
  micro_batch_size = 64
  nbatches = 1
  batch_size = micro_batch_size * tnt.get_size()
  nsamples = nbatches * batch_size
  train_dataset, val_dataset, _ = util.load_dataset(mnist.load_mnist_dataset,
                                                    train_size = nsamples,
                                                    train_batch_size = batch_size,
                                                    val_size = nsamples,
                                                    val_batch_size = batch_size)
  return train_dataset, val_dataset

def train_tnt_and_ref_models_with_callbacks(callbacks, model_config, number_epochs):
  (train_dataset, val_dataset) = train_val_dataset_generator()
  (ref_train_dataset, ref_val_dataset) = train_val_dataset_generator()

  tnt_model_runner, reference_model_runner = gen_model_runners(model_config)

  param_dict = { 'epochs' : number_epochs,
                  'verbose' : 0,
                  'shuffle' : False,
                  'callbacks' : copy.deepcopy(callbacks) }
  tnt_history = tnt_model_runner.model.fit(train_dataset,
                                            validation_data=val_dataset,
                                            **param_dict)
  ref_history = reference_model_runner.model.fit(ref_train_dataset,
                                                 validation_data=ref_val_dataset,
                                                 **param_dict)
  return (tnt_history, ref_history)

def assert_identical_tnt_and_ref_history(tnt_history, ref_history):
  result = [True]
  if tnt.is_master_rank():
    for key in ref_history.history.keys():
      result += [all(np.isclose(tnt_history.history[key], ref_history.history[key], atol=1e-6))]
    result = [all(result)]
  util.assert_on_all_ranks(result)
