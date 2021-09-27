from models import mnist_models as mnist
import training_runner as base_runner
import utilities as util
import tarantella as tnt
import tensorflow as tf

import numpy as np

import logging
import pytest

@pytest.fixture(scope="function", params=[mnist.fc_model_generator,
                                          mnist.lenet5_model_generator,
                                          mnist.sequential_model_generator,
                                          mnist.subclassed_model_generator
                                         ])

def model_runners(request):
  reference_model_runner = base_runner.TrainingRunner(request.param())
  yield reference_model_runner

class TestsDistributedPrediction:
 
  @pytest.mark.parametrize("micro_batch_size", [4])
  @pytest.mark.parametrize("number_epochs", [6])
  @pytest.mark.parametrize("nbatches", [8])
  @pytest.mark.parametrize("test_nbatches", [0, 1, 5])
  @pytest.mark.parametrize("extra_sample", [1, 3, 22])
  def test_compare_accuracy_against_reference(self, model_runners, micro_batch_size,
                                              number_epochs, nbatches, test_nbatches,
                                              extra_sample):
    (_, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                       micro_batch_size,
                                                       shuffle = False,
                                                       last_incomplete_batch_size = extra_sample)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size,
                                                                           shuffle = False,
                                                                           last_incomplete_batch_size = extra_sample)
    
    reference_model_runner = model_runners
    reference_model_runner.train_model(ref_train_dataset, number_epochs)
    reference_predict_result = reference_model_runner.predict_model(ref_test_dataset)

    tnt_model = base_runner.generate_tnt_model_runner(reference_model_runner.model)
    tnt_predict_result = tnt_model.predict_model(test_dataset, distribution = True)
    
    rank = tnt.get_rank()

    if rank == tnt.get_master_rank():
      assert len(reference_predict_result) == len(tnt_predict_result)
      #Since test dataset is reordered, check if each ref output occurred in the tnt output
      for index in range(len(tnt_predict_result)):
        assert np.isclose(reference_predict_result[index], tnt_predict_result, atol=1e-6).all(1).any()
    else:
      assert len(tnt_predict_result) == 0
