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
  reference_model_runner = base_runner.generate_tnt_model_runner(request.param())
  yield reference_model_runner

class TestsDistributedEvaluation:
  
  out_dict = [pytest.param(True,
                           marks=[pytest.mark.tfversion('2.2'),
                                  pytest.mark.tfversion('2.3'),
                                  pytest.mark.tfversion('2.4'),]),
              pytest.param(False,
                           marks=[pytest.mark.tfversion('2.0'),
                                  pytest.mark.tfversion('2.1'),
                                  pytest.mark.tfversion('2.2'),
                                  pytest.mark.tfversion('2.3'),
                                  pytest.mark.tfversion('2.4'),]),
             ]
 
  @pytest.mark.parametrize("micro_batch_size", [32])
  @pytest.mark.parametrize("number_epochs", [6])
  @pytest.mark.parametrize("nbatches", [8])
  @pytest.mark.parametrize("test_nbatches", [0, 12])
  @pytest.mark.parametrize("extra_sample", [1, 33])
  @pytest.mark.parametrize("output_dict", out_dict)
  def test_compare_accuracy_against_reference(self, model_runners, micro_batch_size,
                                              number_epochs, nbatches, test_nbatches,
                                              extra_sample, output_dict):
    (_, test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                       micro_batch_size,
                                                       shuffle = False,
                                                       last_incomplete_batch_size = extra_sample)
    (ref_train_dataset, ref_test_dataset) = util.train_test_mnist_datasets(nbatches, test_nbatches,
                                                                           micro_batch_size,
                                                                           shuffle = False,
                                                                           last_incomplete_batch_size = extra_sample)
    
    tnt_model = model_runners
    tnt_model.train_model(ref_train_dataset, number_epochs)
    reference_model_runner = base_runner.generate_tnt_model_runner(tnt_model.model.model)
    reference_loss_accuracy = reference_model_runner.evaluate_model(test_dataset, return_dict = output_dict)
    
    tnt_loss_accuracy = tnt_model.evaluate_model(ref_test_dataset, return_dict = output_dict, distribution = True)
    
    rank = tnt.get_rank()
    logging.getLogger().info(f"[Rank {rank}] Tarantella[loss, accuracy] = {tnt_loss_accuracy}")
    logging.getLogger().info(f"[Rank {rank}] Reference [loss, accuracy] = {reference_loss_accuracy}")
    
    if not output_dict:
      assert np.isclose(tnt_loss_accuracy[0], reference_loss_accuracy[0], atol=1e-2)
      assert np.isclose(tnt_loss_accuracy[1], reference_loss_accuracy[1], atol=1e-6)
    else:
      assert np.isclose(tnt_loss_accuracy['loss'], reference_loss_accuracy['loss'], atol=1e-2)
      assert np.isclose(tnt_loss_accuracy['sparse_categorical_accuracy'], 
                        reference_loss_accuracy['sparse_categorical_accuracy'], atol=1e-6)
