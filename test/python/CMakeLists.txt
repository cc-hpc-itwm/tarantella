include (add_test_wrappers)

set(localranks_list 3)
tarantella_generate_python_gpi_test(NAME WeightsDataParallel
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/weights_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration
                                    TIMEOUT 1000)

set(localranks_list 3 4)
tarantella_generate_python_gpi_test(NAME AccuracyDataParallel
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/accuracy_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu
                                    TIMEOUT 600)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME OptimizersDataParallelMNIST
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/optimizers_mnist_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 600)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME OptimizersSGDDataParallelMNIST
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/optimizerSGD_momentum_mnist_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 600)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME OptimizersDataParallelCIFAR
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/optimizers_cifar10_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster disabled
                                    TIMEOUT 10000)

tarantella_generate_python_gpi_test(NAME DistributedDatasets
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/datasets/distributed_dataset.py
                                    TIMEOUT 200)

set(localranks_list 1 2 3)
tarantella_generate_python_gpi_test(NAME ModelAPIMNIST
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_api_mnist.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 60)

set(localranks_list 1 2 3 4)
tarantella_generate_python_gpi_test(NAME TensorAllreducer
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_reducer_test.py
                                    TIMEOUT 100)
tarantella_generate_python_gpi_test(NAME TensorBroadcaster
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcaster_test.py
                                    TIMEOUT 100)

set(localranks_list 1 2 3 4)
tarantella_generate_python_gpi_test(NAME CallbacksDataParallel
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/callbacks_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu
                                    TIMEOUT 100)

add_subdirectory(pipelining_parallel)
add_subdirectory(tnt_cli)

