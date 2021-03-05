include (add_test_wrappers)

set(localranks_list 3)
tarantella_generate_python_gpi_test(NAME WeightsDataParallel
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/weights_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration
                                    TIMEOUT 3600)

set(localranks_list 3 4)
tarantella_generate_python_gpi_test(NAME AccuracyDataParallel
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/accuracy_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu
                                    TIMEOUT 10000)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME OptimizersDataParallelMNIST
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/optimizers_mnist_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 10000)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME OptimizersSGDDataParallelMNIST
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/optimizerSGD_momentum_mnist_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 10000)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME OptimizersDataParallelCIFAR
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/data_parallel_training/optimizers_cifar10_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster disabled
                                    TIMEOUT 10000)

tarantella_generate_python_gpi_test(NAME DistributedDatasets
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/datasets/distributed_dataset.py
                                    TIMEOUT 3600)

set(localranks_list 1 2 3 4)
tarantella_generate_python_gpi_test(NAME TensorAllreducer
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_reducer_test.py
                                    TIMEOUT 3600)