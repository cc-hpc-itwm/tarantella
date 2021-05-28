include (add_test_wrappers)

set(localranks_list 2)
tarantella_generate_python_gpi_test(NAME DataParallel_Training_Weights_Adam
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/weights_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration
                                    TIMEOUT 200)

set(localranks_list 2)
tarantella_generate_python_gpi_test(NAME DataParallel_Training_Accuracy_Adam
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/accuracy_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 200)

set(localranks_list 2)
tarantella_generate_python_gpi_test(NAME DataParallel_Training_Accuracy_Adam_Any_Batchsize
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/accuracy_test_any_batch_size.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 400)

set(localranks_list 2)
tarantella_generate_python_gpi_test(NAME DataParallel_Training_Accuracy_Optimizers
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/optimizers_mnist_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration gpu cluster
                                    TIMEOUT 400)

set(localranks_list 2)
tarantella_generate_python_gpi_test(NAME DataParallel_Custom_Training_Loop
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/custom_training_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 40)
