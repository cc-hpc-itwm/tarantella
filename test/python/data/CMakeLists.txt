include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME DataParallel_TF_Datasets_Transformations
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/transformations.py
                                    TIMEOUT 1500)

tarantella_generate_python_gpi_test(NAME DataParallel_TF_Datasets_Microbatching
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/distributed_micro_batching.py
                                    TIMEOUT 1500)

