include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME DataParallel_TF_Datasets
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/distributed_dataset.py
                                    TIMEOUT 200)
