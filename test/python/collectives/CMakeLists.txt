include (add_test_wrappers)

set(localranks_list 1 2 3 4)
tarantella_generate_python_gpi_test(NAME TensorAllreducer
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_allreducer_test.py
                                    TIMEOUT 100)
tarantella_generate_python_gpi_test(NAME TensorBroadcaster
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcaster_test.py
                                    TIMEOUT 100)
