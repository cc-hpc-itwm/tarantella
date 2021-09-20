include (add_test_wrappers)

set(localranks_list 1 3)
tarantella_generate_python_gpi_test(NAME Collectives_TensorAllreducer
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_allreducer_test.py
                                    TIMEOUT 100)
tarantella_generate_python_gpi_test(NAME Collectives_TensorBroadcaster
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcaster_test.py
                                    TIMEOUT 100)
tarantella_generate_python_gpi_test(NAME Collectives_TensorAllgatherver
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_allgatherver_test.py
                                    TIMEOUT 100)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME Collectives_TensorAllreducer
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_allreducer_test.py
                                    LABELS cluster
                                    TIMEOUT 100)
tarantella_generate_python_gpi_test(NAME Collectives_TensorBroadcaster
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_broadcaster_test.py
                                    LABELS cluster
                                    TIMEOUT 100)
tarantella_generate_python_gpi_test(NAME Collectives_TensorAllgatherver
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_allgatherver_test.py
                                    LABELS cluster
                                    TIMEOUT 100)
