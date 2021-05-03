include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME Pipeline_Communicator
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_communicator_4ranks.py
                                    LOCALRANKS_LIST "4"
                                    LABELS cluster
                                    TIMEOUT 100)

tarantella_generate_python_gpi_test(NAME Pipeline_SendRecvLayers
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/send_recv_layers_2ranks.py
                                    LOCALRANKS_LIST "2"
                                    TIMEOUT 100)

tarantella_generate_python_gpi_test(NAME Pipeline_HardcodedPartitions
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_simple_model_hardcoded_2ranks.py
                                    LOCALRANKS_LIST "2"
                                    LABELS integration cluster
                                    TIMEOUT 100)

tarantella_generate_python_gpi_test(NAME Pipeline_HardcodedPartitions_AutoMicrobatching
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_simple_model_2ranks.py
                                    LOCALRANKS_LIST "2"
                                    LABELS integration cluster
                                    TIMEOUT 200)

tarantella_generate_python_gpi_test(NAME Pipeline_Partition_Generator
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/partition_generator.py
                                    LOCALRANKS_LIST "1"
                                    LABELS integration cluster
                                    TIMEOUT 100)
