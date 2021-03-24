include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME PipelineCommunicatorPython
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_communicator_4ranks.py
                                    LOCALRANKS_LIST "4"
                                    TIMEOUT 100)

tarantella_generate_python_gpi_test(NAME SendRecvLayers
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/send_recv_layers_2ranks.py
                                    LOCALRANKS_LIST "2"
                                    TIMEOUT 100)

tarantella_generate_python_gpi_test(NAME PipelineSimpleModel
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_simple_model_2ranks.py
                                    LOCALRANKS_LIST "2"
                                    LABELS integration cluster
                                    TIMEOUT 100)
