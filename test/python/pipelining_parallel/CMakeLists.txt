include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME PipelineCommunicatorPython
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/pipeline_communicator_4ranks.py
                                    LOCALRANKS_LIST "4"
                                    TIMEOUT 100)
