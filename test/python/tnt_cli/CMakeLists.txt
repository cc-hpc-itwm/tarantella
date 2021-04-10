include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME CLIEnvironmentVariables
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/user_env_vars_test.py
                                    TIMEOUT 20)

tarantella_generate_python_gpi_test(NAME CLIRankPinning
                                    LOCALRANKS_LIST "1"
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/rank_pinning_tests.py
                                    TIMEOUT 20)
