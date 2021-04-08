include (add_test_wrappers)

set(localranks_list 1)
tarantella_generate_python_gpi_test(NAME ModelSaveLoadFile
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_save_load.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 300)

set(localranks_list 2 3 4)
tarantella_generate_python_gpi_test(NAME ModelSaveLoadFile
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_save_load.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS gpu cluster
                                    TIMEOUT 300)

set(localranks_list 1)
tarantella_generate_python_gpi_test(NAME ModelSaveLoadState
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_save_load_state.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS gpu cluster integration
                                    TIMEOUT 300)

set(localranks_list 1 2)
tarantella_generate_python_gpi_test(NAME ModelClone
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_clone.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 60)

set(localranks_list 1 2)
tarantella_generate_python_gpi_test(NAME ModelSaveLoadConfiguration
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_save_load_configuration.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 60)
