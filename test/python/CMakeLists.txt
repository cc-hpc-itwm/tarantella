include (add_test_wrappers)

set(localranks_list 1 4)
tarantella_generate_python_gpi_test(NAME DataParallel_TF_Callbacks
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/callbacks_test.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    LABELS integration
                                    TIMEOUT 300)

set(localranks_list 1 3)
tarantella_generate_python_gpi_test(NAME DataParallel_TF_Model
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_api.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 60)

add_subdirectory(model_api)
add_subdirectory(datasets)
add_subdirectory(data_parallel_training)
add_subdirectory(pipelining_parallel)
add_subdirectory(collectives)
add_subdirectory(tnt_cli)
