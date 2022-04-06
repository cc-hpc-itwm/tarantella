include (add_test_wrappers)

set(localranks_list 1 3)
tarantella_generate_python_gpi_test(NAME DataParallel_TF_Model
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model_api.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 200)

add_subdirectory(model_api)
add_subdirectory(callbacks)
add_subdirectory(data)
add_subdirectory(data_parallel_training)
add_subdirectory(pipelining_parallel)
add_subdirectory(collectives)
add_subdirectory(tnt_cli)
