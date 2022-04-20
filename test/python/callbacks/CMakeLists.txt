include (add_test_wrappers)

tarantella_generate_python_gpi_test(NAME TNT_Callbacks_Keras
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/keras_callbacks.py
                                    LOCALRANKS_LIST "1"
                                    LABELS integration
                                    TIMEOUT 300)
tarantella_generate_python_gpi_test(NAME TNT_Callbacks_Keras
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/keras_callbacks.py
                                    LOCALRANKS_LIST "4"
                                    LABELS integration cluster
                                    TIMEOUT 600)

set(localranks_list 4)
tarantella_generate_python_gpi_test(NAME TNT_Callbacks_Tensorboard
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tensorboard_callback.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 200)


set(localranks_list 1 4)
tarantella_generate_python_gpi_test(NAME TNT_Callbacks_ModelCheckpoint
                                    TEST_FILE ${CMAKE_CURRENT_SOURCE_DIR}/modelcheckpoint_callback.py
                                    LOCALRANKS_LIST "${localranks_list}"
                                    TIMEOUT 200)

