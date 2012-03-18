PROJECT (TEST_ENGINE)


CUDA_ADD_EXECUTABLE(enginetest Main)

TARGET_LINK_LIBRARIES(enginetest 
                                    ${ENGINE_STATIC_LIB}
                                    ${GL_EXT_LIB} 
                                    GL 
                                    GLU 
                                    glut 
                                    assimp
                                    freeimage
                                    cudart
                                    cufft
                                    openal
                                    alut
                                     )

INSTALL (TARGETS enginetest DESTINATION bin)