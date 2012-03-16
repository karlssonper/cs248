PROJECT (TEST_CUDAOCEAN)


CUDA_ADD_EXECUTABLE(cudaoceantest Main)

TARGET_LINK_LIBRARIES(cudaoceantest 
                                    ${ENGINE_STATIC_LIB}
                                    ${GL_EXT_LIB} 
                                               GL 
                                               GLU 
                                               glut 
                                               assimp
                                               freeimage
                                               cudart
                                               cufft)

INSTALL (TARGETS cudaoceantest DESTINATION bin)