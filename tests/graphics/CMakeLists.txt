PROJECT (TEST_GRAPHICS)


CUDA_ADD_EXECUTABLE(graphicsTest Main )

TARGET_LINK_LIBRARIES(graphicsTest 
                                     ${ENGINE_STATIC_LIB}
                                     ${GL_EXT_LIB} 
                                     GL 
                                     GLU 
                                     glut 
                                     assimp
                                     cudart 
                                     freeimage)

INSTALL (TARGETS graphicsTest DESTINATION bin)