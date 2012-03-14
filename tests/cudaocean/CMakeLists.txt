CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT (TEST_GRAPHICS)

IF (USE_GLEW GREATER 0)
    SET(GL_EXT_LIB GLEW)
    ADD_DEFINITIONS(-DUSE_GLEW)
ELSE (USE_GLEW GREATER 0)
    SET(GL_EXT_LIB gl3w)
    LINK_DIRECTORIES($ENV{GL3W}/lib)
ENDIF (USE_GLEW GREATER 0)

MESSAGE ( "\nOpenGL Extension Library = ${GL_EXT_LIB}.")
MESSAGE ( "To change -DUSE_GLEW = 1 for GLEW and -DUSE_GLEW=0 for gl3w\n")

INCLUDE_DIRECTORIES($ENV{GL3W}/include
					$ENV{CS248_PROJECT}/src
					$ENV{ASSIMP_INCLUDE})


#CUDA
FIND_PACKAGE(CUDA)

LINK_DIRECTORIES(/usr/local/cuda/lib64)

CUDA_ADD_EXECUTABLE(cudaoceantest Main 
                            $ENV{CS248_PROJECT}/src/cuda/Ocean.cu
                            $ENV{CS248_PROJECT}/src/cuda/Init.cpp
                            $ENV{CS248_PROJECT}/src/Graphics
                            $ENV{CS248_PROJECT}/src/Mesh
                            $ENV{CS248_PROJECT}/src/Node
                            $ENV{CS248_PROJECT}/src/Camera
                            $ENV{CS248_PROJECT}/src/ShaderData)

TARGET_LINK_LIBRARIES(cudaoceantest 
                                    ${GL_EXT_LIB} 
                                               GL 
                                               GLU 
                                               glut 
                                               assimp
                                               freeimage
                                               cufft)

INSTALL (TARGETS cudaoceantest DESTINATION bin)