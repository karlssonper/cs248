PROJECT (TEST_CUDAOCEAN)


CUDA_ADD_EXECUTABLE(cudaoceantest Main 
                            $ENV{CS248_PROJECT}/src/cuda/Ocean.cu
                            $ENV{CS248_PROJECT}/src/CUDA
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