PROJECT (ENGINE_LIB)


CUDA_ADD_LIBRARY(engine STATIC
                            Engine 
                            cuda/Ocean.cu
                            cuda/Emitter.cu
                            CUDA
                            Graphics
                            Mesh
                            Node
                            Camera
                            ShaderData
                            ParticleSystem)

TARGET_LINK_LIBRARIES(engine 
                                    ${GL_EXT_LIB} 
                                               GL 
                                               GLU 
                                               glut 
                                               assimp
                                               freeimage
                                               cudart
                                               cufft)

INSTALL (TARGETS engine DESTINATION lib)

SET(engine ENGINE_LIB)