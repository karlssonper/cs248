PROJECT (TEST_GRAPHICS)


CUDA_ADD_EXECUTABLE(graphicsTest Main 
                            $ENV{CS248_PROJECT}/src/Graphics
                            $ENV{CS248_PROJECT}/src/Mesh
                            $ENV{CS248_PROJECT}/src/Node
                            $ENV{CS248_PROJECT}/src/Camera
                            $ENV{CS248_PROJECT}/src/ShaderData
                            $ENV{CS248_PROJECT}/src/CUDA)

TARGET_LINK_LIBRARIES(graphicsTest ${GL_EXT_LIB} GL GLU glut assimp freeimage)

INSTALL (TARGETS graphicsTest DESTINATION bin)