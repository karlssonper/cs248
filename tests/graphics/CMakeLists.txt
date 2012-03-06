CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT (TEST_GRAPHICS)

INCLUDE_DIRECTORIES($ENV{GL3W}/include
					$ENV{CS248_PROJECT}/src)

LINK_DIRECTORIES($ENV{GL3W}/lib)

ADD_EXECUTABLE(graphicsTest Main 
                            $ENV{CS248_PROJECT}/src/Graphics
                            $ENV{CS248_PROJECT}/src/Mesh
                            $ENV{CS248_PROJECT}/src/Node)

TARGET_LINK_LIBRARIES(graphicsTest gl3w GL GLU glut)

INSTALL (TARGETS graphicsTest DESTINATION bin)