PROJECT (TEST_PARTICLESYSTEM)

CUDA_ADD_EXECUTABLE(particleTest ParticleTest                           
                            $ENV{CS248_PROJECT}/src/Engine
                            $ENV{CS248_PROJECT}/src/cuda/Ocean.cu
                            $ENV{CS248_PROJECT}/src/cuda/Emitter.cu
                            $ENV{CS248_PROJECT}/src/CUDA
                            $ENV{CS248_PROJECT}/src/Graphics
                            $ENV{CS248_PROJECT}/src/Mesh
                            $ENV{CS248_PROJECT}/src/Node
                            $ENV{CS248_PROJECT}/src/Camera
                            $ENV{CS248_PROJECT}/src/ShaderData
                            $ENV{CS248_PROJECT}/src/ParticleSystem
                            $ENV{CS248_PROJECT}/src/Weapon
                            $ENV{CS248_PROJECT}/src/Target
                            $ENV{CS248_PROJECT}/src/HitBox
                            $ENV{CS248_PROJECT}/src/Sound
                            )

TARGET_LINK_LIBRARIES(particleTest ${GL_EXT_LIB} GL GLU glut assimp freeimage openal alut cufft cudart)

INSTALL (TARGETS particleTest DESTINATION bin)