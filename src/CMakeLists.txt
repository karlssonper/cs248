PROJECT (ENGINE_LIB)


CUDA_ADD_EXECUTABLE (main Main
                            Engine 
                            cuda/Ocean.cu
                            cuda/Emitter.cu
                            CUDA
                            Graphics
                            Mesh
                            Node
                            Camera
                            ShaderData
                            ParticleSystem
                            MeshedWeapon
                            MeshedProjectile
                            Target
                            HitBox
                            Sound
                            )

TARGET_LINK_LIBRARIES(main 
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

INSTALL (TARGETS main DESTINATION bin)
