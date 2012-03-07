#ifndef EMITTER_H
#define EMITTER_H

#include <GL/glew.h>
#include <GL/glut.h>
#include <vector>

#include <curand_kernel.h>

class Particle;

class Emitter {
public:

    enum Type {
        EMITTER_STREAM = 0,
        EMITTER_BURST
    };

    struct EmitterParams {
        unsigned int numParticles_;
        float rate_;
        float mass_;
        float startPos_[3];
        float startVel_[3];
        float startAcc_[3];
        float color_[3];
        float lifeTime_;
        unsigned int burstSize_;
        Type emitterType_;
    };

    Emitter(EmitterParams _params);
    void update(float _dt);

    GLuint vboPos() const { return *vboPos_; }
    GLuint vboCol() const { return *vboCol_; }
    GLuint vboTime() const { return *vboTime_; }

    EmitterParams params() const { return params_; }

    

private:
    
    // parameters
    EmitterParams params_;

    // vbo's with the stuff the shaders needs to know about
    GLuint *vboPos_;
    GLuint *vboCol_;
    GLuint *vboTime_;

    // cudart states for random number generation
    curandState *d_randstate_;

    // debug purposes
    void copyPosToHostAndPrint();

    // keeps track of the next slot in the array to put
    // new particles in
    unsigned int nextSlot_;

    // time to next emission
    float nextEmission_;

    // pointers to particle data on device
    float *d_time_;
    float *d_pos_;
    float *d_acc_;
    float *d_vel_;
    float *d_col_;

    unsigned int blocks_;
    unsigned int threads_;
};

#endif