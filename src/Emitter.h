#ifndef EMITTER_H
#define EMITTER_H

#include <GL/glew.h>
#include <GL/glut.h>
#include "MathEngine.h"
#include <vector>

class Particle;

class Emitter {
public:

    enum Type {
        EMITTER_SMOKE = 0,
        EMITTER_FIRE
    };

    struct EmitterParams {
        unsigned int numParticles_;
        float rate_;
        float mass_;
        Vector3 startPos_;
        Vector3 startVel_;
        Vector3 startAcc_;
        Vector3 extForce_;
        Vector3 color_;
        float lifeTime_;
        Type type_;
    };

    Emitter(EmitterParams _params);
    void update(float _dt);

    EmitterParams params_;

    // pointers to particle data on device
    bool *d_act_;
    float *d_time_;
    float *d_pos_;
    float *d_acc_;
    float *d_vel_;
    float *d_col_;

    // vbo's with the stuff the shaders needs to know about
    GLuint *vboPos_;
    GLuint *vboCol_;

    // keeps track of the next slot in the array to put
    // new particles in
    unsigned int nextSlot_;

    // time to next emission
    float nextEmission_;
}

#endif