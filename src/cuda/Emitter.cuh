#ifndef EMITTER_H
#define EMITTER_H


#include <vector>
#include "../MathEngine.h"
#include "../ShaderData.h"


#include <curand_kernel.h>

class Particle;

class Emitter {
public:

    enum Type {
        EMITTER_STREAM = 0,
        EMITTER_BURST
    };

    enum BlendMode {
        BLEND_FIRE = 0,
        BLEND_SMOKE
    };

    struct EmitterParams {
        unsigned int numParticles_;
        float rate_;
        float mass_;
        float startPos_[3];
        float posRandWeight_;
        float startVel_[3];
        float velRandWeight_;
        float startAcc_[3];
        float pointSize_;
        float growthFactor_;
        float lifeTime_;
        unsigned int burstSize_;
        Type emitterType_;
        BlendMode blendMode_;
    };

    Emitter(unsigned int _numParticles, ShaderData*_sd);
    void update(float _dt);
    void burst();
    unsigned int vboPos() const { return vboPos_; }
    unsigned int vboSize() const { return vboSize_; }
    unsigned int vboTime() const { return vboTime_; }
    EmitterParams params() const { return params_; }
    void display() const;
    void posIs(Vector3 _pos);
    void accIs(Vector3 _acc);
    void velIs(Vector3 _vel);
    void massIs(float _mass);
    void rateIs(float _rate);
    void lifeTimeIs(float _lifeTime);
    void burstSizeIs(unsigned int _burstSize);
    void typeIs(Type _emitterType);
    void velRandWeightIs(float _velRandWeight);
    void posRandWeightIs(float _posRandWeight);
    void pointSizeIs(float _pointSize);
    void growthFactorIs(float _growthFactor);
    void shaderDataIs(ShaderData * _shaderData);
    void blendModeIs(BlendMode _blendMode);
    
private:
    
    // parameters
    EmitterParams params_;

    ShaderData * shaderData_;

    // vbo's with the stuff the shaders needs to know about
    unsigned int VAO_;
    unsigned int vboPos_;
    unsigned int vboSize_;
    unsigned int vboTime_;

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
    float *d_size_;

    unsigned int blocks_;
    unsigned int threads_;
};

#endif
