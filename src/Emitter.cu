#include "Emitter.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <iostream>

__global__ void initRand(curandState *_randstate) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // From CURAND library guide
    // Each thread gets same seed, a different sequence number
    // and no offset.

    curand_init(2345, tid, 0, &_randstate[tid]);
}

__global__ void init(Emitter::EmitterParams _p,
                     float *_time,
                     float *_pos,
                     float *_acc,
                     float *_vel,
                     float *_col) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < _p.numParticles_) {

        _time[tid] = 1.0f;

        _pos[3*tid+0] = _p.startPos_[0];
        _pos[3*tid+1] = _p.startPos_[1];
        _pos[3*tid+2] = _p.startPos_[2];

        _acc[3*tid+0] = _p.startAcc_[0];
        _acc[3*tid+1] = _p.startAcc_[1];
        _acc[3*tid+2] = _p.startAcc_[2];

        _vel[3*tid+0] = _p.startVel_[0];
        _vel[3*tid+1] = _p.startVel_[1];
        _vel[3*tid+2] = _p.startVel_[2];

        _col[3*tid+0] = _p.color_[0];
        _col[3*tid+1] = _p.color_[1];
        _col[3*tid+2] = _p.color_[2];

        tid += blockDim.x * gridDim.x;
    }
}
                             
__global__ void newParticle(Emitter::EmitterParams _p,
                            float *_time,
                            float *_pos,
                            float *_acc,
                            float *_vel,
                            float *_col,
                            unsigned int _index,
                            curandState *_randstate) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int index;
    int limit;
    switch (_p.emitterType_) {
    case Emitter::EMITTER_STREAM:
        index = _index; // only add one new particle
        limit = _p.numParticles_; 
        break;
    case Emitter::EMITTER_BURST:
        index = tid; // add several particles
        if (_p.burstSize_ <= _p.numParticles_) limit = _p.burstSize_;
        else limit = _p.numParticles_;
        break;
    }

    while (index < limit) {

        // get three random floats for start velocity, one for time
        float vx_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float vy_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float vz_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float px_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float py_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float pz_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float t_offset = curand_normal(&_randstate[tid]);

        _time[index] = 1.0f + t_offset*0.1;

        _pos[3*index+0] = _p.startPos_[0] + px_offset * _p.posRandWeight_;
        _pos[3*index+1] = _p.startPos_[1] + py_offset * _p.posRandWeight_;
        _pos[3*index+2] = _p.startPos_[2] + pz_offset * _p.posRandWeight_;

        _acc[3*index+0] = _p.startAcc_[0];
        _acc[3*index+1] = _p.startAcc_[1];
        _acc[3*index+2] = _p.startAcc_[2];

        _vel[3*index+0] = _p.startVel_[0] + vx_offset * _p.velRandWeight_;
        _vel[3*index+1] = _p.startVel_[1] + vy_offset * _p.velRandWeight_;
        _vel[3*index+2] = _p.startVel_[2] + vz_offset * _p.velRandWeight_;

        _col[3*index+0] = _p.color_[0];
        _col[3*index+1] = _p.color_[1];
        _col[3*index+2] = _p.color_[2];

        // only run once if stream (only add one at a time)
        if (_p.emitterType_ == Emitter::EMITTER_STREAM) break;

        index += blockDim.x * gridDim.x;

    }

}

__global__ void integrate(Emitter::EmitterParams _p,
                          float *_time,
                          float *_pos,
                          float *_acc,
                          float *_vel,
                          float *_col,
                          float _dt) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < _p.numParticles_) {

        if (_time[tid] > 0.0) {

            // subtract elapsed time
            _time[tid] -= (1.f/_p.lifeTime_)*_dt;

            _vel[3*tid+0] += _dt * _acc[3*tid+0];
            _vel[3*tid+1] += _dt * _acc[3*tid+1];
            _vel[3*tid+2] += _dt * _acc[3*tid+2];

            _pos[3*tid+0] += _dt * _vel[3*tid+0];
            _pos[3*tid+1] += _dt * _vel[3*tid+1];
            _pos[3*tid+2] += _dt * _vel[3*tid+2];

        }

        tid += blockDim.x * gridDim.x;
    }
}

Emitter::Emitter(unsigned int _numParticles) {

    params_.numParticles_ = _numParticles;

    blocks_ = threads_ = 128;
    
    // allocate device memory
    cudaMalloc((void**)&d_time_, sizeof(float)*_numParticles);
    cudaMalloc((void**)&d_pos_, sizeof(float)*3*_numParticles);
    cudaMalloc((void**)&d_acc_, sizeof(float)*3*_numParticles);
    cudaMalloc((void**)&d_vel_, sizeof(float)*3*_numParticles);
    cudaMalloc((void**)&d_col_, sizeof(float)*3*_numParticles);

    // for random states
    cudaMalloc((void**)&d_randstate_, sizeof(curandState)*blocks_*threads_);

    // init
    init<<<blocks_,threads_>>>(params_, d_time_, d_pos_, d_acc_, d_vel_, d_col_);
    initRand<<<blocks_, threads_>>>(d_randstate_);

    //copyPosToHostAndPrint();

    // first particle goes in the first slot
    nextSlot_ = 0;

    // reset time
    nextEmission_ = params_.rate_;

    // generate VBOs
    vboPos_ = new GLuint;
    glGenBuffers(1, vboPos_);
    glBindBuffer(GL_ARRAY_BUFFER, *vboPos_);
    glBufferData(GL_ARRAY_BUFFER, 
                 sizeof(float)*3*_numParticles,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(*vboPos_);

    vboCol_ = new GLuint;
    glGenBuffers(1, vboCol_);
    glBindBuffer(GL_ARRAY_BUFFER, *vboCol_);
    glBufferData(GL_ARRAY_BUFFER, 
                 sizeof(float)*3*_numParticles,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(*vboCol_);

    vboTime_ = new GLuint;
    glGenBuffers(1, vboTime_);
    glBindBuffer(GL_ARRAY_BUFFER, *vboTime_);
    glBufferData(GL_ARRAY_BUFFER, 
                 sizeof(float)*_numParticles,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(*vboTime_);


}

void Emitter::burst() {

    if (params_.emitterType_ != Emitter::EMITTER_BURST) return;

    cudaGLMapBufferObject((void**)&d_pos_, *vboPos_);
    cudaGLMapBufferObject((void**)&d_col_, *vboCol_);
    cudaGLMapBufferObject((void**)&d_time_, *vboTime_);

    newParticle<<<blocks_,threads_>>>(params_, 
                                     d_time_, 
                                     d_pos_, 
                                     d_acc_, 
                                     d_vel_, 
                                     d_col_,
                                     0,
                                     d_randstate_);

    cudaGLUnmapBufferObject(*vboPos_);
    cudaGLUnmapBufferObject(*vboCol_);
    cudaGLUnmapBufferObject(*vboTime_);
}

void Emitter::update(float _dt) {

    cudaGLMapBufferObject((void**)&d_pos_, *vboPos_);
    cudaGLMapBufferObject((void**)&d_col_, *vboCol_);
    cudaGLMapBufferObject((void**)&d_time_, *vboTime_);

    // only care about new emissions if it's a stream
    if (params_.emitterType_ == Emitter::EMITTER_STREAM) {

        // count off elapsed time
        nextEmission_ -= _dt;

        // std::cout << "Next emission: " << nextEmission_ << std::endl;
         //std::cout << "Nect slot: " << nextSlot_ << std::endl;

        if (nextEmission_ < 0.0) {

            // calculate how many particles we should emit
            int numNewParticles = (int)(-nextEmission_/params_.rate_);

            // reset time for next emission
            nextEmission_ += numNewParticles*params_.rate_;
            nextEmission_ += params_.rate_;

            // emit new particles to make up for any overlap in elapsed time
            do {
                // emit a particle
                newParticle<<<1,1>>>(params_, 
                                     d_time_, 
                                     d_pos_, 
                                     d_acc_, 
                                     d_vel_, 
                                     d_col_,
                                     nextSlot_,
                                     d_randstate_);

                // jump forward one slot
                nextSlot_++;
                if (nextSlot_ == params_.numParticles_) nextSlot_ = 0;

                numNewParticles--;

            } while (numNewParticles > 0);
        
        } // if nextemission

    } // if stream
        
    // update all the particles
    integrate<<<blocks_,threads_>>>(params_,
                                    d_time_, 
                                    d_pos_, 
                                    d_acc_, 
                                    d_vel_, 
                                    d_col_,
                                    _dt);

    cudaGLUnmapBufferObject(*vboPos_);
    cudaGLUnmapBufferObject(*vboCol_);
    cudaGLUnmapBufferObject(*vboTime_);

}

void Emitter::posIs(Vector3 _pos) {
    params_.startPos_[0] = _pos.x;
    params_.startPos_[1] = _pos.y;
    params_.startPos_[2] = _pos.z;
}

void Emitter::accIs(Vector3 _acc) {
    params_.startAcc_[0] = _acc.x;
    params_.startAcc_[1] = _acc.y;
    params_.startAcc_[2] = _acc.z;
}

void Emitter::velIs(Vector3 _vel) {
    params_.startVel_[0] = _vel.x;
    params_.startVel_[1] = _vel.y;
    params_.startVel_[2] = _vel.z;
}

void Emitter::rateIs(float _rate) {
    params_.rate_ = _rate;
}

void Emitter::massIs(float _mass) {
    params_.mass_ = _mass;
}

void Emitter::burstSizeIs(unsigned int _burstSize) {
    params_.burstSize_ = _burstSize;
}

void Emitter::lifeTimeIs(float _lifeTime) {
    params_.lifeTime_ = _lifeTime;
}

void Emitter::typeIs(Type _emitterType) {
    params_.emitterType_ = _emitterType;
}

void Emitter::colIs(Vector3 _col) {
    params_.color_[0] = _col.x;
    params_.color_[1] = _col.y;
    params_.color_[2] = _col.z;
}

void Emitter::velRandWeightIs(float _velRandWeight) {
    params_.velRandWeight_ = _velRandWeight;
}

void Emitter::posRandWeightIs(float _posRandWeight) {
    params_.posRandWeight_ = _posRandWeight;
}

void Emitter::copyPosToHostAndPrint() {

    float *h_pos;
    h_pos = new float[params_.numParticles_*3];
    cudaMemcpy(h_pos, d_pos_, sizeof(float)*3*params_.numParticles_, cudaMemcpyDeviceToHost);

    for (int i=0; i<params_.numParticles_; ++i) {
        std::cout << "(" << h_pos[3*i] << ", " << h_pos[3*i+1] << ", " << h_pos[3*i+2] << ")" << std::endl;
    }
    delete h_pos;

}






