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

    if (_index < _p.numParticles_) {

        // get three random floats for start velocity, one for time
        float x_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float y_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float z_offset = 2.f * ( curand_normal(&_randstate[tid]) - 0.5f );
        float t_offset = curand_normal(&_randstate[tid]);

        _time[_index] = 1.0f + t_offset*0.1;

        _pos[3*_index+0] = _p.startPos_[0];
        _pos[3*_index+1] = _p.startPos_[1];
        _pos[3*_index+2] = _p.startPos_[2];

        _acc[3*_index+0] = _p.startAcc_[0];
        _acc[3*_index+1] = _p.startAcc_[1];
        _acc[3*_index+2] = _p.startAcc_[2];

        _vel[3*_index+0] = _p.startVel_[0] + x_offset*0.015;
        _vel[3*_index+1] = _p.startVel_[1] + y_offset*0.015;
        _vel[3*_index+2] = _p.startVel_[2] + z_offset*0.015;

        _col[3*_index+0] = _p.color_[0];
        _col[3*_index+1] = _p.color_[1];
        _col[3*_index+2] = _p.color_[2];

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

Emitter::Emitter(EmitterParams _params) : params_(_params) {

    blocks_ = threads_ = 128;
    
    // allocate device memory
    cudaMalloc((void**)&d_time_, sizeof(float)*params_.numParticles_);
    cudaMalloc((void**)&d_pos_, sizeof(float)*3*params_.numParticles_);
    cudaMalloc((void**)&d_acc_, sizeof(float)*3*params_.numParticles_);
    cudaMalloc((void**)&d_vel_, sizeof(float)*3*params_.numParticles_);
    cudaMalloc((void**)&d_col_, sizeof(float)*3*params_.numParticles_);

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
                 sizeof(float)*3*params_.numParticles_,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(*vboPos_);

    vboCol_ = new GLuint;
    glGenBuffers(1, vboCol_);
    glBindBuffer(GL_ARRAY_BUFFER, *vboCol_);
    glBufferData(GL_ARRAY_BUFFER, 
                 sizeof(float)*3*params_.numParticles_,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(*vboCol_);

    vboTime_ = new GLuint;
    glGenBuffers(1, vboTime_);
    glBindBuffer(GL_ARRAY_BUFFER, *vboTime_);
    glBufferData(GL_ARRAY_BUFFER, 
                 sizeof(float)*params_.numParticles_,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGLRegisterBufferObject(*vboTime_);


}

void Emitter::update(float _dt) {

    cudaGLMapBufferObject((void**)&d_pos_, *vboPos_);
    cudaGLMapBufferObject((void**)&d_col_, *vboCol_);
    cudaGLMapBufferObject((void**)&d_time_, *vboTime_);

    // count off lapsed time
    nextEmission_ -= _dt;

    // std::cout << "Next emission: " << nextEmission_ << std::endl;
     //std::cout << "Nect slot: " << nextSlot_ << std::endl;

    if (nextEmission_ < 0.0) {

        // reset time for next emission
        nextEmission_ += params_.rate_;

        // emit a particle
        newParticle<<<blocks_,threads_>>>(params_, 
                                 d_time_, 
                                 d_pos_, 
                                 d_acc_, 
                                 d_vel_, 
                                 d_col_,
                                 nextSlot_,
                                 d_randstate_);
        
         //copyPosToHostAndPrint();

        // jump forward one slot
        nextSlot_++;
        if (nextSlot_ == params_.numParticles_) nextSlot_ = 0;
    }
        
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

void Emitter::copyPosToHostAndPrint() {

    float *h_pos;
    h_pos = new float[params_.numParticles_*3];
    cudaMemcpy(h_pos, d_pos_, sizeof(float)*3*params_.numParticles_, cudaMemcpyDeviceToHost);

    for (int i=0; i<params_.numParticles_; ++i) {
        std::cout << "(" << h_pos[3*i] << ", " << h_pos[3*i+1] << ", " << h_pos[3*i+2] << ")" << std::endl;
    }
    delete h_pos;

}






