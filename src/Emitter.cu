#include "Emitter.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

__constant__ Emitter::EmitterParams d_params_;

__global__ void init(bool *_act,
                     float *_time,
                     float *_pos,
                     float *_acc,
                     float *_vel,
                     float *_col) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < d_params_.numParticles_) {

        _act[tid] = false;

        _time[tid] = d_params_.lifeTime_;

        _pos[3*tid+0] = d_params_.startPos_[0];
        _pos[3*tid+1] = d_params_.startPos_[1];
        _pos[3*tid+2] = d_params_.startPos_[2];

        _acc[3*tid+0] = d_params_.startAcc_[0];
        _acc[3*tid+1] = d_params_.startAcc_[1];
        _acc[3*tid+2] = d_params_.startAcc_[2];

        _vel[3*tid+0] = d_params_.startVel_[0];
        _vel[3*tid+1] = d_params_.startVel_[1];
        _vel[3*tid+2] = d_params_.startVel_[2];

        _col[3*tid+0] = d_params_.color_[0];
        _col[3*tid+1] = d_params_.color_[1];
        _col[3*tid+2] = d_params_.color_[2];

        tid += blockDim.x * gridDim.x;
    }
}

__global__ void newParticle(bool *_act,
                            float *_time,
                            float *_pos,
                            float *_acc,
                            float *_vel,
                            float *_col,
                            unsigned int _index) {

    if (_index < d_params_.numParticles_) {

        _act[_index] = true;

        _time[_index] = d_params_.lifeTime_;

        _pos[3*_index+0] = d_params_.startPos_[0];
        _pos[3*_index+1] = d_params_.startPos_[1];
        _pos[3*_index+2] = d_params_.startPos_[2];

        _acc[3*_index+0] = d_params_.startAcc_[0];
        _acc[3*_index+1] = d_params_.startAcc_[1];
        _acc[3*_index+2] = d_params_.startAcc_[2];

        _vel[3*_index+0] = d_params_.startVel_[0];
        _vel[3*_index+1] = d_params_.startVel_[1];
        _vel[3*_index+2] = d_params_.startVel_[2];

        _col[3*_index+0] = d_params_.color_[0];
        _col[3*_index+1] = d_params_.color_[1];
        _col[3*_index+2] = d_params_.color_[2];

    }
}

__global__ void integrate(bool *_act,
                            float *_time,
                            float *_pos,
                            float *_acc,
                            float *_vel,
                            float *_col,
                            float _dt) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < d_params_.numParticles_) {

        if (_act[tid] == true) {

            // subtract elapsed time
             _time[tid] -= _dt;

            _vel[3*tid+0] += _dt * _acc[3*tid+0];
            _vel[3*tid+1] += _dt * _acc[3*tid+1];
            _vel[3*tid+2] += _dt * _acc[3*tid+2];

            _pos[3*tid+0] += _dt * _vel[3*tid+0];
            _pos[3*tid+1] += _dt * _vel[3*tid+1];
            _pos[3*tid+2] += _dt * _vel[3*tid+2];

            // set to inactive if the particle is dead
            if (_time[tid] < 0.0) {
                _act[tid] = false;
            }

        }

        tid += blockDim.x * gridDim.x;
    }
}

Emitter::Emitter(EmitterParams _params) : params_(_params) {

    // allocate device memory
    cudaMalloc((void**)&d_act_, sizeof(bool)*params_.numParticles_);
    cudaMalloc((void**)&d_time_, sizeof(float)*params_.numParticles_);
    cudaMalloc((void**)&d_pos_, sizeof(float)*3*params_.numParticles_);
    cudaMalloc((void**)&d_acc_, sizeof(float)*3*params_.numParticles_);
    cudaMalloc((void**)&d_vel_, sizeof(float)*3*params_.numParticles_);
    cudaMalloc((void**)&d_col_, sizeof(float)*3*params_.numParticles_);

    // init
    init<<<128,128>>>(d_act_, d_time_, d_pos_, d_acc_, d_vel_, d_col_);

    // first particle goes in the first slot
    nextSlot_ = 0;

    // reset time
    nextEmission_ = params_.rate_;

    // generate VBO
    vboPos_ = new GLuint;
    glGenBuffers(1, vboPos_);
    glBindBuffer(GL_ARRAY_BUFFER, *vboPos_);
    glBufferData(GL_ARRAY_BUFFER, 
                 sizeof(float)*3*params_.numParticles_,
                 0,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

}

void Emitter::update(float _dt) {

    cudaGLMapBufferObject((void**)&d_pos_, *vboPos_);

    // count off lapsed time
    nextEmission_ -= _dt;

    if (nextEmission_ < 0.0) {

        // reset time for next emission
        nextEmission_ += params_.rate_;

        // emitt a particle
        newParticle<<<128,128>>>(d_act_, 
                                 d_time_, 
                                 d_pos_, 
                                 d_acc_, 
                                 d_vel_, 
                                 d_col_,
                                 nextSlot_);

        // jump forward one slot
        nextSlot_++;
        if (nextSlot_ == params_.numParticles_) nextSlot_ = 0;

        // update all the particles
        integrate<<<128,128>>>(d_act_, 
                               d_time_, 
                               d_pos_, 
                               d_acc_, 
                               d_vel_, 
                               d_col_,
                              _dt);

        cudaGLUnmapBufferObject(*vboPos_);
    }
}






