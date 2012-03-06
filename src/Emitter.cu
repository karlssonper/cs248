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

        _pos[3*tid+0] = d_params_.startPos_.x_;
        _pos[3*tid+1] = d_params_.startPos_.y_;
        _pos[3*tid+2] = d_params_.startPos_.z_;

        _acc[3*tid+0] = d_params_.startAcc_.x_;
        _acc[3*tid+1] = d_params_.startAcc_.y_;
        _acc[3*tid+2] = d_params_.startAcc_.z_;

        _vel[3*tid+0] = d_params_.startVel_.x_;
        _vel[3*tid+1] = d_params_.startVel_.y_;
        _vel[3*tid+2] = d_params_.startVel_.z_;

        _col[3*tid+0] = d_params_.color_.x_;
        _col[3*tid+1] = d_params_.color_.y_;
        _col[3*tid+2] = d_params_.color_.z_;

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

        _pos[3*_index+0] = d_params_.startPos_.x_;
        _pos[3*_index+1] = d_params_.startPos_.y_;
        _pos[3*_index+2] = d_params_.startPos_.z_;

        _acc[3*_index+0] = d_params_.startAcc_.x_;
        _acc[3*_index+1] = d_params_.startAcc_.y_;
        _acc[3*_index+2] = d_params_.startAcc_.z_;

        _vel[3*_index+0] = d_params_.startVel_.x_;
        _vel[3*_index+1] = d_params_.startVel_.y_;
        _vel[3*_index+2] = d_params_.startVel_.z_;

        _col[3*_index+0] = d_params_.color_.x_;
        _col[3*_index+1] = d_params_.color_.y_;
        _col[3*_index+2] = d_params_.color_.z_;

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
    init(d_act_, d_time_, d_pos_, d_acc_, d_vel_, d_col_);

    // first particle goes in the first slot
    nextSlot_ = 0;

    // reset time
    nextEmission_ = params_.rate_;


}

void Emitter::update(float _dt) {

    // count off lapsed time
    nextEmission_ -= _dt;

    if (nextEmission_ < 0.0) {

        // reset time for next emission
        nextEmission_ += params_.rate_;

        // emitt a particle
        newParticle(d_act_, 
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
        integrate(d_act_, 
                  d_time_, 
                  d_pos_, 
                  d_acc_, 
                  d_vel_, 
                  d_col_,
                  _dt);
    }
}






