/*
 * Init.cpp
 *
 *  Created on: Mar 11, 2012
 *      Author: per
 */
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
    #include <windows.h>
#endif

#include <iostream>
#include <cuda_gl_interop.h>

static bool initialized = false;

namespace CUDA {

void init()
{
    std::cerr << "INIT CUDA.cpp!!!!" << std::endl;
    if (initialized) return;
    cudaGLSetGLDevice(0);
    initialized = true;
}

}

