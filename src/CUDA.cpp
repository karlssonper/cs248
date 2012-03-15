/*
 * Init.cpp
 *
 *  Created on: Mar 11, 2012
 *      Author: per
 */
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <cuda_gl_interop.h>

static bool initialized = false;

namespace CUDA {

void init()
{
    if (initialized) return;
    cudaGLSetGLDevice(0);
    initialized = true;
}

}
