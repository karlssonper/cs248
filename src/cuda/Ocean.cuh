/*
 * Ocean.cuh
 *
 *  Created on: Mar 9, 2012
 *      Author: per
 */

#ifndef OCEAN_CUH_
#define OCEAN_CUH_

class ShaderData;

namespace CUDA {
namespace Ocean {

void performIFFT(float time, bool disp);

void updateVBO(bool disp);

void display();

void init();

ShaderData* oceanShaderData();

} //end namespace Ocean
} //end namespace CUDA
#endif
