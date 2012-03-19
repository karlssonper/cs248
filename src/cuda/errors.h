/*
 * errors.h
 *
 *  Created on: Mar 18, 2012
 *      Author: per
 */

#ifndef ERRORS_H_
#define ERRORS_H_

#include <string>
#include <iostream>
namespace CUDA {
static bool checkErrors()
{
    cudaError_t error = cudaGetLastError();
    std::string errorStr(cudaGetErrorString(error));
    if (errorStr == std::string("no error")) return true;
    std::cerr << errorStr << std::endl;
    return false;
}
}



#endif /* ERRORS_H_ */
