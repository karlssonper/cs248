/*
 * linux_helper.h
 *
 *  Created on: Mar 14, 2012
 *      Author: per
 */

#ifndef LINUX_HELPER_H_
#define LINUX_HELPER_H_
#ifdef __CDT_PARSER__
#define __global__
#define __device__
#define __host__
#define __shared__
#define CUDA_KERNEL_DIM(...)

#else
#define CUDA_KERNEL_DIM(...)  <<< __VA_ARGS__ >>>

#endif

#endif /* LINUX_HELPER_H_ */
