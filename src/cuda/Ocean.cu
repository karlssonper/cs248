/*
 * Ocean.cu
 *
 *  Created on: Mar 9, 2012
 *      Author: per
 */


#include "linux_helper.h"
#include "../Graphics.h"
#include "../Camera.h"
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <vector>

#define DIR_X 0
#define DIR_Y 1
#define DIR_Z 2
#define N 128 //REQUENCY RESOLUTION
#define WORLD_SIZE 100.0f // WORLD SIZE
#define OCEAN_DEPTH 200.0f
#define WIND_X 1.0f
#define WIND_Z 0.0f
#define WIND_SPEED 30.0f
#define L WIND_SPEED*WIND_SPEED/9.82f
#define TWO_PI 6.2831853071f
#define DAMP_WAVES 0.5f
#define SHORTEST_WAVELENGTH 0.02f
#define WAVE_HEIGHT 2.0f
#define WIND_ALIGN 2
#define CHOPPYNESS 0.5f

struct OceanVertex
{
    float pos[3];
    float partialU[3];
    float partialV[3];
    float fold;
};

unsigned int VBO_GL;
struct cudaGraphicsResource* VBO_CUDA;
unsigned int VBO_IDX;
unsigned int VAO;
ShaderData * shaderData;
unsigned int idxSize;

//Arrays on the device
cudaArray * d_a;

//FFT data
cufftComplex * idata;
cufftComplex * Y;
cufftReal * odata[3];

//FFT plans
cufftHandle plans[3];

//Read Only Texture
texture<float4, 2, cudaReadModeElementType> t_a;

//Blocks and threads
dim3 bC, tC;//complex part ( DIM X * DIMZ /2)
dim3 bB, tB;//bottom line (DIMX * 1)
dim3 b2D, t2D;//Regular 2D grid (DIMX * DIMZ)

int bLine;
int padding;
float verticalScale;

__device__
float _kx(const int i, const int dimX, const float wx)
{
    if (i <= dimX/2)
        return TWO_PI * i / wx;
    else
        return TWO_PI * (i - dimX + 1) / wx;
}

__device__
float _kz(const int j, const float wz)
{
    return TWO_PI * j / wz;
}

__device__
int idxFFT(const int i, const int j)
{
    return j + i * blockDim.y * gridDim.y;
}

__device__
int idxPos(const int i, const int j)
{
    return i + j * blockDim.x * gridDim.x;
}


template <int dir, bool bottomLine>
__global__
void build(  cufftComplex * idata,
             const float time,
             const int padding,
             const cufftComplex * Y,
             const int bLine = 0 )
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j;
    if (!bottomLine){
        j = threadIdx.y + blockIdx.y * blockDim.y;
    }
    else
        j = bLine;

    //Row major
    const int idx = j + i * padding;

    float kx = _kx(i,N,WORLD_SIZE);
    float kz = _kz(j,WORLD_SIZE);
    const float k = sqrt(kx * kx + kz * kz);

    if (dir == DIR_Y){
        const float4 a = tex2D(t_a, i, j);
        const float omega = sqrt(9.82f * k);
        const float cosOmega = __cosf(omega * time);
        const float sinOmega = __sinf(omega * time);

        const cuComplex expp = make_cuComplex(cosOmega, sinOmega);
        const cuComplex expm = make_cuComplex(cosOmega, -sinOmega);
        const cuComplex ap = make_cuComplex(a.x, a.y);
        const cuComplex am = make_cuComplex(a.z, a.w);

        idata[idx] = cuCaddf(cuCmulf(ap, expp), cuCmulf(am, expm) );
    }else if (dir == DIR_X) {
        if(k != 0)
            idata[idx] = cuCmulf(make_cuComplex(0,kx/k), Y[idx]);
        else
            idata[idx] = make_cuComplex(0,0);
    }else if (dir == DIR_Z) {
        if(k != 0)
            idata[idx] = cuCmulf(make_cuComplex(0,kz/k), Y[idx]);
        else
            idata[idx] = make_cuComplex(0,0);
    }
}

template<bool disp>
__global__
void updatePositions(const float scale,
                     const float scaleY,
                     const float scaleXZ,
                     float * out,
                     const float * yIn,
                     const float * xIn = NULL,
                     const float * zIn = NULL)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;


    const int idx = idxFFT(i,j);

    const float x = i * scale;
    const float z = j * scale;

    const int idxLeft = (i != 0) ? (idxFFT(i-1,j)) : (idxFFT(N-1,j));
    const int idxRight = (i != N-1) ? (idxFFT(i+1,j)) : (idxFFT(0,j));
    const int idxTop = (j != N-1) ? (idxFFT(i,j+1)) : (idxFFT(i,0));
    const int idxBot = (j != 0) ? (idxFFT(i,j-1)) : (idxFFT(i,N-1));


    float dxdu;
    float dzdu;
    float dxdv;
    float dzdv;
    float delta = scaleY /(scale*2.0f);

    if (disp) {
        dxdu = (xIn[idxRight] - xIn[idxLeft]) * delta;
        dxdv = (xIn[idxTop] - xIn[idxBot]) * delta;
        dzdu = (zIn[idxRight] - zIn[idxLeft]) * delta;
        dzdv = (zIn[idxTop] - zIn[idxBot]) * delta;
    } else {
        dxdu = scaleY*1.0f;
        dxdv = 0.0f;
        dzdu = 0.0f;
        dzdv = scaleY*1.0f;
    }

    float dydu = (yIn[idxRight] - yIn[idxLeft])*delta;
    float dydv = (yIn[idxTop] - yIn[idxBot])*delta;

    if (disp){
        float2 dx = make_float2((xIn[idxRight] - xIn[idxLeft])* CHOPPYNESS * N/WORLD_SIZE,
                (zIn[idxRight] - zIn[idxLeft])* CHOPPYNESS * N/WORLD_SIZE) ;
        float2 dy = make_float2((xIn[idxTop] - xIn[idxBot])* CHOPPYNESS * N/WORLD_SIZE,
                (zIn[idxTop] - zIn[idxBot]) * CHOPPYNESS * N/WORLD_SIZE);

        float J = (1.0f + dx.x) * (1.0f + dy.y) - dx.y * dy.x;

        out[idxPos(i,j)*10] = x + scaleXZ * xIn[idx];
        out[idxPos(i,j)*10+1] = scaleY * yIn[idx];
        out[idxPos(i,j)*10+2] = z + scaleXZ * zIn[idx];

        out[idxPos(i,j)*10+3] = dxdu;
        out[idxPos(i,j)*10+4] = dydu;
        out[idxPos(i,j)*10+5] = dzdu;

        out[idxPos(i,j)*10+6] = dxdv;
        out[idxPos(i,j)*10+7] = dydv;
        out[idxPos(i,j)*10+8] = dzdv;

        out[idxPos(i,j)*10+9] = max(1.0f - J, 0.f);
    }
    else {
        out[idxPos(i,j)*10] = x;
        out[idxPos(i,j)*10+1] = scaleY * yIn[idx];
        out[idxPos(i,j)*10+2] = z;

        out[idxPos(i,j)*10+3] = dxdu;
        out[idxPos(i,j)*10+4] = dydu;
        out[idxPos(i,j)*10+5] = dzdu;

        out[idxPos(i,j)*10+6] = dxdv;
        out[idxPos(i,j)*10+7] = dydv;
        out[idxPos(i,j)*10+8] = dzdv;

        out[idxPos(i,j)*10+9] = 1.0f;

    }
};


template<int tpb>
__global__
void getMax( float * in, float * out, const int n)
{
    __shared__ float max[tpb];
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int idx = j + i * blockDim.y * gridDim.y;
    max[tid] = in[idx];

    __syncthreads();

    //Assuming blockDim,x == blockDim.y
    int it = tpb/2;
    while (it != 0){
        if (tid < it && max[tid] < abs(max[tid + it]))
            max[tid] = abs(max[tid + it]);
        __syncthreads();
        it /= 2;
    }
    out[blockIdx.x + blockIdx.y * gridDim.x] = max[0];
};

namespace CUDA {

namespace Ocean {

void performIFFT(float time, bool disp)
{
    build<DIR_Y,false> CUDA_KERNEL_DIM(bC,tC)(Y, time, padding, NULL);
    build<DIR_Y,true> CUDA_KERNEL_DIM(bB,tB)(Y,time,padding,NULL,bLine);

    //std::cerr << "SCALEXZ: " <<scaleXZ << "\n";
    if (disp) {
        build<DIR_X, false> CUDA_KERNEL_DIM(bC,tC)(idata, time, padding, Y);
        build<DIR_X, true> CUDA_KERNEL_DIM(bB,tB)(idata, time, padding,Y,bLine);
        cufftExecC2R(plans[DIR_X], idata, odata[DIR_X]);

        build<DIR_Z, false> CUDA_KERNEL_DIM(bC,tC)(idata, time, padding, Y);
        build<DIR_Z, true> CUDA_KERNEL_DIM(bB,tB)(idata, time, padding,Y,bLine);
        cufftExecC2R(plans[DIR_Z], idata, odata[DIR_Z]);
    }
    cufftExecC2R(plans[DIR_Y], Y, odata[DIR_Y]);
}

void updateVBO(bool disp)
{
    float* positions;
    cudaGraphicsMapResources(1, &VBO_CUDA, 0);
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                          &num_bytes,
                                          VBO_CUDA);
    float scale = WORLD_SIZE / N;
    if (disp){
        updatePositions<true> CUDA_KERNEL_DIM(b2D,t2D)(scale,
                                                       verticalScale,
                                                       CHOPPYNESS,
                                                       positions,
                                                       odata[DIR_Y],
                                                       odata[DIR_X],
                                                       odata[DIR_Z] );
    } else {
        updatePositions<false> CUDA_KERNEL_DIM(b2D,t2D)(scale,
                                                        verticalScale,
                                                        0,
                                                        positions,
                                                        odata[DIR_Y]);
    }

    float * lol = (float*)malloc(N*N*sizeof(OceanVertex));
    cudaMemcpy(lol, positions, N*N*sizeof(OceanVertex),
            cudaMemcpyDeviceToHost);
    for (int i = 0; i < N*N; ++i){
        std::cerr << lol[i*10 + 3] << std::endl;
        std::cerr << lol[i*10 + 4] << std::endl;
        std::cerr << lol[i*10 + 5] << std::endl;

        std::cerr << lol[i*10 + 6] << std::endl;
        std::cerr << lol[i*10 + 7] << std::endl;
        std::cerr << lol[i*10 + 8] << std::endl;
    }
    cudaGraphicsUnmapResources(1, &VBO_CUDA, 0);
}

// Function from NVIDIA.
// Generates Gaussian random number with mean 0 and standard deviation 1.
float gauss()
{
    float u1 = rand() / (float)RAND_MAX;
    float u2 = rand() / (float)RAND_MAX;
    if (u1 < 1e-6f)
        u1 = 1e-6f;
    return sqrtf(-2 * logf(u1)) * cosf(2*M_PI * u2);
}

float philipsSpectrum(float kx, float kz)
{
    float k2 = kx * kx + kz * kz;

    //Prevent 0 division
    if (k2 == 0.f)
        return 0.f;

    float wDmpd = (kx * WIND_X + kz * WIND_Z) / sqrt(k2);

    //Damp the waves in the opposite direction of the wind
    //Often called reflection dampening
    if (wDmpd < 0)
        wDmpd *= (1 - DAMP_WAVES);

    //See Tessendorf paper for final equation
    return exp( -1.f / (k2 * L * L))*
            exp(-k2 * SHORTEST_WAVELENGTH * SHORTEST_WAVELENGTH) *
            pow(fabs(wDmpd), WIND_ALIGN)/ (k2 * k2);
}

void execGetMax(float * in, float * out, const int tpb, const int n)
{
    switch (tpb){
        case 16:
            getMax<16*16> CUDA_KERNEL_DIM(b2D, t2D)(in, out, n);
            break;
        case 8:
            getMax<8*8> CUDA_KERNEL_DIM(b2D, t2D)(in, out, n);
            break;
        case 4:
            getMax<4*4> CUDA_KERNEL_DIM(b2D, t2D)(in, out, n);
            break;
    }
};

float maxHeight()
{
    build<DIR_Y, false> CUDA_KERNEL_DIM(bB,tC) (Y, 0, padding,NULL);
    build<DIR_Y, true> CUDA_KERNEL_DIM(bB,tB) (Y, 0, padding, NULL, bLine);
    cufftExecC2R(plans[DIR_Y], Y, odata[DIR_Y]);

    const int nBlocks = b2D.x * b2D.y;
    const int nThreads = t2D.x;
    const int nElements = nBlocks * nThreads* nThreads;

    float * maxArr = (float * ) malloc(nBlocks * sizeof(float));
    float * d_maxArr;
    cudaMalloc((void**)&d_maxArr, nBlocks * sizeof(float));
    execGetMax(odata[DIR_Y], d_maxArr, nThreads, nElements);

    cudaMemcpy(maxArr, d_maxArr, nBlocks * sizeof(float),
            cudaMemcpyDeviceToHost);

    float max = -10000000;
    for (int i = 0; i < nBlocks; ++i){
        if (maxArr[i] > max)
            max = maxArr[i];
    }
    cudaFree(d_maxArr);
    free(maxArr);
    return max;
};

void display()
{
    const Matrix4 & view = Camera::instance().viewMtx();
    Matrix4 * modelView = shaderData->stdMatrix4Data(MODELVIEW);
    Matrix4 * projection = shaderData->stdMatrix4Data(PROJECTION);
    Matrix3 * normal = shaderData->stdMatrix3Data(NORMAL);

    *modelView = Camera::instance().viewMtx();
    *projection = Camera::instance().projectionMtx();
    *normal = Matrix3(*modelView).inverse().transpose();
    Graphics::instance().drawIndices(VAO, VBO_IDX, idxSize, shaderData);
}

void init()
{
    static std::string name("OceanCUDA");
    Graphics::instance().buffersNew(name, VAO, VBO_GL, VBO_IDX);

    std::vector<float> kx(N);
    std::vector<float> kz(N/2+1);
    for (int i = 0 ; i <= N/2 ; ++i) {
        kx[i] = TWO_PI * i / WORLD_SIZE;
    }
    for (int i = N/2 + 1; i < N; ++i) {
        kx[i] = (i - N +1) * TWO_PI / WORLD_SIZE;
    }
    for (int i = 0 ; i < N/2+1 ; ++i) {
        kz[i] = TWO_PI * i / WORLD_SIZE;
    }

    const int n = kx.size() * kz.size();
    cudaMallocArray( &d_a, &t_a.channelDesc, kx.size(), kz.size() );
    float4 * h_a = (float4 *) malloc(n * sizeof(float4));
    for (int i = 0; i < kx.size(); ++i) {
        for (int j = 0; j < kz.size(); ++j) {
            int idx = i + kx.size()*j;
            float specPositive = sqrt(philipsSpectrum(kx[i],kz[j])/2.f);
            float specNegative = sqrt(philipsSpectrum(-kx[i],-kz[j])/2.f);
            float gaussReal = gauss();
            float gaussImg = gauss();
            //We store the conjugate, i.e minus in 4 slot
            h_a[idx] = make_float4(
                     specPositive * gaussReal,
                     specPositive * gaussImg,
                     specNegative * gaussReal,
                    -specNegative * gaussImg);

        }
    }

    cudaMemcpyToArray( d_a, 0, 0, h_a
            , n * sizeof(float4), cudaMemcpyHostToDevice);

    //No longer needed
    free(h_a);
    cudaBindTextureToArray( t_a, d_a, t_a.channelDesc);

    //Allocate in array (symmetry as in FFTW)
    cudaMalloc((void**)&idata, sizeof(float2) * N * (N/2 + 1));
    cudaMalloc((void**)&Y, sizeof(float2) * N * (N/2 + 1));

    //Loop over each dimension
    for (int i = 0; i < 3; ++i){
        //Output array
        cudaMalloc((void**)&odata[i], sizeof(cufftReal) * N * N);

        //FFTW plans
        cufftPlan2d(&plans[i], N, N, CUFFT_C2R);

        cufftSetCompatibilityMode(plans[i],CUFFT_COMPATIBILITY_FFTW_ASYMMETRIC);
    }

    //For Threads Per Block
    int tpbx, tpbz;
    if (N < 32){
        tpbx = N;
        tpbz = N/2;
    }
    else{
        tpbz = tpbx= 16;
    }
    //For kernel execution
    bC = make_uint3(  N/tpbx, N/(2 *tpbz), 1);
    tC = make_uint3(tpbx, tpbz, 1);

    bB = make_uint3(N/(tpbx), 1, 1);
    tB = make_uint3(tpbx, 1, 1);

    b2D = make_uint3(N/tpbx, N/tpbx, 1);
    t2D = make_uint3(tpbx, tpbx, 1);

    bLine = N/2;
    padding = N/2 + 1;

    verticalScale = WAVE_HEIGHT / maxHeight();

    std::vector<float> vertexData(sizeof(OceanVertex)*N*N);
    std::vector<unsigned int> indices(6*(N-1)*(N-1));
    unsigned int triIdx = 0;
    for (int i = 0; i < N-1; ++i) {
        for (int j = 0; j < N-1; ++j) {
            //Triangle 1
            indices[triIdx++] = i   + N* j;
            indices[triIdx++] = i+1 + N* j;
            indices[triIdx++] = i+1 + N*(j+1);

            //Triangle 2
            indices[triIdx++] = i   + N* j;
            indices[triIdx++] = i+1 + N*(j+1);
            indices[triIdx++] = i   + N*(j+1);
        }
    }
    if (triIdx != indices.size()) {
        std::cerr << "Wrong in Ocean idx array!";
        exit(1);
    }
    idxSize = indices.size();
    shaderData = new ShaderData("../shaders/ocean");
    shaderData->enableMatrix(MODELVIEW);
    shaderData->enableMatrix(PROJECTION);
    shaderData->enableMatrix(NORMAL);

    Graphics::instance().geometryIs(VBO_GL, VBO_IDX, vertexData, indices, VBO_DYNAMIC);
    int id = shaderData->shaderID();
    int locPos = Graphics::instance().shaderAttribLoc(id,"positionIn");
    int locPU = Graphics::instance().shaderAttribLoc(id,"partialUIn");
    int locPV = Graphics::instance().shaderAttribLoc(id,"partialVIn");
    int locFold = Graphics::instance().shaderAttribLoc(id,"foldIn");

    Graphics::instance().bindGeometry(id, VAO, VBO_GL,3,sizeof(OceanVertex),locPos,0);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL,2,sizeof(OceanVertex),locPU,12);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL,2,sizeof(OceanVertex),locPV,24);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL,1,sizeof(OceanVertex),locFold,36);

    cudaGraphicsGLRegisterBuffer(&VBO_CUDA, VBO_GL,
            cudaGraphicsMapFlagsWriteDiscard);
}

} //end namespace

}//end namespace
