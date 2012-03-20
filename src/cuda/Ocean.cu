/*
 * Ocean.cu
 *
 *  Created on: Mar 9, 2012
 *      Author: per
 */

#include "linux_helper.h"
#include "../Graphics.h"
#include "../Engine.h"
#include "../Camera.h"
#include <cuda_gl_interop.h>
#include <cufft.h>
#include <vector>
#include "errors.h"


#define DIR_X 0
#define DIR_Y 1
#define DIR_Z 2
#define N 128 //REQUENCY RESOLUTION
#define WORLD_SIZE 150.0f // WORLD SIZE
#define OCEAN_DEPTH 200.0f
#define WIND_X 1.0f
#define WIND_Z 0.0f
#define WIND_SPEED 30.0f
#define L WIND_SPEED*WIND_SPEED/9.82f
#define TWO_PI 6.2831853071f
#define DAMP_WAVES 0.5f
#define SHORTEST_WAVELENGTH 0.02f
#define WAVE_HEIGHT 2.5f
#define WIND_ALIGN 2
#define CHOPPYNESS 0.5f
#define BOAT_SIZE 2.7f
#define FOAM_TIME 3.5f
#define FOAM_ACTIVATED 5.0f
#define HORIZON_FACTOR 10

struct OceanVertex
{
    float pos[3];
    float partialU[3];
    float partialV[3];
    //float fold;
    float foamTime;
    float foamAlpha;
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
float curTime = 0.f;
float prevTime = 0.f;
float2 * d_boatsXZ;

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
int idxPosPadding(const int i, const int j, int padding)
{
    return + i + (j+padding) * (blockDim.x * gridDim.x);
}

__device__
int idxPos(const int i, const int j)
{
    return i + j * (blockDim.x * gridDim.x);
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

template <int padding>
__global__
void updateFoam(int numBoats, float2 * xz, float scale,float * out)
{

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;


    const int idx = idxPos(i,j);

    const float x = i * scale;
    const float z = (j + padding) * scale;

    bool hit = false;
    int idxP;
    if (padding > 0) {
        idxP= idxPosPadding(i,j,padding);
    } else {
        idxP= idxPos(i,j);
    }
    float size = BOAT_SIZE ;
    for (int k = 0; k < numBoats; ++k) {
        if ((x > xz[k].x-size) &&
            (x < xz[k].x+size) &&
            (z > xz[k].y-size) &&
            (z < xz[k].y+size)) {
            float dx = x-xz[k].x;
            //float dz = z-xz[k].y;
            //float len = sqrt(dx*dx + dz*dz);
            //if(z<xz[k].y && len < size) break;

            hit = true;
            out[idxP*11+9] = FOAM_ACTIVATED;

            out[idxP*11+10] = -abs(dx)/BOAT_SIZE;
            break;
        }
    }
    if (!hit && out[idxP*11+9] > FOAM_TIME) {
        out[idxP*11+9] = FOAM_TIME;
        float lol = out[idxP*11+10];
        out[idxP*11+10] = 1+lol;
    }
}

template<bool disp, int padding>
__global__
void updatePositions(const float scale,
                     const float scaleY,
                     const float scaleXZ,
                     float * out,
                     float dt,
                     const float * yIn,
                     const float * xIn = NULL,
                     const float * zIn = NULL)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;


    const int idx = idxFFT(i,j);

    const float x = i * scale;
    const float z = (j+padding) * scale;

    const int idxLeft = (i != 0) ? (idxFFT(i-1,j)) : (idxFFT(N-1,j));
    const int idxRight = (i != N-1) ? (idxFFT(i+1,j)) : (idxFFT(0,j));
    const int idxTop = (j != N-1) ? (idxFFT(i,j+1)) : (idxFFT(i,0));
    const int idxBot = (j != 0) ? (idxFFT(i,j-1)) : (idxFFT(i,N-1));


    float dxdu;
    float dzdu;
    float dxdv;
    float dzdv;
    float delta = scaleY;//(scale*2.0f);

    if (disp) {
        dxdu = (xIn[idxRight] - xIn[idxLeft]) * delta;
        dxdv = (xIn[idxTop] - xIn[idxBot]) * delta;
        dzdu = (zIn[idxRight] - zIn[idxLeft]) * delta;
        dzdv = (zIn[idxTop] - zIn[idxBot]) * delta;
    } else {
        dxdu = scale;
        dxdv = 0.0f;
        dzdu = 0.0f;
        dzdv = scale;
    }
    float factor = 1.0f;
    if (i >= N - HORIZON_FACTOR){
        factor= float(N-1-i)/HORIZON_FACTOR;
    }

    float dydu = (yIn[idxRight] - yIn[idxLeft]) * scaleY*factor;
    float dydv = (yIn[idxTop] - yIn[idxBot]) * scaleY*factor;

    int idxP;
    if (padding > 0) {
        idxP= idxPosPadding(i,j,padding);
    } else {
        idxP= idxPos(i,j);
    }
    if (disp){
        float2 dx = make_float2((xIn[idxRight] - xIn[idxLeft])* CHOPPYNESS * N/WORLD_SIZE,
                (zIn[idxRight] - zIn[idxLeft])* CHOPPYNESS * N/WORLD_SIZE) ;
        float2 dy = make_float2((xIn[idxTop] - xIn[idxBot])* CHOPPYNESS * N/WORLD_SIZE,
                (zIn[idxTop] - zIn[idxBot]) * CHOPPYNESS * N/WORLD_SIZE);

        //float J = (1.0f + dx.x) * (1.0f + dy.y) - dx.y * dy.x;

        out[idxP*11] = x + scaleXZ * xIn[idx];
        out[idxP*11+1] = scaleY * yIn[idx]*factor;
        out[idxP*11+2] = z + scaleXZ * zIn[idx];

        out[idxP*11+3] = dxdu;
        out[idxP*11+4] = dydu;
        out[idxP*11+5] = dzdu;

        out[idxP*11+6] = dxdv;
        out[idxP*11+7] = dydv;
        out[idxP*11+8] = dzdv;

        //out[idxPos(i,j)*11+9] = max(1.0f - J, 0.f);

        out[idxP*11+9] -= dt;
    }
    else {
        out[idxP*11] = x;
        out[idxP*11+1] = scaleY * yIn[idx]*factor;
        out[idxP*11+2] = z;

        out[idxP*11+3] = dxdu;
        out[idxP*11+4] = dydu;
        out[idxP*11+5] = dzdu;

        out[idxP*11+6] = dxdv;
        out[idxP*11+7] = dydv;
        out[idxP*11+8] = dzdv;

       // out[idxPos(i,j)*11+9] = 1.0f;
        out[idxP*11+9] -= dt;

    }
};

template <int size>
__global__
void smooth(float * pos)
{
    __shared__ float alpha[256+size*32];
    __shared__ float time[256+size*32];
    const int tid = threadIdx.x + blockDim.x * threadIdx.y;
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    const int idx = idxPos(i,j);
    alpha[tid] = pos[idx*11+10];
    time[tid] = pos[idx*11+11];

    if (threadIdx.x < size) {

    }

    __syncthreads();



}

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
    prevTime = curTime;
    curTime = time;

    checkErrors(); //todo remove
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
        updatePositions<true,0> CUDA_KERNEL_DIM(b2D,t2D)(scale,
                                                       verticalScale,
                                                       CHOPPYNESS,
                                                       positions,
                                                       curTime - prevTime,
                                                       odata[DIR_Y],
                                                       odata[DIR_X],
                                                       odata[DIR_Z] );
    } else {
        updatePositions<false,0> CUDA_KERNEL_DIM(b2D,t2D)(scale,
                                                        verticalScale,
                                                        0,
                                                        positions,
                                                        curTime - prevTime,
                                                        odata[DIR_Y]);
        updatePositions<false,N> CUDA_KERNEL_DIM(b2D,t2D)(scale,
                                                        verticalScale,
                                                        0,
                                                        positions,
                                                        curTime - prevTime,
                                                        odata[DIR_Y]);
    }

    cudaGraphicsUnmapResources(1, &VBO_CUDA, 0);
    checkErrors();//todo remove
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

    checkErrors();
    return max;
};

void display()
{
    Matrix4 * modelView = shaderData->stdMatrix4Data(MODELVIEW);
    Matrix4 * invView = shaderData->stdMatrix4Data(INVERSEVIEW);
    Matrix3 * normal = shaderData->stdMatrix3Data(NORMAL);

    *modelView = Engine::instance().camera()->viewMtx();
    *normal = Matrix3(*modelView).inverse().transpose();
    *invView = Engine::instance().camera()->viewMtx().inverse();
    Graphics::instance().drawIndices(VAO, VBO_IDX, idxSize, shaderData);

    checkErrors();
}

void init()
{
    std::cerr << "INIT OCEAN" << std::endl;
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
            //std::cerr << h_a[idx].x << " " << h_a[idx].y << " " << h_a[idx].z << " " << h_a[idx].w << std::endl;

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

    cudaMalloc((void**)&d_boatsXZ, sizeof(float2) * 5);
    std::cerr << "Max height: " << maxHeight() << std::endl;
    verticalScale = WAVE_HEIGHT / maxHeight();

    std::vector<float> vertexData(sizeof(OceanVertex)*N*N*2);
    std::vector<unsigned int> indices(6*(2*N-1)*(N-1));
    unsigned int triIdx = 0;
    for (int i = 0; i < N-1; ++i) {
        for (int j = 0; j < N-1; ++j) {
            //Triangle 1
            indices[triIdx++] = i   + N* (j);
            indices[triIdx++] = i+1 + N* (j);
            indices[triIdx++] = i+1 + N* (j+1);

            //Triangle 2
            indices[triIdx++] = i   + N* (j);
            indices[triIdx++] = i+1 + N* (j+1);
            indices[triIdx++] = i   + N* (j+1);
        }
    }
    for (int i = 0; i < N-1; ++i) {
        for (int j = N-1; j < 2*N-1; ++j) {
            //Triangle 1
            indices[triIdx++] = i   + N* (j);
            indices[triIdx++] = i+1 + N* (j);
            indices[triIdx++] = i+1 + N* (j+1);

            //Triangle 2
            indices[triIdx++] = i   + N* (j);
            indices[triIdx++] = i+1 + N* (j+1);
            indices[triIdx++] = i   + N* (j+1);
        }
    }
    if (triIdx != indices.size()) {
        std::cerr << "Wrong in Ocean idx array!";
        exit(1);
    }
    idxSize = indices.size();
    shaderData = new ShaderData("../shaders/ocean");
    shaderData->enableMatrix(MODELVIEW);
    shaderData->enableMatrix(NORMAL);
    shaderData->enableMatrix(INVERSEVIEW);
    shaderData->enableMatrix(PROJECTION);
    Matrix4 * projection = shaderData->stdMatrix4Data(PROJECTION);
    *projection = Engine::instance().camera()->projectionMtx();
    shaderData->enableMatrix(LIGHTVIEW);
    Matrix4 * lightView = shaderData->stdMatrix4Data(LIGHTVIEW);
     *lightView = Engine::instance().lightCamera()->viewMtx();
     shaderData->enableMatrix(LIGHTPROJECTION);
    Matrix4 * lightProj = shaderData->stdMatrix4Data(LIGHTPROJECTION);
    *lightProj = Engine::instance().lightCamera()->projectionMtx();

    std::string cubeMapStr("CubeMap");
    std::string cubeMapShaderStr("skyboxTex");
    std::vector<std::string> cubeMapTexs(6);
    cubeMapTexs[0] = std::string("../textures/POSITIVE_X.png");
    cubeMapTexs[1] = std::string("../textures/NEGATIVE_X.png");
    cubeMapTexs[2] = std::string("../textures/POSITIVE_Y.png");
    cubeMapTexs[3] = std::string("../textures/NEGATIVE_Y.png");
    cubeMapTexs[4] = std::string("../textures/POSITIVE_Z.png");
    cubeMapTexs[5] = std::string("../textures/NEGATIVE_Z.png");
    shaderData->addCubeTexture(cubeMapShaderStr, cubeMapStr, cubeMapTexs);

    static std::string name("OceanCUDA");
    Graphics::instance().buffersNew(name, VAO, VBO_GL, VBO_IDX);
    Graphics::instance().geometryIs(VBO_GL, VBO_IDX, vertexData, indices, VBO_DYNAMIC);
    int id = shaderData->shaderID();
    int locPos = Graphics::instance().shaderAttribLoc(id,"positionIn");
    int locPU = Graphics::instance().shaderAttribLoc(id,"partialUIn");
    int locPV = Graphics::instance().shaderAttribLoc(id,"partialVIn");
    //int locFold = Graphics::instance().shaderAttribLoc(id,"foldIn");
    int locFoamTime = Graphics::instance().shaderAttribLoc(id,"foamTimeIn");
    int locFoamAlpha = Graphics::instance().shaderAttribLoc(id,"foamAlphaIn");

    int bytes = sizeof(OceanVertex);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL, 3, bytes,locPos,0);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL, 3, bytes,locPU,12);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL, 3, bytes,locPV,24);
    //Graphics::instance().bindGeometry(id, VAO, VBO_GL, 1, bytes,locFold,36);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL, 1, bytes,locFoamTime,36);
    Graphics::instance().bindGeometry(id, VAO, VBO_GL, 1, bytes,locFoamAlpha,40);


    cudaGraphicsGLRegisterBuffer(&VBO_CUDA, VBO_GL,
            cudaGraphicsMapFlagsWriteDiscard);


    checkErrors();
}

std::vector<float> height(std::vector<std::pair<float,float> > _worldPos)
{
    if (_worldPos.size() > 5) std::cerr << "Too many boats at the same time!";
    std::vector<float> h(_worldPos.size());
    cudaGraphicsMapResources(1, &VBO_CUDA, 0);
    float* positions;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                          &num_bytes,
                                          VBO_CUDA);
    std::vector<float2> xz(_worldPos.size()*2);
    for (unsigned int k = 0 ; k < _worldPos.size(); ++k) {
        int i = _worldPos[k].first*N/WORLD_SIZE;
        int j = _worldPos[k].second*N/WORLD_SIZE;
        xz[k] = make_float2(_worldPos[k].first, _worldPos[k].second-5.6f);

        //the last +1 is to get Y value from VBO
        //*10 because each vertex has 10 floats
        int idx = (i + j * N) * 11 + 1;
        
        float temp;
        cudaMemcpy(&temp, positions+idx, sizeof(float),
                cudaMemcpyDeviceToHost);

        //std::cerr << "Height at (" << _worldPos[k].first << ", " <<
                //_worldPos[k].second << "): " << temp << std::endl;
        h[k] = temp;
        
    }

    cudaMemcpy( d_boatsXZ, &xz[0], sizeof(float) * xz.size() ,
            cudaMemcpyHostToDevice);
    updateFoam<0> CUDA_KERNEL_DIM(b2D,t2D)(_worldPos.size(),
                                                    d_boatsXZ,
                                                    WORLD_SIZE / N,
                                                    positions);
    updateFoam<N> CUDA_KERNEL_DIM(b2D,t2D)(_worldPos.size(),
                                                        d_boatsXZ,
                                                        WORLD_SIZE / N,
                                                        positions);

    cudaGraphicsUnmapResources(1, &VBO_CUDA, 0);

    checkErrors();
    return h;
}

ShaderData* oceanShaderData()
{
    return shaderData;
}

} //end namespace

}//end namespace
