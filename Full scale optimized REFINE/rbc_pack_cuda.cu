// rbc_pack_cuda.cu
// MEX: Pack many RBC instances by transforming a canonical voxel cloud.
//
// MATLAB usage:
//   [points_out, kept_idx, kept_offsets] = rbc_pack_cuda( ...
//       single(Pt_local), single(centers), single(euler_xyz), ...
//       single(C_s), single(R_s), single(bbmin), single(bbmax), int32(min_pts));
//
// Inputs:
//   Pt_local   : M x 3 (single) canonical RBC voxels (local coords)
//   centers    : B x 3 (single) candidate centers (um)
//   euler_xyz  : B x 3 (single) rotations (radians), XYZ intrinsic
//   C_s        : 1 x 3 (single) sphere center
//   R_s        : 1 x 1 (single) sphere radius
//   bbmin,bbmax: 1 x 3 (single) simulation bounds
//   min_pts    : scalar int32, min #voxels per instance
//
// Outputs:
//   points_out  : P x 3 (single) concatenated accepted voxels (global coords)
//   kept_idx    : K x 1 (int32) indices (1..B) of accepted instances
//   kept_offsets: (K+1) x 1 (int32) prefix sum into points_out (offsets)
//
// Compile:
//   mexcuda -O -DNDEBUG rbc_pack_cuda.cu

#include "mex.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <cstdint>
#include <vector>

// Thrust
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ mexErrMsgIdAndTxt("rbc_pack_cuda:CUDA", cudaGetErrorString(e)); } } while(0)

#ifdef __CUDA_ARCH__
#define INF_F (__int_as_float(0x7f800000))
#else
#define INF_F (std::numeric_limits<float>::infinity())
#endif

// ---- predicate to avoid extended lambdas ----
struct KeepPred {
    __host__ __device__ bool operator()(unsigned char k) const { return k != 0; }
};

// Build rotation matrix Rz*Ry*Rx for Euler XYZ
__device__ inline void eulXYZ_to_R(const float3 ang, float R[9]) {
    float cx = cosf(ang.x), sx = sinf(ang.x);
    float cy = cosf(ang.y), sy = sinf(ang.y);
    float cz = cosf(ang.z), sz = sinf(ang.z);
    float Rz[9] = { cz,-sz,0,  sz,cz,0,  0,0,1 };
    float Ry[9] = { cy,0,sy,  0,1,0, -sy,0,cy };
    float Rx[9] = { 1,0,0,  0,cx,-sx,  0,sx,cx };
    float T[9];
    #pragma unroll
    for(int r=0;r<3;r++){
        for(int c=0;c<3;c++){
            T[3*r+c]=Ry[3*r+0]*Rx[0+c]+Ry[3*r+1]*Rx[3+c]+Ry[3*r+2]*Rx[6+c];
        }
    }
    #pragma unroll
    for(int r=0;r<3;r++){
        for(int c=0;c<3;c++){
            R[3*r+c]=Rz[3*r+0]*T[0+c]+Rz[3*r+1]*T[3+c]+Rz[3*r+2]*T[6+c];
        }
    }
}

// Count in-bounds points per instance
__global__ void count_kernel(const float *Pt, int M,
                             const float *centers, const float *angles, int B,
                             float3 C, float R, float3 bbmin, float3 bbmax,
                             int min_pts, int *counts, unsigned char *keep)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    float3 c = make_float3(centers[3*b + 0],centers[3*b + 1],centers[3*b + 2]);
    float3 a = make_float3(angles [3*b + 0],angles [3*b + 1], angles [3*b + 2]);
    float Rm[9]; eulXYZ_to_R(a, Rm);
    int cnt = 0;
    float R2 = R*R;
    for (int i=0;i<M;i++){
        float x = Pt[i + 0*(size_t)M], y = Pt[i + 1*(size_t)M], z = Pt[i + 2*(size_t)M];
        float gx = x*Rm[0] + y*Rm[3] + z*Rm[6] + c.x;
        float gy = x*Rm[1] + y*Rm[4] + z*Rm[7] + c.y;
        float gz = x*Rm[2] + y*Rm[5] + z*Rm[8] + c.z;
        if (gx < bbmin.x || gx > bbmax.x || gy < bbmin.y || gy > bbmax.y || gz < bbmin.z || gz > bbmax.z) continue;
        float dx = gx, dy = gy, dz = gz;
        if (dx*dx + dy*dy + dz*dz > R2) continue;

        cnt++;
    }
    counts[b] = cnt;
    keep[b]   = (cnt >= min_pts) ? 1 : 0;
}

// Scatter accepted points into output buffer with offsets
__global__ void scatter_kernel(const float *Pt, int M,
                               const float *centers, const float *angles, int B,
                               float3 C, float R, float3 bbmin, float3 bbmax,
                               const int *kept_indices, const int *kept_offsets, int K,
                               int totalPts,
                               float *outX, float *outY, float *outZ)

{
    int k = blockIdx.x * blockDim.x + threadIdx.x;  // kept instance index
    if (k >= K) return;
    int b = kept_indices[k];

    float3 c = make_float3(centers[3*b + 0],centers[3*b + 1],centers[3*b + 2]);
    float3 a = make_float3(angles [3*b + 0],angles [3*b + 1],angles [3*b + 2]);
    float Rm[9]; eulXYZ_to_R(a, Rm);
    float R2 = R*R;

    const int out0 = kept_offsets[k];
    const int out1 = (k+1 < K) ? kept_offsets[k+1] : totalPts;  // slice end
    int w = 0;
    const int maxw = out1 - out0;

    for (int i=0;i<M && w<maxw;i++){
        float x = Pt[i + 0*(size_t)M];
        float y = Pt[i + 1*(size_t)M];
        float z = Pt[i + 2*(size_t)M];
        float gx = x*Rm[0] + y*Rm[3] + z*Rm[6] + c.x;
        float gy = x*Rm[1] + y*Rm[4] + z*Rm[7] + c.y;
        float gz = x*Rm[2] + y*Rm[5] + z*Rm[8] + c.z;

        // Bound & sphere check
        if (gx<bbmin.x || gx>bbmax.x || gy<bbmin.y || gy>bbmax.y || gz<bbmin.z || gz>bbmax.z) continue;
        float dx = gx, dy = gy, dz = gz;
        if (dx*dx + dy*dy + dz*dz > R2) continue;


        outX[out0 + w] = gx;
        outY[out0 + w] = gy;
        outZ[out0 + w] = gz;
        w++;
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 8)
        mexErrMsgIdAndTxt("rbc_pack_cuda:Args","Expect 8 inputs: Pt_local, centers, euler_xyz, C_s, R_s, bbmin, bbmax, min_pts");

    const mxArray *A = prhs[0], *B = prhs[1], *E = prhs[2];
    const mxArray *Cs = prhs[3], *Rs = prhs[4], *BBmin = prhs[5], *BBmax = prhs[6], *MP = prhs[7];

    if (!mxIsSingle(A) || mxGetN(A)!=3) mexErrMsgIdAndTxt("rbc_pack_cuda:Pt","Pt_local must be Mx3 single");
    if (!mxIsSingle(B) || mxGetN(B)!=3) mexErrMsgIdAndTxt("rbc_pack_cuda:C","centers must be Bx3 single");
    if (!mxIsSingle(E) || mxGetN(E)!=3) mexErrMsgIdAndTxt("rbc_pack_cuda:Ang","euler_xyz must be Bx3 single");
    if (!mxIsSingle(Cs) || mxGetNumberOfElements(Cs)!=3) mexErrMsgIdAndTxt("rbc_pack_cuda:Cs","C_s must be 1x3 single");
    if (!mxIsSingle(Rs) || mxGetNumberOfElements(Rs)!=1) mexErrMsgIdAndTxt("rbc_pack_cuda:Rs","R_s must be single scalar");
    if (!mxIsSingle(BBmin) || mxGetNumberOfElements(BBmin)!=3) mexErrMsgIdAndTxt("rbc_pack_cuda:bbmin","bbmin must be 1x3 single");
    if (!mxIsSingle(BBmax) || mxGetNumberOfElements(BBmax)!=3) mexErrMsgIdAndTxt("rbc_pack_cuda:bbmax","bbmax must be 1x3 single");
    if (!mxIsInt32(MP) || mxGetNumberOfElements(MP)!=1) mexErrMsgIdAndTxt("rbc_pack_cuda:min","min_pts must be int32");

    int M = (int)mxGetM(A);
    int Bn= (int)mxGetM(B);
    const float *hPt = (const float*)mxGetData(A);
    const float *hC  = (const float*)mxGetData(B);
    const float *hE  = (const float*)mxGetData(E);
    const float *hCs = (const float*)mxGetData(Cs);
    const float *hBB0= (const float*)mxGetData(BBmin);
    const float *hBB1= (const float*)mxGetData(BBmax);
    float R = *(const float*)mxGetData(Rs);
    int min_pts = *(const int*)mxGetData(MP);

    float3 C  = make_float3(hCs[0], hCs[1], hCs[2]);
    float3 bb0= make_float3(hBB0[0],hBB0[1],hBB0[2]);
    float3 bb1= make_float3(hBB1[0],hBB1[1],hBB1[2]);

    // Device buffers
    float *dPt=nullptr, *dC=nullptr, *dE=nullptr;
    CUDA_OK(cudaMalloc(&dPt, (size_t)M*3*sizeof(float)));
    CUDA_OK(cudaMemcpy(dPt, hPt, (size_t)M*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dC,  (size_t)Bn*3*sizeof(float)));
    CUDA_OK(cudaMemcpy(dC,  hC,  (size_t)Bn*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMalloc(&dE,  (size_t)Bn*3*sizeof(float)));
    CUDA_OK(cudaMemcpy(dE,  hE,  (size_t)Bn*3*sizeof(float), cudaMemcpyHostToDevice));

    thrust::device_vector<int> d_counts(Bn, 0);
    thrust::device_vector<unsigned char> d_keep(Bn, 0);

    // Pass 1: count
    {
        int block=256, grid=(Bn+block-1)/block;
        count_kernel<<<grid,block>>>(dPt, M, dC, dE, Bn, C, R, bb0, bb1, min_pts,
                                     thrust::raw_pointer_cast(d_counts.data()),
                                     thrust::raw_pointer_cast(d_keep.data()));
        CUDA_OK(cudaDeviceSynchronize());
    }

    // Build list of kept indices
    thrust::device_vector<int> d_idx_all(Bn);
    thrust::sequence(d_idx_all.begin(), d_idx_all.end(), 0);
    thrust::device_vector<int> d_kept_idx(Bn);
    auto end_it = thrust::copy_if(d_idx_all.begin(), d_idx_all.end(),
                                  d_keep.begin(), d_kept_idx.begin(), KeepPred());
    int K = (int)(end_it - d_kept_idx.begin());

    // Gather kept counts and exclusive-scan to offsets
    thrust::device_vector<int> d_kept_counts(K);
    if (K > 0) {
        thrust::gather(d_kept_idx.begin(), d_kept_idx.begin()+K,
                       d_counts.begin(), d_kept_counts.begin());
    }
    thrust::device_vector<int> d_offsets(K+1);
    if (K > 0) {
        d_offsets[0] = 0;
        thrust::exclusive_scan(d_kept_counts.begin(), d_kept_counts.end(), d_offsets.begin()+1);
    } else {
        d_offsets[0] = 0;
    }
    int totalPts = (K > 0) ? d_offsets[K] : 0;

    // Prepare output arrays
    mwSize dimsPts[2]; dimsPts[0] = (mwSize)totalPts; dimsPts[1] = (mwSize)3;
    mwSize dimsK  [2]; dimsK  [0] = (mwSize)K;        dimsK  [1] = (mwSize)1;
    mwSize dimsOfs[2]; dimsOfs[0] = (mwSize)(K+1);    dimsOfs[1] = (mwSize)1;

    plhs[0] = mxCreateNumericArray(2, dimsPts, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(2, dimsK,   mxINT32_CLASS,  mxREAL);
    plhs[2] = mxCreateNumericArray(2, dimsOfs, mxINT32_CLASS,  mxREAL);

    float *hOut = (float*)mxGetData(plhs[0]);
    int   *hKI  = (int*)  mxGetData(plhs[1]);
    int   *hOff = (int*)  mxGetData(plhs[2]);

    // Early exit if no points (still return kept_idx/offsets)
    if (totalPts == 0) {
        if (K > 0) {
            std::vector<int> hKIv(K);
            CUDA_OK(cudaMemcpy(hKIv.data(),
                               thrust::raw_pointer_cast(d_kept_idx.data()),
                               (size_t)K*sizeof(int),
                               cudaMemcpyDeviceToHost));
            for (int i=0;i<K;i++) hKI[i] = hKIv[i] + 1;
        }
        {
            std::vector<int> hOffv(K+1);
            CUDA_OK(cudaMemcpy(hOffv.data(),
                               thrust::raw_pointer_cast(d_offsets.data()),
                               (size_t)(K+1)*sizeof(int),
                               cudaMemcpyDeviceToHost));
            for (int i=0;i<K+1;i++) hOff[i] = hOffv[i];
        }
        cudaFree(dPt); cudaFree(dC); cudaFree(dE);
        return;
    }

    // Device outputs for points (SoA)
    float *dX=nullptr, *dY=nullptr, *dZ=nullptr;
    CUDA_OK(cudaMalloc(&dX, (size_t)totalPts*sizeof(float)));
    CUDA_OK(cudaMalloc(&dY, (size_t)totalPts*sizeof(float)));
    CUDA_OK(cudaMalloc(&dZ, (size_t)totalPts*sizeof(float)));

    // Scatter pass
    if (K > 0 && totalPts > 0)
    {
        int block=256, grid=(K+block-1)/block;
        scatter_kernel<<<grid,block>>>(dPt, M, dC, dE, Bn, C, R, bb0, bb1,
                                       thrust::raw_pointer_cast(d_kept_idx.data()),
                                       thrust::raw_pointer_cast(d_offsets.data()), K,
                                       totalPts,
                                       dX, dY, dZ);
        CUDA_OK(cudaDeviceSynchronize());
    }
    else {
        totalPts = 0;
    }
    // Interleave X,Y,Z into MATLAB [totalPts x 3]
    thrust::device_vector<float> dXYZ((size_t)totalPts*3);
    float *dXYZp = thrust::raw_pointer_cast(dXYZ.data());
    CUDA_OK(cudaMemcpy(dXYZp + 0*(size_t)totalPts, dX, (size_t)totalPts*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(dXYZp + 1*(size_t)totalPts, dY, (size_t)totalPts*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(dXYZp + 2*(size_t)totalPts, dZ, (size_t)totalPts*sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_OK(cudaMemcpy(hOut, dXYZp, (size_t)totalPts*3*sizeof(float), cudaMemcpyDeviceToHost));

    // Copy kept indices (+1 for MATLAB) and offsets back
    {
        std::vector<int> hKIv(K);
        std::vector<int> hOffv(K+1);

        if (K > 0) {
            CUDA_OK(cudaMemcpy(hKIv.data(),
                               thrust::raw_pointer_cast(d_kept_idx.data()),
                               (size_t)K*sizeof(int),
                               cudaMemcpyDeviceToHost));
        }
        CUDA_OK(cudaMemcpy(hOffv.data(),
                           thrust::raw_pointer_cast(d_offsets.data()),
                           (size_t)(K+1)*sizeof(int),
                           cudaMemcpyDeviceToHost));

        for (int i=0;i<K;i++) hKI[i] = hKIv[i] + 1;
        for (int i=0;i<K+1;i++) hOff[i]= hOffv[i];
    }

    cudaFree(dPt); cudaFree(dC); cudaFree(dE);
    cudaFree(dX); cudaFree(dY); cudaFree(dZ);
}