#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define BLK 256
#define CUDA_OK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ \
  mexErrMsgIdAndTxt("forces:cuda","CUDA %d: %s",(int)e,cudaGetErrorString(e)); } }while(0)

#if __CUDA_ARCH__ < 600
__device__ double atomicAdd_double(double* addr, double val){
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
              __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#define ATOM_ADD_D(P,V) atomicAdd_double(P,V)
#else
#define ATOM_ADD_D(P,V) atomicAdd(P,V)
#endif

// ------------ tiny conversion kernels (device) ------------
__global__ void upcast_f2d(int n, const float* a, double* b){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) b[i] = (double)a[i];
}
__global__ void downcast_d2f(int n, const double* a, float* b){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) b[i] = (float)a[i];
}

// ------------ main force kernels (device) ------------
__global__ void forces_kernel_f(const int32_t* __restrict__ bonds, int E,
                                const float*  __restrict__ P,     int N,
                                const float*  __restrict__ L,
                                float target, float k,
                                float* __restrict__ F)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = bonds[2*e + 0];
    int j = bonds[2*e + 1];
    if (i>=1 && j>=1){ i-=1; j-=1; } // accept 1-based too
    if (i<0 || i>=N || j<0 || j>=N) return;

    float xi=P[3*i+0], yi=P[3*i+1], zi=P[3*i+2];
    float xj=P[3*j+0], yj=P[3*j+1], zj=P[3*j+2];
    float vx = xj - xi, vy = yj - yi, vz = zj - zi;

    float bl = L[e];
    if (bl <= 1e-12f) return;
    float invL = 1.0f/bl;
    float fm   = -k * (bl - target);
    float fx = fm * vx * invL;
    float fy = fm * vy * invL;
    float fz = fm * vz * invL;

    atomicAdd(&F[3*i+0], -fx);
    atomicAdd(&F[3*i+1], -fy);
    atomicAdd(&F[3*i+2], -fz);
    atomicAdd(&F[3*j+0],  fx);
    atomicAdd(&F[3*j+1],  fy);
    atomicAdd(&F[3*j+2],  fz);
}

__global__ void forces_kernel_d(const int32_t* __restrict__ bonds, int E,
                                const double* __restrict__ P,     int N,
                                const double* __restrict__ L,
                                double target, double k,
                                double* __restrict__ F)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = bonds[2*e + 0];
    int j = bonds[2*e + 1];
    if (i>=1 && j>=1){ i-=1; j-=1; }
    if (i<0 || i>=N || j<0 || j>=N) return;

    double xi=P[3*i+0], yi=P[3*i+1], zi=P[3*i+2];
    double xj=P[3*j+0], yj=P[3*j+1], zj=P[3*j+2];
    double vx = xj - xi, vy = yj - yi, vz = zj - zi;

    double bl = L[e];
    if (bl <= 1e-12) return;
    double invL = 1.0/bl;
    double fm   = -k * (bl - target);
    double fx = fm * vx * invL;
    double fy = fm * vy * invL;
    double fz = fm * vz * invL;

    ATOM_ADD_D(&F[3*i+0], -fx);
    ATOM_ADD_D(&F[3*i+1], -fy);
    ATOM_ADD_D(&F[3*i+2], -fz);
    ATOM_ADD_D(&F[3*j+0],  fx);
    ATOM_ADD_D(&F[3*j+1],  fy);
    ATOM_ADD_D(&F[3*j+2],  fz);
}

// ------------ host launch wrappers ------------
static void launch_f(const int32_t* dB, int E,
                     const float* dP, int N,
                     const float* dL,
                     float target, float k,
                     float* dF)
{
    dim3 block(BLK), grid((E+BLK-1)/BLK);
    forces_kernel_f<<<grid,block>>>(dB,E,dP,N,dL,target,k,dF);
    CUDA_OK(cudaGetLastError());
}

static void launch_d(const int32_t* dB, int E,
                     const double* dP, int N,
                     const double* dL,
                     double target, double k,
                     double* dF)
{
    dim3 block(BLK), grid((E+BLK-1)/BLK);
    forces_kernel_d<<<grid,block>>>(dB,E,dP,N,dL,target,k,dF);
    CUDA_OK(cudaGetLastError());
}

// =====================================================
// MEX entry
// F = mex_forces_log_normal(bonds:int32, L, lambda, zeta2, k_fibrin, Points)
// - bonds: E×2 int32 CPU
// - L: E×1 (single/double, CPU or GPU)
// - Points: N×3 (single/double, CPU or GPU)
// - Output F matches Points' type and lives on same device as Points
// =====================================================
extern "C"
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxInitGPU();
    if (nrhs != 6)
        mexErrMsgIdAndTxt("forces:args",
            "Usage: F = mex_forces_log_normal(bonds:int32, L, lambda, zeta2, k_fibrin, Points)");

    // bonds
    const mxArray* Bmx = prhs[0];
    if (mxGetClassID(Bmx)!=mxINT32_CLASS || mxGetN(Bmx)!=2)
        mexErrMsgIdAndTxt("forces:bonds","bonds must be E×2 int32 (CPU).");
    int E = (int)mxGetM(Bmx);
    const int32_t* hB = (const int32_t*)mxGetData(Bmx);

    // scalars
    double lambda   = mxGetScalar(prhs[2]);
    double zeta2    = mxGetScalar(prhs[3]);
    double k_fibrin = mxGetScalar(prhs[4]);
    double target_d = exp(lambda + 0.5*zeta2);

    const mxArray* Lmx = prhs[1];
    const mxArray* Pmx = prhs[5];

    bool Pgpu = mxIsGPUArray(Pmx);
    int  N;

    // copy bonds to device
    int32_t* dB=nullptr;
    CUDA_OK(cudaMalloc(&dB, sizeof(int32_t)*E*2));
    CUDA_OK(cudaMemcpy(dB, hB, sizeof(int32_t)*E*2, cudaMemcpyHostToDevice));

    if (Pgpu) {
        // ============== GPU inputs / GPU output ==============
        const mxGPUArray* Pg = mxGPUCreateFromMxArray(Pmx);
        if (mxGPUGetNumberOfDimensions(Pg)!=2 || mxGPUGetDimensions(Pg)[1]!=3)
            mexErrMsgIdAndTxt("forces:Pshape","Points must be N×3.");
        N = (int)mxGPUGetDimensions(Pg)[0];
        mxClassID Pcls = mxGPUGetClassID(Pg); // SINGLE or DOUBLE

        const mxGPUArray* Lg = mxGPUCreateFromMxArray(Lmx);
        if ((int)mxGPUGetDimensions(Lg)[0] != E)
            mexErrMsgIdAndTxt("forces:Lshape","bond_lengths must be E×1.");
        mxClassID Lcls = mxGPUGetClassID(Lg);

        mwSize sz[2] = {(mwSize)N,(mwSize)3};
        mxGPUArray* Fg = mxGPUCreateGPUArray(2, sz, Pcls, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

        if (Pcls == mxDOUBLE_CLASS) {
            const double* dP = (const double*)mxGPUGetDataReadOnly(Pg);
            double* dF       = (double*)mxGPUGetData(Fg);

            const double* dL_use = NULL;
            double* dLtmp = NULL;
            if (Lcls == mxDOUBLE_CLASS) {
                dL_use = (const double*)mxGPUGetDataReadOnly(Lg);
            } else {
                // upcast L (single->double) on device
                const float* dLs = (const float*)mxGPUGetDataReadOnly(Lg);
                CUDA_OK(cudaMalloc(&dLtmp, sizeof(double)*E));
                dim3 b(256), g((E+255)/256);
                upcast_f2d<<<g,b>>>(E, dLs, dLtmp);
                CUDA_OK(cudaGetLastError());
                dL_use = dLtmp;
            }
            launch_d(dB, E, dP, N, dL_use, target_d, k_fibrin, dF);
            if (dLtmp) cudaFree(dLtmp);
        } else {
            const float* dP = (const float*)mxGPUGetDataReadOnly(Pg);
            float* dF       = (float*)mxGPUGetData(Fg);

            const float* dL_use = NULL;
            float* dLtmp = NULL;
            if (Lcls == mxSINGLE_CLASS) {
                dL_use = (const float*)mxGPUGetDataReadOnly(Lg);
            } else {
                // downcast L (double->single) on device
                const double* dLd = (const double*)mxGPUGetDataReadOnly(Lg);
                CUDA_OK(cudaMalloc(&dLtmp, sizeof(float)*E));
                dim3 b(256), g((E+255)/256);
                downcast_d2f<<<g,b>>>(E, dLd, dLtmp);
                CUDA_OK(cudaGetLastError());
                dL_use = dLtmp;
            }
            launch_f(dB, E, dP, N, dL_use, (float)target_d, (float)k_fibrin, dF);
            if (dLtmp) cudaFree(dLtmp);
        }

        plhs[0] = mxGPUCreateMxArrayOnGPU(Fg);
        mxGPUDestroyGPUArray(Fg);
        mxGPUDestroyGPUArray((mxGPUArray*)Pg);
        mxGPUDestroyGPUArray((mxGPUArray*)Lg);
    } else {
        // ============== CPU inputs / CPU output (compute on GPU) ==============
        if (mxGetN(Pmx)!=3) mexErrMsgIdAndTxt("forces:Pshape","Points must be N×3.");
        N = (int)mxGetM(Pmx);
        mxClassID Pcls = mxGetClassID(Pmx);
        mxClassID Lcls = mxGetClassID(Lmx);

        if (Pcls == mxDOUBLE_CLASS) {
            // Points -> device
            const double* hP = (const double*)mxGetData(Pmx);
            double* dP=nullptr; CUDA_OK(cudaMalloc(&dP, sizeof(double)*N*3));
            CUDA_OK(cudaMemcpy(dP, hP, sizeof(double)*N*3, cudaMemcpyHostToDevice));

            // L -> device (upcast if needed on host)
            double* dL=nullptr; CUDA_OK(cudaMalloc(&dL, sizeof(double)*E));
            if (Lcls == mxDOUBLE_CLASS) {
                CUDA_OK(cudaMemcpy(dL, mxGetData(Lmx), sizeof(double)*E, cudaMemcpyHostToDevice));
            } else {
                // host upcast
                const float* Ls = (const float*)mxGetData(Lmx);
                double* tmp = (double*)mxMalloc(sizeof(double)*E);
                for (int i=0;i<E;++i) tmp[i] = (double)Ls[i];
                CUDA_OK(cudaMemcpy(dL, tmp, sizeof(double)*E, cudaMemcpyHostToDevice));
                mxFree(tmp);
            }

            // out
            mxArray* F = mxCreateDoubleMatrix((mwSize)N,(mwSize)3,mxREAL);
            double* hF = (double*)mxGetData(F);
            double* dF=nullptr; CUDA_OK(cudaMalloc(&dF, sizeof(double)*N*3));

            launch_d(dB, E, dP, N, dL, target_d, k_fibrin, dF);
            CUDA_OK(cudaMemcpy(hF, dF, sizeof(double)*N*3, cudaMemcpyDeviceToHost));

            plhs[0] = F;
            cudaFree(dP); cudaFree(dL); cudaFree(dF);
        } else {
            // single
            const float* hP = (const float*)mxGetData(Pmx);
            float* dP=nullptr; CUDA_OK(cudaMalloc(&dP, sizeof(float)*N*3));
            CUDA_OK(cudaMemcpy(dP, hP, sizeof(float)*N*3, cudaMemcpyHostToDevice));

            float* dL=nullptr; CUDA_OK(cudaMalloc(&dL, sizeof(float)*E));
            if (Lcls == mxSINGLE_CLASS) {
                CUDA_OK(cudaMemcpy(dL, mxGetData(Lmx), sizeof(float)*E, cudaMemcpyHostToDevice));
            } else {
                // host downcast
                const double* Ld = (const double*)mxGetData(Lmx);
                float* tmp = (float*)mxMalloc(sizeof(float)*E);
                for (int i=0;i<E;++i) tmp[i] = (float)Ld[i];
                CUDA_OK(cudaMemcpy(dL, tmp, sizeof(float)*E, cudaMemcpyHostToDevice));
                mxFree(tmp);
            }

            mxArray* F = mxCreateNumericMatrix((mwSize)N,(mwSize)3,mxSINGLE_CLASS,mxREAL);
            float* hF = (float*)mxGetData(F);
            float* dF=nullptr; CUDA_OK(cudaMalloc(&dF, sizeof(float)*N*3));

            launch_f(dB, E, dP, N, dL, (float)target_d, (float)k_fibrin, dF);
            CUDA_OK(cudaMemcpy(hF, dF, sizeof(float)*N*3, cudaMemcpyDeviceToHost));

            plhs[0] = F;
            cudaFree(dP); cudaFree(dL); cudaFree(dF);
        }
    }

    cudaFree(dB);
}
