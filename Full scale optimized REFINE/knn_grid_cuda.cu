// knn_grid_cuda.cu
// MEX: [idx, dists] = knn_grid_cuda(Xref, Xqry, int32(k))
//  Xref: N×3 single, Xqry: M×3 single, k: scalar int32
//  idx  : M×k int32 (1-based), dists: M×k single (Euclidean)
// Notes:
//  - MATLAB column-major is handled explicitly.
//  - CPU builds a uniform grid over Xref; GPU kernel searches expanding
//    cell layers until >=k neighbors are found.
//  - No Thrust, no device lambdas, CUDA-only + host STL for the grid build.

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <cstring>
#include <math_constants.h>   // for CUDART_INF_F


// ----------------- Utilities -----------------
static inline void mexCheck(bool cond, const char* msg) {
    if (!cond) {
        mexErrMsgIdAndTxt("knn_grid_cuda:check", "%s", msg);
    }
}

template<typename T>
static inline T clampi(T v, T lo, T hi){ return v<lo?lo:(v>hi?hi:v); }

// Load an (i)-th point from a MATLAB column-major N×3 single array
// X[i] = [ x_i, y_i, z_i ] with column stride N
__device__ __forceinline__ void load_point_colmajor(const float* X, int i, int N, float &x, float &y, float &z)
{
    x = X[i];
    y = X[i + N];
    z = X[i + 2 * N];
}

// Simple fixed-size "max at position tmax" k-heap: keep k best (smallest d2)
template<int KMAX>
struct TopK {
    float d[KMAX];
    int   i[KMAX];
    int   k;
    __device__ __forceinline__ void init(int kk){
        k = kk;
        for (int t=0; t<KMAX; ++t){ d[t] = CUDART_INF_F; i[t] = -1; }
    }
    __device__ __forceinline__ void try_insert(float dist2, int idx){
        // find worst
        int tmax = 0; float vmax = d[0];
        #pragma unroll
        for (int t=1;t<k;++t){ if (d[t] > vmax){ vmax=d[t]; tmax=t; } }
        if (dist2 < vmax){ d[tmax] = dist2; i[tmax] = idx; }
    }
    __device__ __forceinline__ void dump_sorted(float* outd, int* outi, int ld){
        // simple selection sort for k small
        for (int a=0;a<k-1;++a){
            int m=a; float vm=d[a];
            for(int b=a+1;b<k;++b){ if (d[b]<vm){ vm=d[b]; m=b; } }
            if (m!=a){ float td=d[a]; d[a]=d[m]; d[m]=td; int ti=i[a]; i[a]=i[m]; i[m]=ti; }
        }
        for (int t=0;t<k;++t){ outd[t*ld] = sqrtf(d[t]); outi[t*ld] = i[t]+1; } // 1-based
    }
};

// ----------------- Grid (host) -----------------
struct Grid {
    float3 bmin, bmax;
    float  h;
    int    nx, ny, nz;
    // sorted ref indices and cell starts
    std::vector<int>  sorted_idx;   // length Nref
    std::vector<int>  cell_start;   // length (ncells+1)
};

static inline uint64_t cell_key(int ix, int iy, int iz, int nx, int ny, int nz)
{
    // linear index; use 64-bit to be safe
    return (uint64_t)ix + (uint64_t)nx * ((uint64_t)iy + (uint64_t)ny*(uint64_t)iz);
}

static Grid build_grid_from_ref(const float* Xref, int Nref)
{
    // compute bbox (host)
    float xmin =  std::numeric_limits<float>::infinity();
    float ymin =  std::numeric_limits<float>::infinity();
    float zmin =  std::numeric_limits<float>::infinity();
    float xmax = -std::numeric_limits<float>::infinity();
    float ymax = -std::numeric_limits<float>::infinity();
    float zmax = -std::numeric_limits<float>::infinity();

    for (int i=0;i<Nref;++i){
        float x = Xref[i];
        float y = Xref[i + Nref];
        float z = Xref[i + 2*Nref];
        xmin = std::min(xmin,x); ymin = std::min(ymin,y); zmin = std::min(zmin,z);
        xmax = std::max(xmax,x); ymax = std::max(ymax,y); zmax = std::max(zmax,z);
    }
    // Pad a bit
    const float eps = 1e-4f;
    xmin-=eps; ymin-=eps; zmin-=eps;
    xmax+=eps; ymax+=eps; zmax+=eps;

    // heuristic cell size ~ 1.2 * cubic root of volume / N^(1/3)
    float dx = xmax-xmin, dy=ymax-ymin, dz=zmax-zmin;
    float vol = std::max(1e-9f, dx*dy*dz);
    float h = 1.2f * cbrtf(vol / std::max(1, Nref));
    h = std::max(h, 1e-6f);

    int nx = std::max(1, (int)ceilf(dx / h));
    int ny = std::max(1, (int)ceilf(dy / h));
    int nz = std::max(1, (int)ceilf(dz / h));

    // pair (key, index)
    struct Pair { uint64_t key; int idx; };
    std::vector<Pair> pairs; pairs.reserve(Nref);

    for (int i=0;i<Nref;++i){
        float x = Xref[i], y = Xref[i+Nref], z = Xref[i+2*Nref];
        int ix = clampi((int)floorf((x - xmin)/h), 0, nx-1);
        int iy = clampi((int)floorf((y - ymin)/h), 0, ny-1);
        int iz = clampi((int)floorf((z - zmin)/h), 0, nz-1);
        uint64_t key = cell_key(ix,iy,iz,nx,ny,nz);
        pairs.push_back({key, i});
    }

    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b){ return a.key < b.key; });

    const size_t ncells = (size_t)nx*ny*nz;
    std::vector<int> cell_start(ncells + 1, 0);
    std::vector<int> sorted_idx(Nref);

    // fill sorted_idx and cell_start
    size_t p = 0;
    for (size_t c=0;c<ncells;++c){
        uint64_t key = (uint64_t)c;
        cell_start[c] = (int)p;
        while (p<pairs.size() && pairs[p].key == key){
            sorted_idx[(size_t)cell_start[c] + (p - (size_t)cell_start[c])] = pairs[p].idx;
            ++p;
        }
    }
    cell_start[ncells] = (int)Nref;

    Grid G;
    G.bmin = make_float3(xmin,ymin,zmin);
    G.bmax = make_float3(xmax,ymax,zmax);
    G.h    = h;
    G.nx = nx; G.ny = ny; G.nz = nz;
    G.sorted_idx = std::move(sorted_idx);
    G.cell_start = std::move(cell_start);
    return G;
}

// ----------------- Device-side grid access -----------------
struct DeviceGrid {
    float3 bmin; float h;
    int nx, ny, nz;
    const int* __restrict__ cell_start;  // ncells+1
    const int* __restrict__ sorted_idx;  // Nref
    const float* __restrict__ Xref;      // Nref×3, col-major
    int Nref;
};

__device__ __forceinline__ int linear_cell(int ix,int iy,int iz,int nx,int ny,int nz){
    if (ix<0||iy<0||iz<0||ix>=nx||iy>=ny||iz>=nz) return -1;
    return ix + nx*(iy + ny*iz);
}

template<int KMAX>
__global__ void knn_kernel(const DeviceGrid G,
                           const float* __restrict__ Xqry, int Nqry,
                           int k,
                           int* __restrict__ out_idx,   // M×k column-major
                           float* __restrict__ out_dist // M×k column-major
                           )
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= Nqry) return;

    // load query
    float qx,qy,qz;
    load_point_colmajor(Xqry, m, Nqry, qx,qy,qz);

    // its cell
    int ix = (int)floorf((qx - G.bmin.x)/G.h);
    int iy = (int)floorf((qy - G.bmin.y)/G.h);
    int iz = (int)floorf((qz - G.bmin.z)/G.h);
    ix = max(0, min(ix, G.nx - 1));
    iy = max(0, min(iy, G.ny - 1));
    iz = max(0, min(iz, G.nz - 1));

    TopK<KMAX> heap; heap.init(k);

    // expand layers until we fill k (or reach reasonable limit)
    int maxL = 128; // hard cap for safety; usually small layers suffice
    int found = 0;

    for (int L=0; L<maxL; ++L){
        // visit cube of cells with Chebyshev radius L
        for (int dz=-L; dz<=L; ++dz){
            int cz = iz+dz; if (cz<0||cz>=G.nz) continue;
            for (int dy=-L; dy<=L; ++dy){
                int cy = iy+dy; if (cy<0||cy>=G.ny) continue;
                for (int dx=-L; dx<=L; ++dx){
                    int cx = ix+dx; if (cx<0||cx>=G.nx) continue;

                    // Only the "shell" of the cube for L>0 to avoid duplicates
                    if (L>0){
                        bool on_shell = (abs(dx)==L) || (abs(dy)==L) || (abs(dz)==L);
                        if (!on_shell) continue;
                    }

                    int lc = linear_cell(cx,cy,cz,G.nx,G.ny,G.nz);
                    if (lc < 0) continue;

                    int a = G.cell_start[lc];
                    int b = G.cell_start[lc+1];
                    for (int p=a; p<b; ++p){
                        int j = G.sorted_idx[p]; // ref index
                        float rx,ry,rz;
                        load_point_colmajor(G.Xref, j, G.Nref, rx,ry,rz);
                        float dx = qx-rx, dy = qy-ry, dz = qz-rz;
                        float d2 = dx*dx + dy*dy + dz*dz;
                        heap.try_insert(d2, j);
                    }
                }
            }
        }

        // early stop if we have k finite neighbors
        found = 0;
        for (int t=0;t<k;++t) if (heap.i[t]>=0) ++found;
        if (found >= k) break;
    }

    // write out column-major; one column per neighbor t
    // out row stride = Nqry
    for (int t=0;t<k;++t){
        int row = m + t*Nqry;
        if (heap.i[t] >= 0){
            out_idx[row]  = heap.i[t] + 1;              // 1-based
            out_dist[row] = sqrtf(heap.d[t]);
        } else {
            out_idx[row]  = 0;
            out_dist[row] = CUDART_INF_F;
        }
    }

    // final pass: sort the k neighbors by distance (in-place columns)
    // (small k, simple selection sort)
    // Read back, sort locally, and write again to same positions
    float td[ KMAX ];
    int   ti[ KMAX ];
    for (int t=0;t<k;++t){ td[t]=out_dist[m + t*Nqry]; ti[t]=out_idx[m + t*Nqry]; }
    for (int a=0;a<k-1;++a){
        int m2=a; float vm=td[a];
        for(int b=a+1;b<k;++b){ if (td[b]<vm){ vm=td[b]; m2=b; } }
        if (m2!=a){ float x=td[a]; td[a]=td[m2]; td[m2]=x; int y=ti[a]; ti[a]=ti[m2]; ti[m2]=y; }
    }
    for (int t=0;t<k;++t){ out_dist[m + t*Nqry]=td[t]; out_idx[m + t*Nqry]=ti[t]; }
}

// ----------------- MEX entry -----------------
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mxInitGPU();

    // --------- Validate args ---------
    if (nrhs != 3)
        mexErrMsgIdAndTxt("knn_grid_cuda:args",
            "Usage: [idx, dists] = knn_grid_cuda(Xref, Xqry, int32(k))");
    if (nlhs != 2)
        mexErrMsgIdAndTxt("knn_grid_cuda:args", "Need exactly two outputs");

    const mxArray* Aref = prhs[0];
    const mxArray* Aqry = prhs[1];
    const mxArray* Ak   = prhs[2];

    if (!mxIsInt32(Ak) || mxIsComplex(Ak) || mxGetNumberOfElements(Ak) != 1)
        mexErrMsgIdAndTxt("knn_grid_cuda:type","k must be a scalar int32");
    const int k = *(int32_T*)mxGetData(Ak);
    if (k <= 0 || k > 64)
        mexErrMsgIdAndTxt("knn_grid_cuda:k","k must be in [1,64]");

    // --------- Handle CPU/GPU inputs ---------
    const bool ArefIsGPU = mxIsGPUArray(Aref);
    const bool AqryIsGPU = mxIsGPUArray(Aqry);
    const bool returnOnGPU = (ArefIsGPU || AqryIsGPU);

    // Host copies for grid build
    const float* Xref_h = nullptr;
    std::vector<float> Xref_host; // owned if we copy from GPU

    // Device pointers for kernel
    const float* d_Xref = nullptr;
    const float* d_Xqry = nullptr;
    float* d_Xref_owner = nullptr; // owned device allocations if we upload CPU data
    float* d_Xqry_owner = nullptr;

    // Dimensions
    int Nref = 0, Nqry = 0;

    // ---- Xref (N×3 single) ----
    if (ArefIsGPU) {
        // gpuArray(single), 2D, 3 cols
        mxGPUArray const* gA = mxGPUCreateFromMxArray(Aref);
        if (mxGPUGetClassID(gA) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(gA) != 2)
            mexErrMsgIdAndTxt("knn_grid_cuda:type","Xref gpuArray must be N×3 single");
        const mwSize* dimsA = mxGPUGetDimensions(gA);
        if (dimsA[1] != 3)
            mexErrMsgIdAndTxt("knn_grid_cuda:shape","Xref must be N×3 single");
        Nref = (int)dimsA[0];
        d_Xref = static_cast<const float*>(mxGPUGetDataReadOnly(gA));

        // copy device->host to build the grid
        Xref_host.resize((size_t)Nref * 3);
        cudaError_t e = cudaMemcpy(Xref_host.data(), d_Xref, sizeof(float)*Nref*3, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess)
            mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        Xref_h = Xref_host.data();

        // keep mxGPUArray handle alive until we finish
        // (we'll destroy at the end of the function)
        // store in a local so we can destroy:
        struct GuardGPU { const mxGPUArray* a; ~GuardGPU(){ if(a) mxGPUDestroyGPUArray((mxGPUArray*)a); } } guardA{gA};
        // move guard into scope end by lambda capture trick (no op here, just ensure lifetime)
        (void)guardA;
    } else {
        // CPU single
        if (mxGetClassID(Aref) != mxSINGLE_CLASS || mxGetNumberOfDimensions(Aref) != 2 || mxGetN(Aref) != 3)
            mexErrMsgIdAndTxt("knn_grid_cuda:type","Xref must be N×3 single");
        Nref = (int)mxGetM(Aref);
        Xref_h = static_cast<const float*>(mxGetData(Aref));
        // upload to device
        cudaError_t e;
        e = cudaMalloc((void**)&d_Xref_owner, sizeof(float)*Nref*3);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        e = cudaMemcpy(d_Xref_owner, Xref_h, sizeof(float)*Nref*3, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        d_Xref = d_Xref_owner;
    }

    // ---- Xqry (M×3 single) ----
    if (AqryIsGPU) {
        mxGPUArray const* gB = mxGPUCreateFromMxArray(Aqry);
        if (mxGPUGetClassID(gB) != mxSINGLE_CLASS || mxGPUGetNumberOfDimensions(gB) != 2)
            mexErrMsgIdAndTxt("knn_grid_cuda:type","Xqry gpuArray must be M×3 single");
        const mwSize* dimsB = mxGPUGetDimensions(gB);
        if (dimsB[1] != 3)
            mexErrMsgIdAndTxt("knn_grid_cuda:shape","Xqry must be M×3 single");
        Nqry = (int)dimsB[0];
        d_Xqry = static_cast<const float*>(mxGPUGetDataReadOnly(gB));
        struct GuardGPU { const mxGPUArray* b; ~GuardGPU(){ if(b) mxGPUDestroyGPUArray((mxGPUArray*)b); } } guardB{gB};
        (void)guardB;
    } else {
        if (mxGetClassID(Aqry) != mxSINGLE_CLASS || mxGetNumberOfDimensions(Aqry) != 2 || mxGetN(Aqry) != 3)
            mexErrMsgIdAndTxt("knn_grid_cuda:type","Xqry must be M×3 single");
        Nqry = (int)mxGetM(Aqry);
        const float* Xqry_h = static_cast<const float*>(mxGetData(Aqry));
        cudaError_t e;
        e = cudaMalloc((void**)&d_Xqry_owner, sizeof(float)*Nqry*3);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        e = cudaMemcpy(d_Xqry_owner, Xqry_h, sizeof(float)*Nqry*3, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        d_Xqry = d_Xqry_owner;
    }

    // --------- Build grid on host using Xref_h ----------
    Grid G = build_grid_from_ref(Xref_h, Nref);

    // --------- Move grid data to device ----------
    int   *d_sorted_idx=nullptr, *d_cell_start=nullptr;
    cudaError_t e;
    e = cudaMalloc((void**)&d_sorted_idx,  sizeof(int)*G.sorted_idx.size());
    if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
    e = cudaMalloc((void**)&d_cell_start,  sizeof(int)*G.cell_start.size());
    if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
    e = cudaMemcpy(d_sorted_idx, G.sorted_idx.data(), sizeof(int)*G.sorted_idx.size(), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
    e = cudaMemcpy(d_cell_start, G.cell_start.data(), sizeof(int)*G.cell_start.size(), cudaMemcpyHostToDevice);
    if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));

    // --------- Prepare outputs (GPU or CPU) ----------
    int  *d_out_idx  = nullptr;
    float*d_out_dist = nullptr;

    mxGPUArray *gOutIdx = nullptr, *gOutDst = nullptr;

    if (returnOnGPU) {
        // Create GPU outputs and use their device pointers directly
        mwSize dims[2] = { (mwSize)Nqry, (mwSize)k };
        gOutIdx = mxGPUCreateGPUArray(2, dims, mxINT32_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        gOutDst = mxGPUCreateGPUArray(2, dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
        d_out_idx  = static_cast<int*>(mxGPUGetData(gOutIdx));
        d_out_dist = static_cast<float*>(mxGPUGetData(gOutDst));
    } else {
        // Allocate temporary device buffers; we'll copy back to CPU outputs
        e = cudaMalloc((void**)&d_out_idx,  sizeof(int)*Nqry*k);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        e = cudaMalloc((void**)&d_out_dist, sizeof(float)*Nqry*k);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
    }

    // --------- Launch kernel ----------
    DeviceGrid dG;
    dG.bmin = G.bmin; dG.h = G.h;
    dG.nx = G.nx; dG.ny = G.ny; dG.nz = G.nz;
    dG.cell_start = d_cell_start;
    dG.sorted_idx = d_sorted_idx;
    dG.Xref = d_Xref;
    dG.Nref = Nref;

    dim3 block(256);
    dim3 grid((Nqry + block.x - 1)/block.x);
    const int KMAX = (k <= 8 ? 8 : (k<=16 ? 16 : (k<=32 ? 32 : 64)));

    switch (KMAX) {
    case 8:  knn_kernel<8>  <<<grid,block>>>(dG, d_Xqry, Nqry, k, d_out_idx, d_out_dist); break;
    case 16: knn_kernel<16> <<<grid,block>>>(dG, d_Xqry, Nqry, k, d_out_idx, d_out_dist); break;
    case 32: knn_kernel<32> <<<grid,block>>>(dG, d_Xqry, Nqry, k, d_out_idx, d_out_dist); break;
    default: knn_kernel<64> <<<grid,block>>>(dG, d_Xqry, Nqry, k, d_out_idx, d_out_dist); break;
    }
    e = cudaPeekAtLastError(); if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
    e = cudaDeviceSynchronize(); if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));

    // --------- Set MATLAB outputs ----------
    if (returnOnGPU) {
        plhs[0] = mxGPUCreateMxArrayOnGPU(gOutIdx);
        plhs[1] = mxGPUCreateMxArrayOnGPU(gOutDst);
        mxGPUDestroyGPUArray(gOutIdx);
        mxGPUDestroyGPUArray(gOutDst);
    } else {
        plhs[0] = mxCreateNumericMatrix(Nqry, k, mxINT32_CLASS, mxREAL);
        plhs[1] = mxCreateNumericMatrix(Nqry, k, mxSINGLE_CLASS, mxREAL);
        int*   out_idx  = static_cast<int*>(mxGetData(plhs[0]));
        float* out_dist = static_cast<float*>(mxGetData(plhs[1]));
        e = cudaMemcpy(out_idx,  d_out_idx,  sizeof(int)*Nqry*k,   cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        e = cudaMemcpy(out_dist, d_out_dist, sizeof(float)*Nqry*k, cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) mexErrMsgIdAndTxt("knn_grid_cuda:cuda", cudaGetErrorString(e));
        cudaFree(d_out_idx);
        cudaFree(d_out_dist);
    }

    // --------- Cleanup ----------
    if (d_Xref_owner) cudaFree(d_Xref_owner);
    if (d_Xqry_owner) cudaFree(d_Xqry_owner);
    cudaFree(d_sorted_idx);
    cudaFree(d_cell_start);
}
