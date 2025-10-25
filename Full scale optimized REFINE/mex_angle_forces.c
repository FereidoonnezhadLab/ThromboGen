#include "mex.h"
#define _USE_MATH_DEFINES
#include <math.h>

/* Compute angle-based forces at each node.
   MATLAB equivalent: for each node i, take unit vectors from i to its neighbors,
   ideal angle = 2*pi/valency, for each pair (v1,v2) add: w * (angle - ideal) * normalize(cross(v1,v2)).
   Inputs:
     bonds: E x 2 (double, 1-based indices)
     Points: N x 3 (double)
     weight: scalar (double)
   Output:
     F: N x 3 (double)
*/

static void normalize(double v[3])
{
    double n = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    if (n>1e-20) { v[0]/=n; v[1]/=n; v[2]/=n; } else { v[0]=v[1]=v[2]=0.0; }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3) mexErrMsgIdAndTxt("ang:args","Usage: F = mex_angle_forces(bonds, Points, weight)");
    const mxArray *B = prhs[0], *P = prhs[1], *W = prhs[2];

    if (!mxIsDouble(B) || !mxIsDouble(P) || !mxIsDouble(W))
        mexErrMsgIdAndTxt("ang:type","All inputs must be double.");
    if (mxGetN(B)!=2 || mxGetN(P)!=3)
        mexErrMsgIdAndTxt("ang:dim","bonds must be E x 2, Points must be N x 3.");

    mwSize E = mxGetM(B);
    mwSize N = mxGetM(P);

    double *Bp = mxGetPr(B);
    double *Pp = mxGetPr(P);
    double weight = mxGetScalar(W);

    /* Build adjacency (CSR-like) */
    mwSize *deg = (mwSize*)mxCalloc(N, sizeof(mwSize));
    for (mwSize e=0; e<E; ++e) {
        int i = (int)Bp[e] - 1;
        int j = (int)Bp[e + E] - 1;
        if (i>=0 && i<N && j>=0 && j<N) { deg[i]++; deg[j]++; }
    }
    mwSize *offset = (mwSize*)mxCalloc(N+1, sizeof(mwSize));
    offset[0]=0;
    for (mwSize i=0;i<N;++i) offset[i+1] = offset[i] + deg[i];

    int *adj = (int*)mxCalloc(offset[N], sizeof(int));
    mwSize *cur = (mwSize*)mxCalloc(N, sizeof(mwSize));
    for (mwSize e=0; e<E; ++e) {
        int i = (int)Bp[e] - 1;
        int j = (int)Bp[e + E] - 1;
        if (i>=0 && i<N && j>=0 && j<N) {
            adj[offset[i] + cur[i]++] = j;
            adj[offset[j] + cur[j]++] = i;
        }
    }

    plhs[0] = mxCreateDoubleMatrix(N, 3, mxREAL);
    double *Fp = mxGetPr(plhs[0]);

    for (mwSize i=0;i<N;++i) {
        mwSize d = deg[i];
        if (d < 2) continue;

        /* Collect normalized neighbor vectors */
        double *Vi = (double*)mxCalloc(d*3, sizeof(double));
        double xi = Pp[i], yi = Pp[i+N], zi = Pp[i+2*N];
        for (mwSize k=0;k<d;++k) {
            int j = adj[offset[i]+k];
            double v[3] = { Pp[j] - xi, Pp[j+N] - yi, Pp[j+2*N] - zi };
            normalize(v);
            Vi[k] = v[0]; Vi[k+d] = v[1]; Vi[k+2*d] = v[2];
        }
        double ideal = 2.0 * M_PI / (double)d;
        double Fi[3] = {0,0,0};
        for (mwSize a=0;a<d-1;++a) {
            double v1[3] = { Vi[a], Vi[a+d], Vi[a+2*d] };
            for (mwSize b=a+1;b<d;++b) {
                double v2[3] = { Vi[b], Vi[b+d], Vi[b+2*d] };
                double dotv = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
                if (dotv >  1.0) dotv = 1.0;
                if (dotv < -1.0) dotv = -1.0;
                double ang = acos(dotv);
                double adiff = ang - ideal;
                /* torque dir = normalized cross(v1,v2) */
                double tor[3] = { v1[1]*v2[2] - v1[2]*v2[1],
                                  v1[2]*v2[0] - v1[0]*v2[2],
                                  v1[0]*v2[1] - v1[1]*v2[0] };
                normalize(tor);
                Fi[0] += weight * adiff * tor[0];
                Fi[1] += weight * adiff * tor[1];
                Fi[2] += weight * adiff * tor[2];
            }
        }
        Fp[i]        = Fi[0];
        Fp[i+N]      = Fi[1];
        Fp[i+2*N]    = Fi[2];
        mxFree(Vi);
    }

    mxFree(deg); mxFree(offset); mxFree(adj); mxFree(cur);
}
