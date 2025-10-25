#include "mex.h"
#include <math.h>
#include <stdbool.h>

/* Ray: x -> +inf along +X. Triangle intersection test in 3D.
   Returns 1 if ray from p crosses triangle (v0,v1,v2).
   Robustness: basic EPS; assumes closed, non-self-intersecting mesh. */

static bool rayTriIntersect(const double p[3], const double v0[3], const double v1[3], const double v2[3]) {
    const double EPS = 1e-12;
    /* Moller–Trumbore but with ray along +X: parametric p + t*[1,0,0] */
    /* Solve for intersection with triangle plane; since ray dir is [1,0,0], we can
       compute plane equation n·(x - v0) = 0; intersect x = p + [t,0,0] -> n·([p0+t - v0, p1 - v0, p2 - v0]) = 0 -> t = ... */
    /* General approach: convert triangle to 2D (Y-Z plane) and do segment test */
    /* Project onto YZ plane and check if point is left of edges with x crossover */
    /* Simpler robust approach: compute intersection of ray with triangle plane, then barycentrics. */

    /* Triangle edges */
    double e1[3] = { v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2] };
    double e2[3] = { v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2] };

    /* Plane normal */
    double n[3] = {
        e1[1]*e2[2]-e1[2]*e2[1],
        e1[2]*e2[0]-e1[0]*e2[2],
        e1[0]*e2[1]-e1[1]*e2[0]
    };
    double ndir = n[0]; /* dot(n, [1,0,0]) */

    if (fabs(ndir) < EPS) return false; /* Ray parallel to plane */

    /* t for intersection along +X */
    double w0[3] = { p[0]-v0[0], p[1]-v0[1], p[2]-v0[2] };
    double t = - (n[0]*w0[0] + n[1]*w0[1] + n[2]*w0[2]) / ndir;
    if (t <= EPS) return false; /* only count strictly positive crossings to avoid boundary double-counts */

    double x[3] = { p[0] + t, p[1], p[2] };

    /* Barycentric test */
    double v[3] = { x[0]-v0[0], x[1]-v0[1], x[2]-v0[2] };

    double d00 = e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2];
    double d01 = e1[0]*e2[0] + e1[1]*e2[1] + e1[2]*e2[2];
    double d11 = e2[0]*e2[0] + e2[1]*e2[1] + e2[2]*e2[2];
    double d20 = v[0]*e1[0] + v[1]*e1[1] + v[2]*e1[2];
    double d21 = v[0]*e2[0] + v[1]*e2[1] + v[2]*e2[2];

    double denom = d00 * d11 - d01 * d01;
    if (fabs(denom) < EPS) return false;
    double inv = 1.0 / denom;
    double a = (d11 * d20 - d01 * d21) * inv;
    double b = (d00 * d21 - d01 * d20) * inv;

    if (a >= -EPS && b >= -EPS && (a + b) <= 1.0 + EPS) return true;
    return false;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Usage: inside = mex_inpolyhedron(F, V, P)
       F: Mx3 int (1-based)
       V: Nx3 double
       P: Kx3 double
       inside: Kx1 logical
    */
    if (nrhs != 3) mexErrMsgIdAndTxt("inpoly:args","Usage: inside = mex_inpolyhedron(F,V,P)");
    const mxArray *F = prhs[0], *V = prhs[1], *P = prhs[2];

    mwSize m = mxGetM(F);
    mwSize nv = mxGetM(V);
    mwSize k = mxGetM(P);

    if (mxGetN(F)!=3 || mxGetN(V)!=3 || mxGetN(P)!=3)
        mexErrMsgIdAndTxt("inpoly:dim","F,V,P must be *x3");

    if (!mxIsDouble(V) || !mxIsDouble(P))
        mexErrMsgIdAndTxt("inpoly:type","V and P must be double");

    /* Get pointers */
    double *Vp = mxGetPr(V);
    double *Pp = mxGetPr(P);

    /* Faces: accept double or int; MATLAB is 1-based */
    double *Fp = mxGetPr(F);

    plhs[0] = mxCreateLogicalMatrix(k,1);
    mxLogical *inside = mxGetLogicals(plhs[0]);

    /* Preload triangle data */
    for (mwSize pi=0; pi<k; ++pi) {
        const double p[3] = { Pp[pi], Pp[pi + k], Pp[pi + 2*k] };
        int count = 0;
        for (mwSize fi=0; fi<m; ++fi) {
            int i0 = (int)Fp[fi] - 1;
            int i1 = (int)Fp[fi + m] - 1;
            int i2 = (int)Fp[fi + 2*m] - 1;
            if (i0<0 || i1<0 || i2<0 || i0>=nv || i1>=nv || i2>=nv) continue;

            double v0[3] = { Vp[i0], Vp[i0 + nv], Vp[i0 + 2*nv] };
            double v1[3] = { Vp[i1], Vp[i1 + nv], Vp[i1 + 2*nv] };
            double v2[3] = { Vp[i2], Vp[i2 + nv], Vp[i2 + 2*nv] };

            if (rayTriIntersect(p, v0, v1, v2)) ++count;
        }
        inside[pi] = (count % 2) ? 1 : 0; /* odd crossings => inside */
    }
}
