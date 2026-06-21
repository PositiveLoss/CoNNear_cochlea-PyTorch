#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265358979323846

typedef struct tridiag_matrix {
    double *a;
    double *b;
    double *c;
} Tridiag_M;

static inline double interpl_4(
    const double a,
    const double b,
    const double c,
    const double d,
    const double frac
) {
    (void)a;
    (void)d;
    return b * (1.0 - frac) + c * frac;
}

static double cubic_interpolate(
    const double y0,
    const double y1,
    const double y2,
    const double y3,
    const double mu
) {
    const double mu2 = mu * mu;
    const double a0 = y3 - y2 - y0 + y1;
    const double a1 = y0 - y1 - a0;
    const double a2 = y2 - y0;
    const double a3 = y1;

    return a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3;
}

static inline double cos_interpl(
    const double a,
    const double b,
    const double frac
) {
    const double mu2 = (1.0 - cos(frac * PI)) / 2.0;
    return a * (1.0 - mu2) + b * mu2;
}

void solve_tridiagonal(Tridiag_M *t, double *r, double *x, const int N) {
    if (N <= 0) {
        return;
    }

    double *cprime = malloc((size_t)N * sizeof(*cprime));
    if (cprime == NULL) {
        fprintf(stderr, "solve_tridiagonal: failed to allocate scratch space\n");
        return;
    }

    cprime[0] = t->c[0] / t->b[0];
    x[0] = r[0] / t->b[0];

    /* loop from 1 to N - 1 inclusive */
    int in;
    for (in = 1; in < N; in++) {
        double m = 1.0 / (t->b[in] - t->a[in] * cprime[in - 1]);
        cprime[in] = t->c[in] * m;
        x[in] = (r[in] - t->a[in] * x[in - 1]) * m;
    }
    /* loop from N - 2 to 0 inclusive, safely testing loop end condition */
    for (in = N - 1; in-- > 0; ){
        x[in] = x[in] - cprime[in] * x[in + 1]; /*wrong cprime[in] ebasta!*/
    }
    /* free scratch space */
    free(cprime);
}

void delay_line(
    double *Y,
    int *delay0,
    int *delay1,
    int *delay2,
    int *delay3,
    double *dev,
    double *out,
    const int M,
    const int N
) {
    for (int i = 0; i < N; i++) {
        const int k = M * i;
        if (dev[i] < 1.0) {
            out[i] = cubic_interpolate(
                Y[k + delay0[i]],
                Y[k + delay1[i]],
                Y[k + delay2[i]],
                Y[k + delay3[i]],
                dev[i]
            );
        } else {
            out[i] = cubic_interpolate(
                Y[k + (delay0[i] + 1) % M],
                Y[k + (delay1[i] + 1) % M],
                Y[k + (delay2[i] + 1) % M],
                Y[k + (delay3[i] + 1) % M],
                dev[i] - 1.0
            );
        }
    }
}

void calculate_g(
    double *V,
    double *Y,
    double *sherad_factor,
    double *sheraD,
    double *sheraRho,
    double *Yzweig,
    double *omega,
    double *g,
    const double d_m_factor,
    const int n
) {
    g[0] = d_m_factor * V[0];
    for (int i = 1; i < n; i++) {
        g[i] = sherad_factor[i] * sheraD[i] * V[i]
            + omega[i] * omega[i] * (Y[i] + sheraRho[i] * Yzweig[i]);
    }
}
