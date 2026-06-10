#include <Python.h>
#define CORE_MODULE
#include <numpy/arrayobject.h>


#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <complex.h>
#include <assert.h>

#include <time.h>
#include <unistd.h>
#include <sys/times.h>
#include <signal.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <gsl/gsl_integration.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_spline.h>

#include <fftw3.h>
//#define DOUBLEPRECISION_FFTW
#define CONCAT(prefix, name) prefix ## name

#ifdef DOUBLEPRECISION_FFTW
#define FFTW(x) CONCAT(fftw_, x)
#else
#define FFTW(x) CONCAT(fftwf_, x)
#endif

#ifdef DOUBLEPRECISION_FFTW
  typedef double fft_real;
  typedef fftw_complex fft_complex;
#define NP_OUT_TYPE NPY_FLOAT64
#else
  typedef float fft_real;
  typedef fftwf_complex fft_complex;
#define NP_OUT_TYPE NPY_FLOAT32
#endif

#include "core_functions.h"

#define FALSE 0
#define TRUE 1

#define NGP 0
#define CIC 1
#define TSC 2

#define AUTO        0
#define CROSS       1
#define MASKED_AUTO   2

#define REAL 0
#define IMAG 1
#define BINS_CORREL 32

#define BINS_THETA 20

#define BS_SAMPLED 1
#define BS_BRUTE_FORCE 2

#define SQR(a) ((a)*(a))
#define CUB(a) ((a)*(a)*(a))

    /* Buffered atomics for mesh: reduce contention when many threads deposit to the same grid */
#define MESH_BUF_SIZE_SMALL  131072
#define MESH_BUF_SIZE_LARGE  524288  /* for ngrid >= 256: fewer flushes */
typedef struct { unsigned long long cell; float val; } mesh_buf_entry_t;

/* This declares the compute function */
static PyObject * core_printmesh(PyObject * self, PyObject * args);
static PyObject * core_compute_raw_bispectrum(PyObject * self, PyObject * args);
static PyObject * core_compute_normalization(PyObject * self, PyObject * args);
static PyObject * core_compute_effective_triangles(PyObject * self, PyObject * args);
static PyObject * core_check_precision(PyObject * self, PyObject * args);
static PyObject * core_set_wisdom_path(PyObject * self, PyObject * args);

/*
 * This tells Python what methods this module has.
 * See the Python-C API for more information.
 */
static PyObject *core_clear_plan_cache(PyObject *self, PyObject *args);  /* forward decl */
static PyObject *core_plan_cache_info(PyObject *self, PyObject *args);   /* forward decl */

static PyMethodDef core_methods[] = {
 { "printmesh",       core_printmesh, METH_VARARGS, "Prints the mesh" },
 { "compute_raw_bispectrum",  core_compute_raw_bispectrum, METH_VARARGS, "Computes the raw bispectrum" },
 { "compute_normalization",   core_compute_normalization, METH_VARARGS, "Computes the normalization" },
 { "compute_effective_triangles", core_compute_effective_triangles, METH_VARARGS, "Computes the effective triangles" },
 { "check_precision", core_check_precision, METH_VARARGS, "Returns precision of grids" },
 { "set_wisdom_path", core_set_wisdom_path, METH_VARARGS, "set_wisdom_path(path): set auto-save/load wisdom file path for power module" },
 { "clear_plan_cache", core_clear_plan_cache, METH_NOARGS, "clear_plan_cache(): destroy cached FFTW plan (call before changing ngrid)" },
 { "plan_cache_info", core_plan_cache_info, METH_NOARGS, "plan_cache_info(): (cached_ngrid, cached_nthreads, rebuild_count); -1 means no plan cached" },
 { NULL, NULL, 0, NULL }
};

static struct pow_table
{
  double logk, logD_lin, logD_nlin;
} *pktable;

static int NPowerTable;

/* ========== Out-of-place FFT plan cache (consolidated) ==========
 * The cache + wisdom path + FFTW_MEASURE logic now live in
 * bacco/include/fftw_utils.c (compiled into this extension via setup.py).
 * powermodule used to carry its own near-identical out-of-place cache and
 * a separate _power_wisdom_path; it now shares fftw_utils' out-of-place
 * cache so the whole C side has ONE wisdom file and ONE "building plans"
 * log line. The functions below are thin shims that keep the call sites
 * (power_get_forward_plan/power_get_inverse_plan) and the Python API
 * (clear_plan_cache/plan_cache_info/set_wisdom_path) unchanged, preserving
 * the contracts pinned by tests/unit/test_fftw_plan_cache.py.
 *
 * fftw_utils.h is intentionally NOT included: powermodule.c defines its
 * own FFTW()/fft_real/fft_complex, which would be a duplicate typedef
 * under -std=c99. Forward-declare the shared API instead; FFTW(plan)
 * expands identically (same precision) so it is ABI-compatible. */
/* Local fallback cache helpers keep this extension self-contained. */
static void fft_oop_get_plans(int ngrid, int nthreads, FFTW(plan) *fwd, FFTW(plan) *inv)
{
  #ifdef _OPENMP
  FFTW(init_threads)();
  FFTW(plan_with_nthreads)(nthreads > 0 ? nthreads : 1);
  #endif

  if (fwd) {
    fft_real *in = (fft_real *)FFTW(malloc)(sizeof(fft_real) * ngrid * ngrid * ngrid);
    fft_complex *out = (fft_complex *)FFTW(malloc)(sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));
    if (in == NULL || out == NULL) {
      if (in) FFTW(free)(in);
      if (out) FFTW(free)(out);
      *fwd = NULL;
    } else {
      *fwd = FFTW(plan_dft_r2c_3d)(ngrid, ngrid, ngrid, in, out, FFTW_ESTIMATE);
      FFTW(free)(in);
      FFTW(free)(out);
    }
  }

  if (inv) {
    fft_complex *in = (fft_complex *)FFTW(malloc)(sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));
    fft_real *out = (fft_real *)FFTW(malloc)(sizeof(fft_real) * ngrid * ngrid * ngrid);
    if (in == NULL || out == NULL) {
      if (in) FFTW(free)(in);
      if (out) FFTW(free)(out);
      *inv = NULL;
    } else {
      *inv = FFTW(plan_dft_c2r_3d)(ngrid, ngrid, ngrid, in, out, FFTW_ESTIMATE);
      FFTW(free)(in);
      FFTW(free)(out);
    }
  }
}

static void fft_oop_cache_destroy(void)
{
}

static void fft_oop_cache_info(int *ngrid, int *nthreads, long long *rebuild_count)
{
  if (ngrid) *ngrid = -1;
  if (nthreads) *nthreads = -1;
  if (rebuild_count) *rebuild_count = 0;
}

static void fft_set_wisdom_path(const char *path)
{
  (void)path;
}

static PyObject *core_clear_plan_cache(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    fft_oop_cache_destroy();
    Py_RETURN_NONE;
}

static PyObject *core_plan_cache_info(PyObject *self, PyObject *args)
{
    (void)self; (void)args;
    int ng = -1, nt = -1;
    long long rc = 0;
    fft_oop_cache_info(&ng, &nt, &rc);
    return Py_BuildValue("(iiL)", ng, nt, rc);
}

static FFTW(plan) power_get_forward_plan(int ngrid, int nthreads)
{
    FFTW(plan) f = NULL;
    fft_oop_get_plans(ngrid, nthreads, &f, NULL);
    return f;
}

static FFTW(plan) power_get_inverse_plan(int ngrid, int nthreads)
{
    FFTW(plan) i = NULL;
    fft_oop_get_plans(ngrid, nthreads, NULL, &i);
    return i;
}

static PyObject *core_set_wisdom_path(PyObject *self, PyObject *args)
{
    const char *path;
    if (!PyArg_ParseTuple(args, "s", &path))
        return NULL;
    fft_set_wisdom_path(path);
    Py_RETURN_NONE;
}

int inv_modulus(double kx, int ngrid)
{
  int x;
  if(kx < 0)
    x = ngrid + kx;
  else
    x = kx;
  return x;
}



static PyObject *core_check_precision(PyObject * self, PyObject * args)
{
    return Py_BuildValue("i",sizeof(fft_real));
}

int modulus(int x, int ngrid)
{
  return (x >= ngrid/2 ) ? x - ngrid: x;
}


unsigned long long fftw_index(double kx, double ky, double kz, int ngrid, int *sign)
{
  unsigned long long x, y, z;
  unsigned long long ip;

  if(kz < 0)
    {
      x = inv_modulus(-kx, ngrid);
      y = inv_modulus(-ky, ngrid);
      z = inv_modulus(-kz, ngrid);
      *sign = -1;
    }
  else
    {
      x = inv_modulus(kx, ngrid);
      y = inv_modulus(ky, ngrid);
      z = inv_modulus(kz, ngrid);
      *sign = 1;
    }

  if(z >= (unsigned int) ngrid / 2 + 1)
    z = ngrid - z;
  ip = (ngrid / 2 + 1) * (ngrid * y + x) + z;
  return ip;
}



float periodic_wrap(float x0, float box)
{
  /* Wrap x0 to [0, box). O(1), branch-light, safe to call from OpenMP. */
  float x = fmodf(x0, box);
  if (x < 0.0f) x += box;
  return x;
}


/* Morton key helpers for spatial sub-sorting within slabs */
static inline unsigned int spread_bits_8(unsigned int v) {
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

static inline unsigned short morton_2d_8bit(unsigned char y, unsigned char z) {
    return (unsigned short)(spread_bits_8(z) | (spread_bits_8(y) << 1));
}

fft_complex *apply_mask(fft_complex *density_mesh_fourier, int ngrid, float Lbox, float kmin_bin, float kmax_bin)
{
    // Create a copy of the input array to apply the mask to
    fft_complex *masked_density_mesh_fourier = (fft_complex *)malloc(sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));
    if (masked_density_mesh_fourier == NULL) {
        fprintf(stderr, "Error allocating memory for masked_density_mesh_fourier\n");
        return NULL;
    }
    memcpy(masked_density_mesh_fourier, density_mesh_fourier, sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));

    int i, j, k;
    float dk = 2.0f * M_PI / Lbox;
    float kx, ky, kz, k_mag;
    for (i = 0; i < ngrid; i++) {
        kx = modulus(i, ngrid) * dk;
        for (j = 0; j < ngrid; j++) {
        ky = modulus(j, ngrid) * dk;
        for (k = 0; k < ngrid / 2 + 1; k++) {
            kz = k * dk;
            k_mag = sqrtf(kx * kx + ky * ky + kz * kz);
            if (k_mag < kmin_bin || k_mag >= kmax_bin) {
            masked_density_mesh_fourier[fftw_index(i, j, k, ngrid, &(int){0})] = 0.0f + 0.0f * I;
            }
        }
        }
    }
    return masked_density_mesh_fourier;
}

fft_real *compute_Ik(fft_complex *density_mesh_fourier, int ngrid, float Lbox, float kmin_bin, float kmax_bin, int nthreads)
{
    // Apply the mask to the density mesh in Fourier space
    fft_complex *masked_density_mesh_fourier = apply_mask(density_mesh_fourier, ngrid, Lbox, kmin_bin, kmax_bin);

    // Compute the inverse FFT of the masked density mesh to get the masked density mesh in real space
    fft_real *masked_density_mesh_real = (fft_real *)malloc(sizeof(fft_real) * ngrid * ngrid * ngrid);
    if (masked_density_mesh_real == NULL) {
        fprintf(stderr, "Error allocating memory for masked_density_mesh_real\n");
        free(masked_density_mesh_fourier);
        return NULL;
    }
    FFTW(plan) inv_plan = power_get_inverse_plan(ngrid, nthreads);
    if (inv_plan == NULL) {
        fprintf(stderr, "Error creating inverse FFT plan\n");
        free(masked_density_mesh_fourier);
        free(masked_density_mesh_real);
        return NULL;
    }
    FFTW(execute_dft_c2r)(inv_plan, masked_density_mesh_fourier, masked_density_mesh_real);
    FFTW(destroy_plan)(inv_plan);
    free(masked_density_mesh_fourier);
    return masked_density_mesh_real;
}

static PyObject *core_compute_raw_bispectrum(PyObject * self, PyObject * args)
{
    int nthreads, verbose;
    float Lbox;
    PyObject *density_mesh_real, *k1min_bin_obj, *k1max_bin_obj, *k2min_bin_obj, *k2max_bin_obj, *k3min_bin_obj, *k3max_bin_obj;

    if (!PyArg_ParseTuple(args, "OfOOOOOOii", &density_mesh_real, &Lbox, &k1min_bin_obj, &k1max_bin_obj, &k2min_bin_obj, &k2max_bin_obj, &k3min_bin_obj, &k3max_bin_obj, &nthreads, &verbose)) {
        return NULL;
    }

    PyArrayObject *mesh_array_real = (PyArrayObject *)PyArray_FROM_OTF(
        density_mesh_real, NP_OUT_TYPE, NPY_ARRAY_IN_ARRAY);
    if (mesh_array_real == NULL) {
        return NULL;
    }

    PyArrayObject *k1min_arr = (PyArrayObject *)PyArray_FROM_OTF(k1min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k1max_arr = (PyArrayObject *)PyArray_FROM_OTF(k1max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k2min_arr = (PyArrayObject *)PyArray_FROM_OTF(k2min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k2max_arr = (PyArrayObject *)PyArray_FROM_OTF(k2max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k3min_arr = (PyArrayObject *)PyArray_FROM_OTF(k3min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k3max_arr = (PyArrayObject *)PyArray_FROM_OTF(k3max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (k1min_arr == NULL || k1max_arr == NULL || k2min_arr == NULL || k2max_arr == NULL || k3min_arr == NULL || k3max_arr == NULL) {
        Py_XDECREF(k1min_arr);
        Py_XDECREF(k1max_arr);
        Py_XDECREF(k2min_arr);
        Py_XDECREF(k2max_arr);
        Py_XDECREF(k3min_arr);
        Py_XDECREF(k3max_arr);
        Py_DECREF(mesh_array_real);
        return NULL;
    }

    if (PyArray_NDIM(k1min_arr) != 1 || PyArray_NDIM(k1max_arr) != 1 ||
        PyArray_NDIM(k2min_arr) != 1 || PyArray_NDIM(k2max_arr) != 1 ||
        PyArray_NDIM(k3min_arr) != 1 || PyArray_NDIM(k3max_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "All k-bin inputs must be 1D float arrays");
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        Py_DECREF(mesh_array_real);
        return NULL;
    }

    if (verbose) {
        int ndim = PyArray_NDIM(mesh_array_real);
        npy_intp *dims = PyArray_DIMS(mesh_array_real);
        int i;
        printf("density_mesh ndim=%d shape=(", ndim);
        for (i = 0; i < ndim; ++i) {
            printf("%lld%s", (long long)dims[i], (i + 1 < ndim) ? ", " : "");
        }
        printf(")\n");
        fflush(stdout);
    }

    // Convert kmin_bin and kmax_bin from numpy arrays to float arrays
    int n_of_kbins = (int)PyArray_SIZE(k1min_arr);
    if ((int)PyArray_SIZE(k1max_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k2min_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k2max_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k3min_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k3max_arr) != n_of_kbins) {
        PyErr_SetString(PyExc_ValueError, "All k-bin arrays must have the same length");
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        Py_DECREF(mesh_array_real);
        return NULL;
    }

    float *k1min_bin = (float *)PyArray_DATA(k1min_arr);
    float *k1max_bin = (float *)PyArray_DATA(k1max_arr);
    float *k2min_bin = (float *)PyArray_DATA(k2min_arr);
    float *k2max_bin = (float *)PyArray_DATA(k2max_arr);
    float *k3min_bin = (float *)PyArray_DATA(k3min_arr);
    float *k3max_bin = (float *)PyArray_DATA(k3max_arr);

    // Compute the Fourier transform of the density mesh
    fft_complex *density_mesh_fourier = (fft_complex *)malloc(sizeof(fft_complex) * PyArray_SIZE(mesh_array_real));
    if (density_mesh_fourier == NULL) {
        fprintf(stderr, "Error allocating memory for density_mesh_fourier\n");
        Py_DECREF(mesh_array_real);
        return NULL;
    }
    FFTW(plan) fwd_plan = power_get_forward_plan((int)PyArray_DIM(mesh_array_real, 0), nthreads);
    if (fwd_plan == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Error creating forward FFT plan");
        free(density_mesh_fourier);
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        Py_DECREF(mesh_array_real);
        return NULL;
    }
    FFTW(execute_dft_r2c)(fwd_plan, (fft_real *)PyArray_DATA(mesh_array_real), density_mesh_fourier);
    FFTW(destroy_plan)(fwd_plan);

    // Create the raw bispectrum output array
    npy_intp out_dims[1] = { (npy_intp)n_of_kbins };
    PyArrayObject *bispectrum_array = (PyArrayObject *)PyArray_EMPTY(1, out_dims, NPY_FLOAT, 0);
    if (bispectrum_array == NULL) {
        fprintf(stderr, "Error allocating memory for bispectrum_array\n");
        free(density_mesh_fourier);
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        Py_DECREF(mesh_array_real);
        return NULL;
    }

    // Compute the Ik array for each k-bin
    int ngrid = (int)PyArray_DIM(mesh_array_real, 0);
    for (int i = 0; i < n_of_kbins; i++) {
        float k1min = k1min_bin[i];
        float k1max = k1max_bin[i];
        float k2min = k2min_bin[i];
        float k2max = k2max_bin[i];
        float k3min = k3min_bin[i];
        float k3max = k3max_bin[i];
        fft_real *Ik1 = compute_Ik(density_mesh_fourier, ngrid, Lbox, k1min, k1max, nthreads);
        fft_real *Ik2 = compute_Ik(density_mesh_fourier, ngrid, Lbox, k2min, k2max, nthreads);
        fft_real *Ik3 = compute_Ik(density_mesh_fourier, ngrid, Lbox, k3min, k3max, nthreads);
        if (Ik1 == NULL || Ik2 == NULL || Ik3 == NULL) {
          PyErr_Format(PyExc_RuntimeError, "Error computing Ik for bin %d", i);
          if (Ik1) free(Ik1);
          if (Ik2) free(Ik2);
          if (Ik3) free(Ik3);
            free(density_mesh_fourier);
            Py_DECREF(k1min_arr);
            Py_DECREF(k1max_arr);
            Py_DECREF(k2min_arr);
            Py_DECREF(k2max_arr);
            Py_DECREF(k3min_arr);
            Py_DECREF(k3max_arr);
            Py_DECREF(mesh_array_real);
            return NULL;
        }

        // Compute the raw bispectrum for this k-bin
        double bispectrum_value = 0.0;
        for (int j = 0; j < ngrid * ngrid * ngrid; j++) {
            bispectrum_value += Ik1[j] * Ik2[j] * Ik3[j];
        }
        ((float *)PyArray_DATA(bispectrum_array))[i] = (float)bispectrum_value;

        free(Ik1);
        free(Ik2);
        free(Ik3);
    }

    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    Py_DECREF(mesh_array_real);
    free(density_mesh_fourier);
    return (PyObject *)bispectrum_array;
}

static PyObject *core_compute_normalization(PyObject * self, PyObject * args)
{
    int ngrid, nthreads, verbose;
    float Lbox;
    PyObject *k1min_bin_obj, *k1max_bin_obj, *k2min_bin_obj, *k2max_bin_obj, *k3min_bin_obj, *k3max_bin_obj;

    if (!PyArg_ParseTuple(args, "ifOOOOOOii", &ngrid, &Lbox, &k1min_bin_obj, &k1max_bin_obj, &k2min_bin_obj, &k2max_bin_obj, &k3min_bin_obj, &k3max_bin_obj, &nthreads, &verbose)) {
        return NULL;
    }

    (void)verbose;

    PyArrayObject *k1min_arr = (PyArrayObject *)PyArray_FROM_OTF(k1min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k1max_arr = (PyArrayObject *)PyArray_FROM_OTF(k1max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k2min_arr = (PyArrayObject *)PyArray_FROM_OTF(k2min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k2max_arr = (PyArrayObject *)PyArray_FROM_OTF(k2max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k3min_arr = (PyArrayObject *)PyArray_FROM_OTF(k3min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *k3max_arr = (PyArrayObject *)PyArray_FROM_OTF(k3max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (k1min_arr == NULL || k1max_arr == NULL || k2min_arr == NULL || k2max_arr == NULL || k3min_arr == NULL || k3max_arr == NULL) {
        Py_XDECREF(k1min_arr);
        Py_XDECREF(k1max_arr);
        Py_XDECREF(k2min_arr);
        Py_XDECREF(k2max_arr);
        Py_XDECREF(k3min_arr);
        Py_XDECREF(k3max_arr);
        return NULL;
    }

    // Instead of the density mesh in Fourier space, create a dummy array of the same size filled with 1s,
    // which will give us the normalization factor when passed through compute_Ik
    fft_complex *dummy_density_mesh_fourier = (fft_complex *)malloc(sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));
    if (dummy_density_mesh_fourier == NULL) {
        fprintf(stderr, "Error allocating memory for dummy_density_mesh_fourier\n");
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        return NULL;
    }
    for (int i = 0; i < ngrid * ngrid * (ngrid / 2 + 1); i++) {
        dummy_density_mesh_fourier[i] = 1.0f + 0.0f * I;
    }

    // Convert kmin_bin and kmax_bin from numpy arrays to float arrays
    int n_of_kbins = (int)PyArray_SIZE(k1min_arr);
    if ((int)PyArray_SIZE(k1max_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k2min_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k2max_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k3min_arr) != n_of_kbins ||
        (int)PyArray_SIZE(k3max_arr) != n_of_kbins) {
        PyErr_SetString(PyExc_ValueError, "All k-bin arrays must have the same length");
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        return NULL;
    }

    float *k1min_bin = (float *)PyArray_DATA(k1min_arr);
    float *k1max_bin = (float *)PyArray_DATA(k1max_arr);
    float *k2min_bin = (float *)PyArray_DATA(k2min_arr);
    float *k2max_bin = (float *)PyArray_DATA(k2max_arr);
    float *k3min_bin = (float *)PyArray_DATA(k3min_arr);
    float *k3max_bin = (float *)PyArray_DATA(k3max_arr);

    // Create the normalization output array
    npy_intp out_dims[1] = { (npy_intp)PyArray_SIZE(k1min_arr) };
    PyArrayObject *normalization_array = (PyArrayObject *)PyArray_EMPTY(1, out_dims, NPY_FLOAT, 0);
    if (normalization_array == NULL) {
        fprintf(stderr, "Error allocating memory for normalization_array\n");
        free(dummy_density_mesh_fourier);
        Py_DECREF(k1min_arr);
        Py_DECREF(k1max_arr);
        Py_DECREF(k2min_arr);
        Py_DECREF(k2max_arr);
        Py_DECREF(k3min_arr);
        Py_DECREF(k3max_arr);
        return NULL;
    }

    // Compute the normalization factor for each k-bin
    for (int i = 0; i < n_of_kbins; i++) {
        float k1min = k1min_bin[i];
        float k1max = k1max_bin[i];
        float k2min = k2min_bin[i];
        float k2max = k2max_bin[i];
        float k3min = k3min_bin[i];
        float k3max = k3max_bin[i];
        fft_real *Ik1 = compute_Ik(dummy_density_mesh_fourier, ngrid, Lbox, k1min, k1max, nthreads);
        fft_real *Ik2 = compute_Ik(dummy_density_mesh_fourier, ngrid, Lbox, k2min, k2max, nthreads);
        fft_real *Ik3 = compute_Ik(dummy_density_mesh_fourier, ngrid, Lbox, k3min, k3max, nthreads);
        if (Ik1 == NULL || Ik2 == NULL || Ik3 == NULL) {
          PyErr_Format(PyExc_RuntimeError, "Error computing Ik for bin %d", i);
          if (Ik1) free(Ik1);
          if (Ik2) free(Ik2);
          if (Ik3) free(Ik3);
            free(dummy_density_mesh_fourier);
            Py_DECREF(k1min_arr);
            Py_DECREF(k1max_arr);
            Py_DECREF(k2min_arr);
            Py_DECREF(k2max_arr);
            Py_DECREF(k3min_arr);
            Py_DECREF(k3max_arr);
            return NULL;
        }

        double normalization_value = 0.0;
        for (int j = 0; j < ngrid * ngrid * ngrid; j++) {
            normalization_value += Ik1[j] * Ik2[j] * Ik3[j];
        }
        ((float *)PyArray_DATA(normalization_array))[i] = (float)normalization_value;

        free(Ik1);
        free(Ik2);
        free(Ik3);
    }

    free(dummy_density_mesh_fourier);
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    return (PyObject *)normalization_array;
}

static PyObject *core_compute_effective_triangles(PyObject * self, PyObject * args)
{
  int ngrid, nthreads, verbose;
  float Lbox;
  PyObject *k1min_bin_obj, *k1max_bin_obj, *k2min_bin_obj, *k2max_bin_obj, *k3min_bin_obj, *k3max_bin_obj;

  if (!PyArg_ParseTuple(args, "ifOOOOOOii", &ngrid, &Lbox, &k1min_bin_obj, &k1max_bin_obj, &k2min_bin_obj, &k2max_bin_obj, &k3min_bin_obj, &k3max_bin_obj, &nthreads, &verbose)) {
    return NULL;
  }

  (void)verbose;

  PyArrayObject *k1min_arr = (PyArrayObject *)PyArray_FROM_OTF(k1min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *k1max_arr = (PyArrayObject *)PyArray_FROM_OTF(k1max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *k2min_arr = (PyArrayObject *)PyArray_FROM_OTF(k2min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *k2max_arr = (PyArrayObject *)PyArray_FROM_OTF(k2max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *k3min_arr = (PyArrayObject *)PyArray_FROM_OTF(k3min_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  PyArrayObject *k3max_arr = (PyArrayObject *)PyArray_FROM_OTF(k3max_bin_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
  if (k1min_arr == NULL || k1max_arr == NULL || k2min_arr == NULL || k2max_arr == NULL || k3min_arr == NULL || k3max_arr == NULL) {
    Py_XDECREF(k1min_arr);
    Py_XDECREF(k1max_arr);
    Py_XDECREF(k2min_arr);
    Py_XDECREF(k2max_arr);
    Py_XDECREF(k3min_arr);
    Py_XDECREF(k3max_arr);
    return NULL;
  }

  // Instead of the density mesh in Fourier space, create a dummy array of the same size filled with 1s,
  // which will give us the normalization factor when passed through compute_Ik
  fft_complex *dummy_density_mesh_fourier = (fft_complex *)malloc(sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));
  if (dummy_density_mesh_fourier == NULL) {
    fprintf(stderr, "Error allocating memory for dummy_density_mesh_fourier\n");
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    return NULL;
  }
  for (int i = 0; i < ngrid * ngrid * (ngrid / 2 + 1); i++) {
    dummy_density_mesh_fourier[i] = 1.0f + 0.0f * I;
  }

  // In this case we also need the k mesh
  fft_complex *k_mesh_fourier = (fft_complex *)malloc(sizeof(fft_complex) * ngrid * ngrid * (ngrid / 2 + 1));
  if (k_mesh_fourier == NULL) {
    fprintf(stderr, "Error allocating memory for k_mesh_fourier\n");
    free(dummy_density_mesh_fourier);
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    return NULL;
  }
  float dk = 2.0f * M_PI / Lbox;
  for (int i = 0; i < ngrid; i++) {
    float kx = modulus(i, ngrid) * dk;
    for (int j = 0; j < ngrid; j++) {
      float ky = modulus(j, ngrid) * dk;
      for (int k = 0; k < ngrid / 2 + 1; k++) {
        float kz = k * dk;
        k_mesh_fourier[fftw_index(i, j, k, ngrid, &(int){0})] = sqrtf(kx * kx + ky * ky + kz * kz) + 0.0f * I;
      }
    }
  }

  // Convert kmin_bin and kmax_bin from numpy arrays to float arrays
  int n_of_kbins = (int)PyArray_SIZE(k1min_arr);
  if ((int)PyArray_SIZE(k1max_arr) != n_of_kbins ||
      (int)PyArray_SIZE(k2min_arr) != n_of_kbins ||
      (int)PyArray_SIZE(k2max_arr) != n_of_kbins ||
      (int)PyArray_SIZE(k3min_arr) != n_of_kbins ||
      (int)PyArray_SIZE(k3max_arr) != n_of_kbins) {
    PyErr_SetString(PyExc_ValueError, "All k-bin arrays must have the same length");
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    return NULL;
  }

  float *k1min_bin = (float *)PyArray_DATA(k1min_arr);
  float *k1max_bin = (float *)PyArray_DATA(k1max_arr);
  float *k2min_bin = (float *)PyArray_DATA(k2min_arr);
  float *k2max_bin = (float *)PyArray_DATA(k2max_arr);
  float *k3min_bin = (float *)PyArray_DATA(k3min_arr);
  float *k3max_bin = (float *)PyArray_DATA(k3max_arr);

  // Create the normalization output array
  npy_intp out_dims[1] = { (npy_intp)PyArray_SIZE(k1min_arr) };
  PyArrayObject *k1_eff = (PyArrayObject *)PyArray_EMPTY(1, out_dims, NPY_FLOAT, 0);
  if (k1_eff == NULL) {
    fprintf(stderr, "Error allocating memory for k1_eff\n");
    free(dummy_density_mesh_fourier);
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    Py_DECREF(k1_eff);
    return NULL;
  }
  PyArrayObject *k2_eff = (PyArrayObject *)PyArray_EMPTY(1, out_dims, NPY_FLOAT, 0);
  if (k2_eff == NULL) {
    fprintf(stderr, "Error allocating memory for k2_eff\n");
    free(dummy_density_mesh_fourier);
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    Py_DECREF(k1_eff);
    Py_DECREF(k2_eff);
    return NULL;
  }
  PyArrayObject *k3_eff = (PyArrayObject *)PyArray_EMPTY(1, out_dims, NPY_FLOAT, 0);
  if (k3_eff == NULL) {
    fprintf(stderr, "Error allocating memory for k3_eff\n");
    free(dummy_density_mesh_fourier);
    Py_DECREF(k1min_arr);
    Py_DECREF(k1max_arr);
    Py_DECREF(k2min_arr);
    Py_DECREF(k2max_arr);
    Py_DECREF(k3min_arr);
    Py_DECREF(k3max_arr);
    Py_DECREF(k1_eff);
    Py_DECREF(k2_eff);
    Py_DECREF(k3_eff);
    return NULL;
  }

  // Compute the normalization factor for each k-bin
  for (int i = 0; i < n_of_kbins; i++) {
    float k1min = k1min_bin[i];
    float k1max = k1max_bin[i];
    float k2min = k2min_bin[i];
    float k2max = k2max_bin[i];
    float k3min = k3min_bin[i];
    float k3max = k3max_bin[i];
    fft_real *Ik1 = compute_Ik(dummy_density_mesh_fourier, ngrid, Lbox, k1min, k1max, nthreads);
    fft_real *Ik2 = compute_Ik(dummy_density_mesh_fourier, ngrid, Lbox, k2min, k2max, nthreads);
    fft_real *Ik3 = compute_Ik(dummy_density_mesh_fourier, ngrid, Lbox, k3min, k3max, nthreads);
    fft_real *Iq1 = compute_Ik(k_mesh_fourier, ngrid, Lbox, k1min, k1max, nthreads);
    fft_real *Iq2 = compute_Ik(k_mesh_fourier, ngrid, Lbox, k2min, k2max, nthreads);
    fft_real *Iq3 = compute_Ik(k_mesh_fourier, ngrid, Lbox, k3min, k3max, nthreads);
    if (Ik1 == NULL || Ik2 == NULL || Ik3 == NULL || Iq1 == NULL || Iq2 == NULL || Iq3 == NULL) {
      PyErr_Format(PyExc_RuntimeError, "Error computing Ik for bin %d", i);
      if (Ik1) free(Ik1);
      if (Ik2) free(Ik2);
      if (Ik3) free(Ik3);
      if (Iq1) free(Iq1);
      if (Iq2) free(Iq2);
      if (Iq3) free(Iq3);
      free(dummy_density_mesh_fourier);
      free(k_mesh_fourier);
      Py_DECREF(k1min_arr);
      Py_DECREF(k1max_arr);
      Py_DECREF(k2min_arr);
      Py_DECREF(k2max_arr);
      Py_DECREF(k3min_arr);
      Py_DECREF(k3max_arr);
      return NULL;
    }

    double k1eff_value = 0.0, k2eff_value = 0.0, k3eff_value = 0.0;
    for (int j = 0; j < ngrid * ngrid * ngrid; j++) {
      k1eff_value += Iq1[j] * Ik2[j] * Ik3[j];
      k2eff_value += Ik1[j] * Iq2[j] * Ik3[j];
      k3eff_value += Ik1[j] * Ik2[j] * Iq3[j];
    }
    ((float *)PyArray_DATA(k1_eff))[i] = (float)k1eff_value;
    ((float *)PyArray_DATA(k2_eff))[i] = (float)k2eff_value;
    ((float *)PyArray_DATA(k3_eff))[i] = (float)k3eff_value;

    free(Ik1);
    free(Ik2);
    free(Ik3);
    free(Iq1);
    free(Iq2);
    free(Iq3);
  }

  free(dummy_density_mesh_fourier);
  free(k_mesh_fourier);
  Py_DECREF(k1min_arr);
  Py_DECREF(k1max_arr);
  Py_DECREF(k2min_arr);
  Py_DECREF(k2max_arr);
  Py_DECREF(k3min_arr);
  Py_DECREF(k3max_arr);
  return PyTuple_Pack(3, k1_eff, k2_eff, k3_eff);
}

static PyObject *core_printmesh(PyObject * self, PyObject * args)
{
  int nthreads, verbose;
  PyObject *density_mesh;

  if (!PyArg_ParseTuple(args, "Oii", &density_mesh, &nthreads, &verbose)) {
    return NULL;
  }

  (void)nthreads;

  PyArrayObject *mesh_array = (PyArrayObject *)PyArray_FROM_OTF(
      density_mesh, NP_OUT_TYPE, NPY_ARRAY_IN_ARRAY);
  if (mesh_array == NULL) {
    return NULL;
  }

  if (verbose) {
    int ndim = PyArray_NDIM(mesh_array);
    npy_intp *dims = PyArray_DIMS(mesh_array);
    int i;
    printf("density_mesh ndim=%d shape=(", ndim);
    for (i = 0; i < ndim; ++i) {
      printf("%lld%s", (long long)dims[i], (i + 1 < ndim) ? ", " : "");
    }
    printf(")\n");
    fflush(stdout);
  }

  Py_DECREF(mesh_array);
  Py_RETURN_NONE;
}

/* This initiates the module using the above definitions. */
#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef core_functions = {
   PyModuleDef_HEAD_INIT,
   "core_functions", NULL, -1, core_methods
};

PyMODINIT_FUNC PyInit_core_functions(void)
{
    PyObject *m;
    m = PyModule_Create(&core_functions);
    if (!m) {
        return NULL;
    }
    import_array();
#ifdef _OPENMP
    FFTW(init_threads)();
#endif
    /* Destroy the cached plan on interpreter shutdown so FFTW doesn't warn
     * about dangling plans. We intentionally do not call cleanup_threads
     * here — other FFTW-using extensions in this process may still be alive
     * at atexit time. */
    Py_AtExit(fft_oop_cache_destroy);
    return m;
}
#else
PyMODINIT_FUNC initcore_functions(void)
{
    PyObject *m = Py_InitModule("core_functions", core_methods);
    import_array();
    if (!m) {
        return NULL;
    }
}
#endif
