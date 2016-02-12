/*
 * m3d_nfft.c --
 *
 * Implementation of MiRA-3D model with NFFT for Yorick.
 *
 *-----------------------------------------------------------------------------
 *
 * Copyright (C) 2013-2014, Ferréol Soulez <ferreol.soulez@univ-lyon1.fr>
 * and Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
 *
 * This software is governed by the CeCILL-C license under French law and
 * abiding by the rules of distribution of free software.  You can use, modify
 * and/or redistribute the software under the terms of the CeCILL-C license as
 * circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info".
 *
 * As a counterpart to the access to the source code and rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty and the software's author, the holder of the
 * economic rights, and the successive licensors have only limited liability.
 *
 * In this respect, the user's attention is drawn to the risks associated with
 * loading, using, modifying and/or developing or reproducing the software by
 * the user in light of its specific status of free software, that may mean
 * that it is complicated to manipulate, and that also therefore means that it
 * is reserved for developers and experienced professionals having in-depth
 * computer knowledge. Users are therefore encouraged to load and test the
 * software's suitability as regards their requirements in conditions enabling
 * the security of their systems and/or data to be ensured and, more
 * generally, to use and operate it in the same conditions as regards
 * security.
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL-C license and that you accept its terms.
 *
 *-----------------------------------------------------------------------------
 */

#if !defined(DEFAULT_FFTW_FLAGS)
# error This file is supposed to be included by yor_nfft.c
#endif

static long hunt(const double x[], long n, double xp, long ip);
static void *p_new(size_t size);

#define NEW(type, number) ((type*)p_new(sizeof(type)*(number)))

/*---------------------------------------------------------------------------*/
/* MIRA-3D OPERATOR */

/* MiRA-3D operator is implementd by:
 *   m3d_op_class = class definition for Yorick
 *   m3d_op_obj_t = type for instance of MiRA-3D operator
 *   m3d_op_sub_t = type for sub-plane of MiRA-3D operator
 */

/* Structure to stores an instance of the MIRA-3D operator. */
typedef struct _m3d_op_obj m3d_op_obj_t;
typedef struct _m3d_op_sub m3d_op_sub_t;

struct _m3d_op_sub {
  nfft_plan plan;
  double *u, *v; /* output spatial frequencies */
  double w;  /* wavelength for this sub-plane */
  long n; /* number of output complex visibilities */
  int initialized; /* NFFT plan has been initialized */
};

struct _m3d_op_obj {
  m3d_op_sub_t *sub;
  long nevals;
  long m, nx, ny, nw;
  double *u, *v, *w; /* output spatial frequencies and wavelength
                        all of length M */
  double pixelsize;

  /* interpolator data */
  double *c0;         /* interpolation weight for left sub-plane */
  long *k0, *i0, *i1; /* left sub-plane index, freq. index inside left
                         sub-plane and right sub-plane (all of lenght M) */

  long complex_meas; /* flag:  complex_meas=0 if measurements are pairs of real */
  int num_threads;
};

/* Methods. */
static void m3d_free(void *);
static void m3d_print(void *);
static void m3d_eval(void *, int);
static void m3d_extract(void *, char *);

/* Class definition. */
static y_userobj_t m3d_op_class = {
  /* type_name:  */ "MiRA-3D operator",
  /* on_free:    */ m3d_free,
  /* on_print:   */ m3d_print,
  /* on_eval:    */ m3d_eval,
  /* on_extract: */ m3d_extract,
  /* uo_ops:     */ (void *)0
};

static void m3d_free(void *ptr)
{
  m3d_op_sub_t *sub;
  m3d_op_obj_t *op = (m3d_op_obj_t *)ptr;
  long k, nw;

  nw = op->nw;

  if (op->sub != NULL) {
    /* Free all sub-planes. */
    for (k = 0; k < nw; ++k) {
      sub = &op->sub[k];
      if (sub->initialized) {
        nfft_finalize(&sub->plan);
      }
      p_free(sub->u);
      p_free(sub->v);
    }
    p_free(op->sub);
  }

  /* Free other stuff. */
  p_free(op->u);
  p_free(op->v);
  p_free(op->w);
  p_free(op->c0);
  p_free(op->k0);
  p_free(op->i0);
  p_free(op->i1);
}

static void m3d_print(void *ptr)
{
  y_print("MiRA-3D operator", 1);
}

static void m3d_eval(void *ptr, int argc)
{
  double c0, c1;
  m3d_op_obj_t *op = (m3d_op_obj_t *)ptr;
  m3d_op_sub_t *sub;
  const double *src;
  double *dst;
  long dims[Y_DIMSIZE];
  long i, i0, i1, j, k, k0, m, dim, nsrc, stride, nw;
  int arg_type, job = 0;

  /* Get the direction of the transform and check number of arguments. */
  if (argc == 2) {
    arg_type = yarg_typeid(0);
    if (IS_INTEGER(arg_type) && yarg_rank(0) == 0) {
      job = ygets_i(0);
    } else if (arg_type != Y_VOID) {
      y_error("bad job for MiRA-3D operator");
    }
    yarg_drop(1);
  } else if (argc != 1) {
    y_error("syntax: op(a) or op(a, job) with op the MiRA-3D operator");
  }

  /* Get dimensions. (M = number of data, NW = number of model
     wavelengths.) */
  nw = op->nw;
  m = op->m;
  stride = op->nx*op->ny;

  /* Get argument and apply transform (NFFT and spectral interpolation). */
  arg_type = yarg_typeid(0);
  if (job) {
    /* Apply adjoint.  Argument must be complex or pairs of reals. */
    double *f0, *f1;
    const double *f_hat;
    if (arg_type == Y_COMPLEX) {
      src = ygeta_z(0, &nsrc, dims);
      if (dims[0] > 1 || nsrc != m) {
        goto bad_dims;
      }
    } else if (arg_type <= Y_DOUBLE) {
      src = ygeta_d(0, &nsrc, dims);
      if (dims[0] > 2 || dims[1] != 2 || nsrc != 2*m) {
        goto bad_dims;
      }
    } else {
      goto bad_type;
    }
    for (k = 0; k < nw; ++k) {
      sub = &op->sub[k];
      if (sub->n > 0) {
        memset(sub->plan.f, 0, sub->n*(2*sizeof(double)));
      }
    }
    for (j = 0; j < m; ++j) {
      k0 = op->k0[j];
      c0 = op->c0[j];
      if (c0 == 1.0) {
        f0 = (double *)op->sub[k0].plan.f;
        i0 = op->i0[j];
        f0[2*i0]   += src[2*j];
        f0[2*i0+1] += src[2*j+1];
      } else if (c0 == 0.0) {
        f1 = (double *)op->sub[k0 + 1].plan.f;
        i1 = op->i1[j];
        f1[2*i1]   += src[2*j];
        f1[2*i1+1] += src[2*j+1];
      } else {
        f0 = (double *)op->sub[k0].plan.f;
        f1 = (double *)op->sub[k0 + 1].plan.f;
        i0 = op->i0[j];
        i1 = op->i1[j];
        c1 = 1.0 - c0;
        f0[2*i0]   += c0*src[2*j];
        f0[2*i0+1] += c0*src[2*j+1];
        f1[2*i1]   += c1*src[2*j];
        f1[2*i1+1] += c1*src[2*j+1];
      }
    }
    yarg_drop(1); /* source no longer needed */
    if (nw == 1) {
      /* Gray case. FIXME: optimize spectral interpolation as well */
      dims[0] = 2;
      dims[1] = op->nx;
      dims[2] = op->ny;
    } else {
      /* Poly-chromatic case. */
      dims[0] = 3;
      dims[1] = op->nx;
      dims[2] = op->ny;
      dims[3] = op->nw;
    }
    dst = ypush_d(dims);
    /* Apply NFFT operator for all planes. */
#pragma omp parallel for schedule(dynamic) default(none) shared(nw,stride,op,dst)
    for (int k = 0; k < nw; ++k) {
      m3d_op_sub_t*   sub = &op->sub[k];
      if (sub->n > 0) {
        nfft_adjoint(&sub->plan);
        const double * f_hat = (const double *)sub->plan.f_hat;
        for (int i = 0; i < stride; ++i) {
          *(dst+i+k*stride) = f_hat[2*i];
        }
      }
    }
  } else {
    /* Apply direct transform. */
    const double *f0, *f1;
    double *f_hat;

    if (arg_type <= Y_DOUBLE) {
      src = ygeta_d(0, NULL, dims);

      if (!(((dims[0] == 2 && nw == 1) || (dims[0] == 3 && dims[3] == nw)) &&
            dims[1] == op->nx && dims[2] == op->ny)) {
        goto bad_dims;
      }
    } else {
      goto bad_type;
    }

    /* Apply NFFT operator for all planes. */
#pragma  omp parallel for schedule(dynamic) default(none) shared(stride,op,src,nw)
    for (int k = 0; k < nw; ++k) {
      m3d_op_sub_t*       sub = &op->sub[k];
      if (sub->n > 0) {
        double* f_hat = (double *)sub->plan.f_hat;
        for (int i = 0; i < stride; ++i) {
          f_hat[2*i] =  *(src+i+k*stride);
          f_hat[2*i+1] = 0.0;
        }
        nfft_trafo(&sub->plan);
      }
    }
    yarg_drop(1); /* source no longer needed */
    /* Push destination array and perform spectral interpolation. */
    if (op->complex_meas) {
      dims[0] = 1;
      dims[1] = m;
      dst = ypush_z(dims);
    } else {
      dims[0] = 2;
      dims[1] = 2;
      dims[2] = m;
      dst = ypush_d(dims);
    }
    for (j = 0; j < m; ++j) {
      k0 = op->k0[j];
      c0 = op->c0[j];
      if (c0 == 1.0) {
        f0 = (const double *)op->sub[k0].plan.f;
        i0 = op->i0[j];
        dst[2*j]   = f0[2*i0];
        dst[2*j+1] = f0[2*i0+1];
      } else if (c0 == 0.0) {
        f1 = (const double *)op->sub[k0 + 1].plan.f;
        i1 = op->i1[j];
        dst[2*j]   = f1[2*i1];
        dst[2*j+1] = f1[2*i1+1];
      } else {
        f0 = (const double *)op->sub[k0].plan.f;
        f1 = (const double *)op->sub[k0 + 1].plan.f;
        i0 = op->i0[j];
        i1 = op->i1[j];
        c1 = 1.0 - c0;
        dst[2*j]   = c0*f0[2*i0]   + c1*f1[2*i1];
        dst[2*j+1] = c0*f0[2*i0+1] + c1*f1[2*i1+1];
      }
    }
  }
  ++op->nevals;
  return;

 bad_dims:
  y_error("bad dimensions");
 bad_type:
  y_error("bad data type");
}

/* Implement the on_extract method to query a member of the object. */
static void m3d_extract(void *ptr, char *member)
{
  y_error("syntax error");
}

static long add_freq(m3d_op_sub_t *sub, double u, double v);

void Y_nfft_mira3d_new(int argc)
{
  m3d_op_obj_t *op;
  m3d_op_sub_t *sub;
  double c0;
  long dims[Y_DIMSIZE];
  long inp_dims[2], ovr_dims[2];
  const double *u, *v, *w, *wlist;
  m3d_op_sub_t *s0, *s1;
  double pixelsize;
  long m, mp, nx, ny, nw;
  long j, k, k0, k1, i0, i1;
  long dim;
  int monochromatic,complex_meas=0, id, num_threads=1,index;
  int iarg = argc;
  double cutoff, min_dim, ovr_fact;
  unsigned int nfft_flags;
  unsigned int fftw_flags;

  /* Set up defaults. */
  INITIALIZE;
  cutoff = default_cutoff;
  ovr_fact = 2.0;
  nfft_flags = get_nfft_flags(-1);
  fftw_flags = get_fftw_flags(-1);

  /* Get the arguments. */

  if (--iarg < 0) goto bad_nargs;
  u = ygeta_d(iarg, &m, NULL);

  if (--iarg < 0) goto bad_nargs;
  v = ygeta_d(iarg, &mp, NULL);
  if (mp != m) y_error("U and V must have the same lenght");

  if (--iarg < 0) goto bad_nargs;
  w = ygeta_d(iarg, &mp, NULL);
  if (mp != m) y_error("U, V and W must have the same lenght");

  if (--iarg < 0) goto bad_nargs;
  pixelsize = ygets_d(iarg);
  if (pixelsize <= 0.0) y_error("invalid pixel size");

  if (--iarg < 0) goto bad_nargs;
  nx = ygets_l(iarg);
  if (nx <= 0) y_error("invalid X-dimension");

  if (--iarg < 0) goto bad_nargs;
  ny = ygets_l(iarg);
  if (ny <= 0) y_error("invalid Y-dimension");

  if (--iarg < 0) goto bad_nargs;
  wlist = ygeta_d(iarg, &nw, dims);
  monochromatic = (nw == 1);
  if (dims[0] > 1) y_error("wavelengths must be a scalar or a vector");
  for (k = 1; k < nw; ++k) {
    if (wlist[k] <= wlist[k - 1]) {
      y_error("wavelengths must be strictly increasing");
    }
  }
  ARG_LOOP(iarg, argc) {
    index = yarg_key(iarg);
    if (index > 0L) {
      if (index == complex_meas_index) {
        id = yarg_typeid(--iarg);
        if (get_scalar_int(iarg, &complex_meas) != SUCCESS) {
          y_error("bad value for COMPLEX_MEAS keyword");
        }

      }
      if (index == num_threads_index) {
        id = yarg_typeid(--iarg);
        if (get_scalar_int(iarg, &num_threads) != SUCCESS) {
          y_error("bad value for NUM_THREADS keyword");
        }
      }
    }
  }

  /* Compute oversampled dimension. */
  min_dim = ovr_fact*MAX(nx, ny);
  dim = best_fft_length((long)min_dim);
  while (dim < min_dim) {
    dim = best_fft_length(dim + 1);
  }
  inp_dims[0] = nx;
  inp_dims[1] = ny;
  ovr_dims[0] = dim;
  ovr_dims[1] = dim;

  /* Create object instance. */
  op = (m3d_op_obj_t *)ypush_obj(&m3d_op_class, sizeof(m3d_op_obj_t));
  op->m = m;
  op->nx = nx;
  op->ny = ny;
  op->nw = nw;
  op->pixelsize = pixelsize;
  op->complex_meas = complex_meas;
  op->sub = NEW(m3d_op_sub_t, nw);
  op->c0 = NEW(double, m);
  op->k0 = NEW(long, m);
  op->i0 = NEW(long, m);
  op->i1 = NEW(long, m);
  op->num_threads = num_threads;
  omp_set_num_threads(num_threads);

  /* Initialize the wavelengths of the sub-planes. */
  for (k = 0; k < nw; ++k) {
    op->sub[k].w = wlist[k];
  }

  /* Build interpolator. */
  k1 = -1;
  for (j = 0; j < m; ++j) {
    k1 = hunt(wlist, nw, w[j], k1);
    k0 = k1 - 1;
    if (k0 >= 0) {
      s0 = &op->sub[k0];
      i0 = s0->n++;
    } else {
      s0 = NULL;
      i0 = -1;
    }
    if (k1 < nw) {
      s1 = &op->sub[k1];
      i1 = s1->n++;
    } else {
      s1 = NULL;
      i1 = -1;
    }
    if (s0 == NULL) {
      c0 = 0.0;
    } else if (s1 == NULL) {
      c0 = 1.0;
    } else {
      c0 = (s1->w - w[j])/(s1->w - s0->w); /* FIXME: optimization possible */
    }
    op->k0[j] = k0;
    op->i0[j] = i0;
    op->i1[j] = i1;
    op->c0[j] = c0;
  }


  /* Create the NFFT operators. */
  for (k = 0; k < nw; ++k) {
    sub = &op->sub[k];
    if (sub->n > 0) {
      sub->u = NEW(double, sub->n);
      sub->v = NEW(double, sub->n);
    }
    sub->n = 0;
  }

  for (j = 0; j < m; ++j) {
    k0 = op->k0[j];
    if (k0 >= 0) {
      i0 = add_freq(&op->sub[k0], u[j], v[j]);
    } else {
      i0 = -1;
    }
    if (k0 + 1 < nw) {
      i1 = add_freq(&op->sub[k0 + 1], u[j], v[j]);
    } else {
      i1 = -1;
    }

    op->i0[j] = i0;
    op->i1[j] = i1;
  }

  for (k = 0; k < nw; ++k) {
    sub = &op->sub[k];
    if (sub->n > 0) {
      sub->u = p_realloc(sub->u, sub->n*sizeof(double));
      sub->v = p_realloc(sub->v, sub->n*sizeof(double));
    } else {
      double *tmp_u = sub->u;
      double *tmp_v = sub->v;
      sub->u = NULL;
      sub->v = NULL;
      p_free(tmp_u);
      p_free(tmp_v);
    }
  }

  for (k = 0; k < nw; ++k) {
    sub = &op->sub[k];
    if (sub->n > 0) {
      const double *x[2];
      x[0] = sub->u;
      x[1] = sub->v;
      make_nfft_plan(&sub->plan, 2, inp_dims, ovr_dims,
                     sub->n, x, pixelsize, cutoff,
                     nfft_flags, fftw_flags);
      sub->initialized = TRUE;
    }
  }
  return;

 bad_nargs:
  y_error("bad number of arguments");
}

/* FIXME: stupid O(n^2) function */
static long add_freq(m3d_op_sub_t *sub, double u, double v)
{
  long j, n = sub->n;

  for (j = 0; j < n; ++j) {
    if (sub->u[j] == u && sub->v[j] == v) {
      return j;
    }
  }

  sub->u[j] = u;
  sub->v[j] = v;
  ++sub->n;
  return j;
}

/*---------------------------------------------------------------------------*/
/* UTILITIES */

static long hunt(const double x[], long n, double xp, long ip)
{
  /* Based on the hunt routine given in Numerical Recipes (Press, et al.,
     Cambridge University Press, 1988), section 3.4.

     Here, x[n] is a monotone array and, if xp lies in the interval
     from x[0] to x[n-1], then
       x[h-1] <= xp < x[h]  (h is the value returned by hunt), or
       x[h-1] >= xp > x[h], as x is ascending or descending
     The value 0 or n will be returned if xp lies outside the interval.
   */
  long jl, ju;
  int ascend = (x[n-1] > x[0]);

  if (ip < 1 || ip > n - 1) {
    /* Caller has declined to make an initial guess, so fall back to garden
       variety bisection method. */
    if ((xp >= x[n-1]) == ascend) return n;
    if ((xp < x[0]) == ascend) return 0;
    jl = 0;
    ju = n - 1;
  } else {
    /* Search from initial guess IP in ever increasing steps to bracket XP. */
    long inc = 1;
    jl = ip;
    if ((xp >= x[ip]) == ascend) {
      /* Search toward larger index values. */
      if (ip == n - 1) return n;
      jl = ip;
      ju = ip + inc;
      while ((xp >= x[ju]) == ascend) {
        jl = ju;
        inc += inc;
        ju += inc;
        if (ju >= n) {
          if ((xp >= x[n-1]) == ascend) return n;
          ju = n;
          break;
        }
      }
    } else {
      /* Search toward smaller index values. */
      if (ip == 0) return 0;
      ju = ip;
      jl = ip - inc;
      while ((xp < x[jl]) == ascend) {
        ju = jl;
        inc += inc;
        jl -= inc;
        if (jl < 0) {
          if ((xp < x[0]) == ascend) return 0;
          jl = 0;
          break;
        }
      }
    }
  }

  /* Have x[jl] <= xp < x[ju] for ascend; otherwise,
     have x[jl] >= xp > x[ju]. */
  while (ju - jl > 1) {
    ip = (jl + ju) >> 1;
    if ((xp >= x[ip]) == ascend) jl = ip;
    else ju = ip;
  }
  return ju;
}

static void *p_new(size_t size)
{
  void *ptr;
  if (size <= 0) {
    ptr = NULL;
  } else {
    ptr = p_malloc(size);
    memset(ptr, 0, size);
  }
  return ptr;
}
