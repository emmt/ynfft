/*
 * yor_nfft.c --
 *
 * Implement NFFT Yorick interface.
 *
 *-----------------------------------------------------------------------------
 *
 * Copyright (C) 2012 Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
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

/* IMPLEMENTATION NOTES
 * ====================
 *
 * The dimensions are ordered as in FFTW which is reverse order of Yorick.
 * That is Yorick dimensions list is (RANK,DIM1,DIM2,...) while for NFFT (and
 * FFTW) it is (...,DIM2,DIM1):
 *
 *   dims_yorick(t) = t-th dimension
 *                  = dims_nfft[rank - t]
 *
 * This also applies for the coordinate system used for the nodes:
 *
 *    x_yorick(j,t) = t-th coordinate of j-th node
 *                  = x_nfft[(rank - t) + (j - 1)*rank]
 *
 * with t = 1, ..., rank and j = 1, ..., num_nodes (beware of the bounds of
 * these ranges).
 *
 * Using 0-based indices, this leads to:
 *
 *    x_yorick[j + t*num_nodes] = x_nfft[(rank - 1 - t) + j*rank]
 *
 */

#include <string.h>
#include <stdio.h>
#include <yapi.h>
#include <pstdlib.h>
#include <play.h>
#include <ydata.h>

#include <omp.h>
#include <nfft3.h>
#include <fftw3.h>

static long hunt(const double x[], long n, double xp, long ip);
static void *p_new(size_t size);

#define SAFE_FREE(ptr)   if ((ptr) == NULL) /* do nothing */ ; else p_free(ptr)
#define NEW(type, number) ((type *)p_new((number)*sizeof(type)))

#define MAX(a,b)    ((a) >= (b) ? (a) : (b))
#define MIN(a,b)    ((a) <= (b) ? (a) : (b))

#define BUILTIN(name) PLUG_API void Y_nfft_##name
#define MAX_RANK 3 /* maximum number of dimensions (limit set by NFFT) */

#define FALSE      0
#define TRUE       1

#define SUCCESS    0
#define FAILURE  (-1)

#if (Y_CHAR != 0)
# error code assumes that Y_CHAR = 0
#endif
#if (Y_SHORT != 1)
# error code assumes that Y_SHORT = 1
#endif
#if (Y_INT != 2)
# error code assumes that Y_INT = 2
#endif
#if (Y_LONG != 3)
# error code assumes that Y_LONG = 3
#endif
#if (Y_FLOAT != 4)
# error code assumes that Y_FLOAT = 4
#endif
#if (Y_DOUBLE != 5)
# error code assumes that Y_DOUBLE = 5
#endif
#if (Y_COMPLEX != 6)
# error code assumes that Y_COMPLEX = 6
#endif

/* Loop over arguments of a built-in function (i is the variable index, n is
   the number of arguments). */
#define ARG_LOOP(i,n)   for (i = (n) - 1; i >= 0; --i)

/* Values returned by yarg_number. */
#define NUM_NONE    0
#define NUM_INTEGER 1
#define NUM_REAL    2
#define NUM_COMPLEX 3

#define IS_INTEGER(id)           ((id) >= Y_CHAR && (id) <= Y_LONG)
#define IS_REAL(id)              ((id) == Y_DOUBLE || (id) == Y_FLOAT)
#define IS_COMPLEX(id)           ((id) == Y_COMPLEX)
#define IS_INTEGER_OR_REAL(id)   ((id) >= Y_CHAR && (id) <= Y_DOUBLE)
#define IS_NUMERICAL(id)         ((id) >= Y_CHAR && (id) <= Y_COMPLEX)

/* Flags and options (must match values in nfft.i). */
#define YNFFT_PRE_PHI_HUT    (1 <<  0)
#define YNFFT_FG_PSI         (1 <<  1)
#define YNFFT_PRE_LIN_PSI    (1 <<  2)
#define YNFFT_PRE_FG_PSI     (1 <<  3)
#define YNFFT_PRE_PSI        (1 <<  4)
#define YNFFT_PRE_FULL_PSI   (1 <<  5)
#define YNFFT_SORT_NODES     (1 << 11)
#define YNFFT_ESTIMATE       (1 << 20)
#define YNFFT_MEASURE        (2 << 20)
#define YNFFT_PATIENT        (3 << 20)
#define YNFFT_EXHAUSTIVE     (4 << 20)
static unsigned int get_flags(unsigned int nfft_flags,
                              unsigned int fftw_flags);
static unsigned int get_nfft_flags(unsigned int flags);
static unsigned int get_fftw_flags(unsigned int flags);

/* Sets of flags/options for FFTW library. */
#define DEFAULT_FFTW_FLAGS (FFTW_ESTIMATE)
#define MINIMAL_FFTW_FLAGS (FFTW_DESTROY_INPUT)
#define ALLOWED_FFTW_FLAGS (FFTW_ESTIMATE | FFTW_MEASURE | \
                            FFTW_PATIENT  | FFTW_EXHAUSTIVE)

/* Sets of flags/options for NFFT library. */
#define SORT_NODES NFFT_SORT_NODES /* oddly this macro is the only one prefixed in NFFT */
#define DEFAULT_NFFT_FLAGS  (PRE_PHI_HUT | PRE_PSI)
#define _MINIMAL_NFFT_FLAGS  MALLOC_X  | MALLOC_F_HAT | MALLOC_F | \
                             FFTW_INIT | FFT_OUT_OF_PLACE
#ifdef _OPENMP
# define MINIMAL_NFFT_FLAGS (_MINIMAL_NFFT_FLAGS | NFFT_OMP_BLOCKWISE_ADJOINT)
#else
# define MINIMAL_NFFT_FLAGS (_MINIMAL_NFFT_FLAGS)
#endif
#define ALLOWED_NFFT_FLAGS (PRE_PHI_HUT | FG_PSI  | PRE_LIN_PSI  | \
                            PRE_FG_PSI  | PRE_PSI | PRE_FULL_PSI | \
                            SORT_NODES)

/* Indices of keywords. */
static long cutoff_index = -1L;
static long extend_index = -1L;
static long flags_index = -1L;
static long inp_dims_index = -1L;
static long nevals_index = -1L;
static long nodes_index = -1L;
static long num_nodes_index = -1L;
static long ovr_dims_index = -1L;
static long ovr_fact_index = -1L;
static long rank_index = -1L;
static long fftw_flags_index = -1L;
static long nfft_flags_index = -1L;
static long complex_meas_index = -1L;
static long num_threads_index = -1L;

/* Default value for cutoff number (negative means not yet determined). */
static long default_cutoff = -1;

#define INITIALIZE if (default_cutoff > 0) /* do nothing */; else initialize()
static void initialize(void)
{
  if (default_cutoff < 1) {
#define SET_INDEX(name) name##_index =  yget_global(#name, 0)
    SET_INDEX(cutoff);
    SET_INDEX(extend);
    SET_INDEX(flags);
    SET_INDEX(inp_dims);
    SET_INDEX(nevals);
    SET_INDEX(nodes);
    SET_INDEX(num_nodes);
    SET_INDEX(ovr_dims);
    SET_INDEX(ovr_fact);
    SET_INDEX(rank);
    SET_INDEX(nfft_flags);
    SET_INDEX(complex_meas);
    SET_INDEX(num_threads);
    SET_INDEX(fftw_flags);
#undef SET_INDEX
    {
      /* Get default size of window (in case of interupts, must be done
         last). */
      nfft_plan p;
      nfft_init_1d(&p, 100, 1);
      default_cutoff = p.m;
      nfft_finalize(&p);
    }
  }
}

/*---------------------------------------------------------------------------*/
/* MANAGE YORICK ARGUMENTS */

static int get_scalar_int(int iarg, int *ptr)
{
  if (yarg_number(iarg) == NUM_INTEGER && yarg_rank(iarg) == 0) {
    long temp = ygets_l(iarg);
    *ptr = (int)temp;
    if (*ptr == temp) {
      return SUCCESS;
    }
  }
  return FAILURE;
}

#if 0
static int get_scalar_long(int iarg, long *ptr)
{
  if (yarg_number(iarg) == NUM_INTEGER && yarg_rank(iarg) == 0) {
    *ptr = ygets_l(iarg);
    return SUCCESS;
  }
  return FAILURE;
}

static int get_scalar_double(int iarg, double *ptr)
{
  int type = yarg_number(iarg);
  if ((type == NUM_INTEGER || type == NUM_REAL) && yarg_rank(iarg) == 0) {
    *ptr = ygets_d(iarg);
    return SUCCESS;
  }
  return FAILURE;
}
#endif

/*---------------------------------------------------------------------------*/
/* OTHER UTILITY FUNCTIONS */

static int best_fft_length(int len)
{
  int best, l2, l3, l5;
  best = 2*len;
  for (l5 = 1; l5 < best; l5 *= 5) {
    for (l3 = l5; l3 < best; l3 *= 3) {
      for (l2 = l3; l2 < len; l2 *= 2) {
        ; /* empty loop body */
      }
      if (l2 == len) {
        return len;
      }
      if (l2 < best) {
        best = l2;
      }
    }
  }
  return best;
}

static const char *ordinal_suffix(long n)
{
  if (n > 0) {
    switch (n%10) {
    case 1:
      if (n != 11) return "st";
      break;
    case 2:
      if (n != 12) return "nd";
      break;
    case 3:
      if (n != 13) return "rd";
      break;
    }
  }
  return "th";
}

/*---------------------------------------------------------------------------*/
/* DECLARATION OF THE OPERATOR CLASS */

/* Structure to stores an instance of the operator. */
typedef struct _op op_t;
struct _op {
  nfft_plan plan;
  long nevals;
  int initialized;
};

#define GET_RANK(op)            ((op)->plan.d)
#define GET_NUM_NODES(op)       ((op)->plan.M_total)
#define GET_NODES(op)           ((op)->plan.x)
#define GET_INP_DIMS(op)        ((op)->plan.N)
#define GET_OVR_DIMS(op)        ((op)->plan.n)
#define GET_OVR_FACT(op)        ((op)->plan.sigma)
#define GET_NFFT_FLAGS(op)      ((op)->plan.nfft_flags)
#define GET_FFTW_FLAGS(op)      ((op)->plan.fftw_flags)
#define GET_CUTOFF(op)          ((op)->plan.m)
#define GET_NEVALS(op)          ((op)->nevals)

/* Methods. */
static void op_free(void *);
static void op_print(void *);
static void op_eval(void *, int);
static void op_extract(void *, char *);

/* Class definition. */
static y_userobj_t op_class = {
  /* type_name:  */ "NFFT operator",
  /* on_free:    */ op_free,
  /* on_print:   */ op_print,
  /* on_eval:    */ op_eval,
  /* on_extract: */ op_extract,
  /* uo_ops:     */ (void *)0
};

/*---------------------------------------------------------------------------*/
/* IMPLEMENTATION OF METHODS FOR THE OPERATOR CLASS */

static void op_free(void *ptr)
{
  op_t *op = (op_t *)ptr;
  if (op->initialized) {
    nfft_finalize(&op->plan);
  }
}

static void op_print(void *ptr)
{
  y_print("NFFT operator", 1);
}

static void op_eval(void *ptr, int argc)
{
  op_t *op = (op_t *)ptr;
  const double *src;
  double *dst;
  const int *inp_dims;
  long dims[Y_DIMSIZE];
  long t, rank, num_nodes, dim, nsrc, ndst;
  int arg_type, job = 0;

  /* Get the direction of the transform and check number of arguments. */
  if (argc == 2) {
    arg_type = yarg_typeid(0);
    if (IS_INTEGER(arg_type) && yarg_rank(0) == 0) {
      job = ygets_i(0);
    } else if (arg_type != Y_VOID) {
      y_error("bad job");
    }
    yarg_drop(1);
  } else if (argc != 1) {
    y_error("syntax: op(a) or op(a, job) with op the NFFT operator");
  }

  /*FIXME: for efficiency make the conversion and padding myself */
  /*FIXME: add a flag to pretend result is a double */
  src = ygeta_z(0, &nsrc, dims);
  rank = GET_RANK(op);
  num_nodes = GET_NUM_NODES(op);
  inp_dims = GET_INP_DIMS(op);
  if (job) {
    /* Apply adjoint. */
    if (dims[0] > 1 || nsrc != num_nodes) {
      goto bad_dims;
    }
    memcpy(op->plan.f, src, nsrc*(2*sizeof(double)));
    yarg_drop(1); /* source no longer needed */
    nfft_adjoint(&op->plan);
    dims[0] = rank;
    ndst = 1;
    for (t = 0; t < rank; ++t) {
      dim = inp_dims[rank - 1 - t];
      ndst *= dim;
      dims[1 + t] = dim;
    }
    dst = ypush_z(dims);
    memcpy(dst, op->plan.f_hat, ndst*(2*sizeof(double)));
  } else {
    /* Apply transform. */
    if (dims[0] != rank) {
      goto bad_dims;
    }
    for (t = 0; t < rank; ++t) {
      if (dims[1 + t] != inp_dims[rank - 1 - t]) {
        goto bad_dims;
      }
    }
    memcpy(op->plan.f_hat, src, nsrc*(2*sizeof(double)));
    yarg_drop(1); /* source no longer needed */
    nfft_trafo(&op->plan);
    ndst = num_nodes;
    dims[0] = (ndst > 1 ? 1 : 0);
    dims[1] = ndst;
    dst = ypush_z(dims);
    memcpy(dst, op->plan.f, ndst*(2*sizeof(double)));
  }
  ++op->nevals;
  return;

 bad_dims:
  y_error("bad dimensions");
}

/* Implement the on_extract method to query a member of the object. */
static void op_extract(void *ptr, char *member)
{
  op_t *op = (op_t *)ptr;
  long index = yget_global(member, 0);
  long dims[Y_DIMSIZE];
  long j, t, rank, num_nodes;

  /* No needs to initialize as this as laready been done by the op_new
     method. */
  if (index == ovr_fact_index) {
    double *dst;
    const double *src = GET_OVR_FACT(op);
    rank = GET_RANK(op);
    dims[0] = 1;
    dims[1] = rank;
    dst = ypush_d(dims);
    for (t = 0; t < rank; ++t) {
      dst[t] = src[rank - 1 - t];
    }
  } else if (index == rank_index) {
    long value = GET_RANK(op);
    ypush_long(value);
  } else if (index == ovr_dims_index) {
    const int *src = GET_OVR_DIMS(op);
    long *dst;
    rank = GET_RANK(op);
    dims[0] = 1;
    dims[1] = rank + 1;
    dst = ypush_l(dims);
    dst[0] = rank;
    for (t = 0; t < rank; ++t) {
      dst[1 + t] = src[rank - 1 - t];
    }
  } else if (index == inp_dims_index) {
    const int *src = GET_INP_DIMS(op);
    long *dst;
    rank = GET_RANK(op);
    dims[0] = 1;
    dims[1] = rank + 1;
    dst = ypush_l(dims);
    dst[0] = rank;
    for (t = 0; t < rank; ++t) {
      dst[1 + t] = src[rank - 1 - t];
    }
  } else if (index == nodes_index) {
    /* Copy coordinates of nodes. */
    double *dst;
    const double *src = GET_NODES(op);
    if (src == NULL) {
      y_error("creation of NFFT plan failed");
    }
    rank = GET_RANK(op);
    num_nodes = GET_NUM_NODES(op);
    dims[0] = (rank > 1 ? 2 : 1);
    dims[1] = num_nodes;
    dims[2] = rank;
    dst = ypush_d(dims);
    for (t = 0; t < rank; ++t) {
      for (j = 0; j < num_nodes; ++j) {
        /* Take into account storage order:
         *   x_yorick[j + num_nodes*t] = x_nfft[(rank - 1 - t) + j*rank]
         */
        dst[j + num_nodes*t] = src[rank - 1 - t + rank*j];
      }
    }
  } else if (index == num_nodes_index) {
    long value = GET_NUM_NODES(op);
    ypush_long(value);
  } else if (index == cutoff_index) {
    long value = GET_CUTOFF(op);
    ypush_long(value);
  } else if (index == nevals_index) {
    long value = GET_NEVALS(op);
    ypush_long(value);
  } else if (index == flags_index) {
    int value = get_flags(GET_NFFT_FLAGS(op), GET_FFTW_FLAGS(op));
    ypush_int(value);
  } else if (index == nfft_flags_index) {
    int value = GET_NFFT_FLAGS(op);
    ypush_int(value);
  } else if (index == fftw_flags_index) {
    int value = GET_FFTW_FLAGS(op);
    ypush_int(value);
  } else {
    y_error("invalid NFFT member");
    /*ypush_nil();*/
  }
}

static unsigned int get_flags(unsigned int nfft_flags,
                              unsigned int fftw_flags)
{
  unsigned int flags;

#define CASE(a) case FFTW_##a: flags = YNFFT_##a; break
  switch (fftw_flags & (FFTW_ESTIMATE|FFTW_MEASURE|FFTW_PATIENT|FFTW_EXHAUSTIVE)) {
  CASE(ESTIMATE);
  CASE(MEASURE);
  CASE(PATIENT);
  CASE(EXHAUSTIVE);
  default:
    flags = 0;
  }
#undef CASE

#define SET_FLAG(a) if ((nfft_flags & a) != 0) flags |= YNFFT_##a
  SET_FLAG(PRE_PHI_HUT);
  SET_FLAG(FG_PSI);
  SET_FLAG(PRE_LIN_PSI);
  SET_FLAG(PRE_FG_PSI);
  SET_FLAG(PRE_PSI);
  SET_FLAG(PRE_FULL_PSI);
  SET_FLAG(SORT_NODES);
#undef SET_FLAG

  return flags;
}

static unsigned int get_nfft_flags(unsigned int flags)
{
  if (flags == -1) {
    return (MINIMAL_NFFT_FLAGS | DEFAULT_NFFT_FLAGS);
  } else {
    unsigned int nfft_flags = MINIMAL_NFFT_FLAGS;
#define SET_FLAG(a) if ((flags & YNFFT_##a) != 0) nfft_flags |= a
    SET_FLAG(PRE_PHI_HUT);
    SET_FLAG(FG_PSI);
    SET_FLAG(PRE_LIN_PSI);
    SET_FLAG(PRE_FG_PSI);
    SET_FLAG(PRE_PSI);
    SET_FLAG(PRE_FULL_PSI);
    SET_FLAG(SORT_NODES);
#undef SET_FLAG
    return nfft_flags;
  }
}

static unsigned int get_fftw_flags(unsigned int flags)
{
  if (flags != -1) {
#define CASE(a) case YNFFT_##a: return (MINIMAL_FFTW_FLAGS | FFTW_##a); break
    switch (flags & (YNFFT_ESTIMATE|YNFFT_MEASURE|YNFFT_PATIENT|YNFFT_EXHAUSTIVE)) {
      CASE(ESTIMATE);
      CASE(MEASURE);
      CASE(PATIENT);
      CASE(EXHAUSTIVE);
    }
#undef CASE
  }
  return (MINIMAL_FFTW_FLAGS | DEFAULT_FFTW_FLAGS);
}

static void make_nfft_plan(nfft_plan *plan, long rank,
                           const long inp_dims[],
                           const long ovr_dims[],
                           long num_nodes, const double *x[], double xscale,
                           double cutoff,
                           unsigned int nfft_flags,
                           unsigned int fftw_flags);

/*---------------------------------------------------------------------------*/
/* BUILT-IN FUNCTIONS */

BUILTIN(version)(int argc)
{
  ypush_q(NULL)[0] = p_strcpy("0.0.1");
}

BUILTIN(indgen)(int argc)
{
  double stp;
  long k, k0, k1, len, dims[2];

  if (argc == 2) {
    stp = ygets_d(0);
    if (stp <= 0.0) {
      y_error("step size must be strictly positive");
    }
    yarg_drop(1);
  } else {
    stp = 0.0;
    if (argc != 1) {
      y_error("nfft_indgen takes one or two arguments");
    }
  }
  len = ygets_l(0);
  if (len <= 0 || len%2 != 0) {
    y_error("length must be a strictly positive even number");
  }
  dims[0] = 1;
  dims[1] = len;
  k0 = -(len/2);
  k1 = len + k0;
  if (stp > 0.0) {
    double *dst = ypush_d(dims) - k0;
    for (k = k0; k < k1; ++k) {
      dst[k] = k*stp;
    }
  } else {
    long *dst = ypush_l(dims) - k0;
    for (k = k0; k < k1; ++k) {
      dst[k] = k;
    }
  }
}

BUILTIN(new)(int argc)
{
  /* Oversampling factors. */
  double        ovr_fact_buf[MAX_RANK];
  const double *ovr_fact = NULL;
  long          ovr_fact_ntot = 0;
  int           ovr_fact_rank = -1;

  /* Oversampled dimensions. */
  const long   *ovr_dims = NULL;
  long          ovr_dims_ntot = 0;
  int           ovr_dims_rank = -1;
  long          ovr_dims_buf[MAX_RANK];

  /* Dimensions of the input space. */
  long         *inp_dims;
  long          inp_dims_ntot;
  long          inp_dims_buf[MAX_RANK];

  /* Other parameters of the transform. */
  int           rank;
  int           cutoff = -1;
  int           flags = -1, nfft_flags, fftw_flags;
  long          num_nodes = 0;
  const double *x[MAX_RANK];

  long j;
  op_t *op;
  int t;
  int iarg, packed = FALSE, positional_args = 0, arg_is_dim;
  char errbuf[256];

  /* Variables to store the characteristics of an argument. */
  long index;
  long arg_ntot, arg_dims[Y_DIMSIZE];
  int  arg_type;


  /* Setup internals. */
  INITIALIZE;

  /* Calling this builtin as a subroutine does not make sense. */
  if (yarg_subroutine()) {
    y_error("nfft_new must be called as a function");
  }

  /* First pass to count the number of positional arguments. */
  positional_args = 0;
  ARG_LOOP(iarg, argc) {
    index = yarg_key(iarg);
    if (index < 0L) {
      ++positional_args;
    } else {
      --iarg;
    }
  }
  if (positional_args < 2) {
    y_error("too few arguments");
  }
  if (positional_args > 2*MAX_RANK) {
    sprintf(errbuf, "too many arguments (max. rank = %d)", MAX_RANK);
    y_error(errbuf);
  }
  if (positional_args%2 != 0) {
    y_error("bad number of arguments");
  }
  packed = (positional_args == 2);
  inp_dims = (packed ? NULL : inp_dims_buf);
  inp_dims_ntot = 0;
  rank = 0;
  for (t = 0; t < rank; ++t) {
    inp_dims_buf[t] = -1;
  }

  /* Parse arguments. */
  arg_is_dim = FALSE;
  ARG_LOOP(iarg, argc) {
    index = yarg_key(iarg);
    if (index < 0L) {
      /* Got a positional argument. */
      arg_is_dim = !arg_is_dim; /* toggle dim/coord flag */
      if (arg_is_dim) {
        /* Parse input dimension(s). */
        if (yarg_number(iarg) != NUM_INTEGER) {
          y_error("bad type for dimension(s)");
        }
        if (packed) {
          /* This must be the first positional argument. */
          inp_dims = ygeta_l(iarg, &inp_dims_ntot, NULL);
        } else if (yarg_rank(iarg) == 0) {
          if (++rank > MAX_RANK) {
            goto too_many_dims;
          }
          inp_dims[rank - 1] = ygets_l(iarg);
        } else {
          goto expecting_salar_dim;
        }
      } else {
        /* Get coordinates. */
        arg_type = yarg_number(iarg);
        if (arg_type != NUM_INTEGER && arg_type != NUM_REAL) {
          goto bad_coord_type;
        }
        if (packed) {
          /* Get packed coordinates. */
          const double *arg_ptr;
          arg_ptr = ygeta_d(iarg, &arg_ntot, arg_dims);
          switch (arg_dims[0]) {
          case 0:
            num_nodes = 1;
            rank = 1;
            break;
          case 1:
            num_nodes = arg_dims[1];
            rank = 1;
            break;
          case 2:
            num_nodes = arg_dims[1];
            rank = arg_dims[2];
            break;
          default:
            goto bad_coord_dims;
          }
          if (rank > MAX_RANK) {
            goto too_many_dims;
          }
          if (inp_dims_ntot != rank) {
            /* Input dimensions not given as [N1,N2,...,ND]. */
            if (inp_dims_ntot == rank + 1 && inp_dims[0] == rank) {
              /* Input dimensions given as [D,N1,N2,...,ND]. */
              ++inp_dims; /* skip the leading D */
            } else {
              goto incompatible_dims;
            }
          }
          for (t = 0; t < rank; ++t) {
            x[t] = arg_ptr + num_nodes*t;
          }
        } else {
          /* Get split coordinates. */
          if (yarg_rank(iarg) > 1) {
             goto bad_coord_dims;
          }
          x[rank - 1] = ygeta_d(iarg, &arg_ntot, NULL);
          if (rank == 1) {
            num_nodes = arg_ntot;
          } else if (num_nodes != arg_ntot) {
            goto not_same_lengths;
          }
        }
      }
    } else if (index == cutoff_index) {
      --iarg;
      if (get_scalar_int(iarg, &cutoff) != SUCCESS || cutoff < 0) {
        y_error("invalid value for CUTOFF keyword");
      }
    } else if (index == flags_index) {
      --iarg;
      if (get_scalar_int(iarg, &flags) != SUCCESS) {
        y_error("invalid value for FFTW_FLAGS keyword");
      }
    } else if (index == ovr_dims_index) {
      /* Get oversampled dimensions. */
      if (ovr_fact != NULL) {
        goto ovr_fact_and_ovr_dims_exclusive;
      }
      --iarg;
      if (yarg_number(iarg) != NUM_INTEGER) {
        y_error("bad type for OVR_DIMS keyword");
      }
      ovr_dims_rank = yarg_rank(iarg);
      if (ovr_dims_rank != 0 && ovr_dims_rank != 1) {
        y_error("bad dimensions for OVR_DIMS keyword");
      }
      ovr_dims = ygeta_l(iarg, &ovr_dims_ntot, NULL);
    } else if (index == ovr_fact_index) {
      /* Get oversampling factors. */
      if (ovr_dims != NULL) {
        goto ovr_fact_and_ovr_dims_exclusive;
      }
      --iarg;
      arg_type = yarg_number(iarg);
      if (arg_type != NUM_INTEGER && arg_type != NUM_REAL) {
        y_error("bad type for OVR_FACT keyword");
      }
      ovr_fact_rank = yarg_rank(iarg);
      if (ovr_fact_rank == 0) {
        ovr_fact_buf[0] = ygets_d(iarg);
        ovr_fact = ovr_fact_buf;
        ovr_fact_ntot = 1;
      } else if (ovr_fact_rank == 1) {
        ovr_fact = ygeta_d(iarg, &ovr_fact_ntot, NULL);
      } else {
        y_error("bad dimensions for OVR_FACT keyword");
      }
    } else {
      y_error("unknown keyword");
    }
  }

  /* N_t (for t = 0,...,d-1) are the dimensions of the input array, they
   * must be even numbers and coordinates along these dimensions are:
   * -N_t/2 <= k_t < N_t/2
   *
   * Extended dimensions n_t must be strictly larger (n_t > N_t) and
   * preferably good dimensions for the FFT (by FFTW). Moreover the
   * dimensions N_t must be strictly larger than the cutoff
   * parameter m.
   */

  /* Use default cutoff if not set. */
  if (cutoff < 0) {
    cutoff = default_cutoff;
  }

  /* Check node coordinates. */
  for (t = 0; t < rank; ++t) {
    for (j = 0; j < num_nodes; ++j) {
      if (x[t][j] < -0.5 || x[t][j] >= 0.5) {
        y_error("coordinates of nonequidistant nodes out of range [-0.5,0.5)");
      }
    }
  }

  /* Check input dimensions. */
  for (t = 0; t < rank; ++t) {
    if (inp_dims[t] <= cutoff) {
      sprintf(errbuf, "%d%s dimension is smaller than cut-off (= %d)",
              t, ordinal_suffix(t), cutoff);
      y_error(errbuf);
    }
    if (inp_dims[t]%2 != 0) {
      sprintf(errbuf, "%d%s dimension is odd (must be even)",
              t, ordinal_suffix(t));
      y_error(errbuf);
    }
  }

  /* Check oversampled dimensions, if given; otherwise, compute them. */
  if (ovr_dims != NULL) {
    if (ovr_dims_rank == 0) {
      /* Duplicate scalar oversampled dimension. */
      for (t = 0; t < rank; ++t) {
        ovr_dims_buf[t] = ovr_dims[0];
      }
      ovr_dims = ovr_dims_buf;
    } else if (ovr_dims_ntot != rank) {
      y_error("bad number of oversampled dimensions in OVR_DIMS keyword");
    }
    for (t = 0; t < rank; ++t) {
      if (ovr_dims[t] <= inp_dims[t]) {
        sprintf(errbuf,
                "%d%s oversampled dimension too small in OVR_DIMS keyword",
                t, ordinal_suffix(t));
        y_error(errbuf);
      }
    }
  } else {
    /* Check oversampling factors, if given; otherwise, use a default
       oversampling factor of 2. */
    if (ovr_fact == NULL) {
      /* Duplicate scalar oversampled factor. */
      for (t = 0; t < rank; ++t) {
        ovr_fact_buf[t] = 2.0;
      }
      ovr_fact = ovr_fact_buf;
    } else {
      if (ovr_fact_rank == 0) {
        for (t = 0; t < rank; ++t) {
          ovr_fact_buf[t] = ovr_fact[0];
        }
        ovr_fact = ovr_fact_buf;
      } else if (ovr_fact_ntot != rank) {
        y_error("bad number of oversampling factors in OVR_FACT keyword");
      }
    }

    /* Compute oversampled dimensions. */
    for (t = 0; t < rank; ++t) {
      long dim;
      if (ovr_fact[t] <= 0.0) {
        dim = -1;
      } else {
        double min_dim = ovr_fact[t]*inp_dims[t];
        dim = best_fft_length((long)min_dim);
        while (dim < min_dim) {
          dim = best_fft_length(dim + 1);
        }
      }
      if (dim <= inp_dims[t]) {
        sprintf(errbuf, "oversampling factor too small for %d%s "
                "dimension in OVR_FACT keyword",
                t, ordinal_suffix(t));
        y_error(errbuf);
      }
      ovr_dims_buf[t] = dim;
    }
    ovr_dims = ovr_dims_buf;
  }

  /* Set flags (defaults come from file nfft.c in NFFT3 sources). */
  fftw_flags = get_fftw_flags(flags);
  nfft_flags = get_nfft_flags(flags);
  if (flags == -1 && rank > 1) {
    /* According to NFFT documentation, this should yield some speedup. */
    nfft_flags |= NFFT_SORT_NODES;
  }

  /* Create object instance. */
  op = (op_t *)ypush_obj(&op_class, sizeof(op_t));


  /* Create NFFT plan. */
  make_nfft_plan(&op->plan, rank, inp_dims, ovr_dims,
                 num_nodes, x, 1.0, cutoff, nfft_flags, fftw_flags);
  op->initialized = TRUE;

  return;

  /* Error management. */
  {
    char *errmsg;
#define THROW(msg) errmsg = msg; goto throw
  too_many_dims:
    sprintf(errbuf, "too many dimensions (max. rank = %d)", MAX_RANK);
    y_error(errbuf);
  bad_coord_type:
    THROW("bad type for coordinates of nonequidistant nodes");
  bad_coord_dims:
    THROW("bad dimensions for coordinates of nonequidistant nodes");
  not_same_lengths:
    THROW("coordinates of nonequidistant nodes have not same lengths");
  expecting_salar_dim:
    THROW("expecting scalar dimension");
  ovr_fact_and_ovr_dims_exclusive:
    THROW("keywords OVR_FACT and OVR_DIMS are exclusive");
  incompatible_dims:
    THROW("number of input dimensions does not rank of coordinates");
#undef THROW
  throw:
    y_error(errmsg);
  }
}

static void make_nfft_plan(nfft_plan *plan, long rank,
                           const long inp_dims[],
                           const long ovr_dims[],
                           long num_nodes, const double *x[], double xscale,
                           double cutoff,
                           unsigned int nfft_flags,
                           unsigned int fftw_flags)
{
  /* Work arrays to convert/swap integers. */
  int _inp_dims[MAX_RANK], _ovr_dims[MAX_RANK];
  int _num_nodes;
  long j, t;
  double *plan_x;

  if (rank > MAX_RANK) {
    y_error("too many dimensions");
  }

  /* Create NFFT plan (taking care of integer overflow for 64-bit -> 32-bit
     conversion and of ordering of dimensions). */
  if ((_num_nodes = (int)num_nodes) != num_nodes) {
    goto integer_overflow;
  }
  for (t = 0; t < rank; ++t) {
    if ((_inp_dims[rank - 1 - t] = (int)inp_dims[t]) != inp_dims[t] ||
        (_ovr_dims[rank - 1 - t] = (int)ovr_dims[t]) != ovr_dims[t]) {
      goto integer_overflow;
    }
  }
  nfft_init_guru(plan, rank, _inp_dims, _num_nodes,
                 _ovr_dims, cutoff, nfft_flags, fftw_flags);

  /* Copy coordinates of nodes.  Taking into account storage order:
   *   x_yorick[j + num_nodes*t] = x_nfft[(rank - 1 - t) + j*rank]
   */
  plan_x = plan->x;
  if (plan_x == NULL) {
      /* FIXME: do more extensive tests. */
    y_error("creation of NFFT plan failed");
  }
  for (t = 0; t < rank; ++t) {
    for (j = 0; j < num_nodes; ++j) {
      plan_x[rank - 1 - t + rank*j] = xscale*x[t][j];
    }
  }

  /* Precompute interpolation weights. */
  if ((plan->nfft_flags & PRE_ONE_PSI) != 0) {
    nfft_precompute_one_psi(plan);
  }
  return;

 integer_overflow:
  y_error("integer overflow (too many nodes or dimensions too large)");
}

/*---------------------------------------------------------------------------*/
/* MIRA-3D OPERATOR */

/* MiRA-3D opertaor is implementd by:
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
      SAFE_FREE(sub->u);
      SAFE_FREE(sub->v);
    }
    p_free(op->sub);
  }

  /* Free other stuff. */
  SAFE_FREE(op->u);
  SAFE_FREE(op->v);
  SAFE_FREE(op->w);
  SAFE_FREE(op->c0);
  SAFE_FREE(op->k0);
  SAFE_FREE(op->i0);
  SAFE_FREE(op->i1);
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
#pragma  omp parallel for schedule(dynamic) default(none) shared(nw,stride,op,dst)  
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
    if(op->complex_meas){
    dims[0] = 1;
    dims[1] = m;
      dst = ypush_z(dims);
    }
    else{ 
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
      if (index == complex_meas_index){
	id = yarg_typeid(--iarg);
        if (get_scalar_int(iarg, &complex_meas) != SUCCESS) {
	  y_error("bad value for COMPLEX_MEAS keyword");
	}
        
      }
      if (index == num_threads_index){
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
  op->complex_meas=complex_meas,
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
      SAFE_FREE(tmp_u);
      SAFE_FREE(tmp_v);
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

/*
 * Local Variables:
 * mode: C
 * tab-width: 8
 * c-basic-offset: 2
 * indent-tabs-mode: nil
 * fill-column: 78
 * coding: utf-8
 * End:
 */
