/*
 * yor_nfft.c --
 *
 * Implementation of NFFT Yorick interface.
 *
 *-----------------------------------------------------------------------------
 *
 * Copyright (C) 2012, 2015-2016, Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
 * Copyright (C) 2013-2014, Ferréol Soulez <ferreol.soulez@univ-lyon1.fr> and
 *                          Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
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

/*
 * IMPLEMENTATION NOTES
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
 */

#include <string.h>
#include <stdio.h>
#include <yapi.h>
#include <pstdlib.h>
#include <play.h>
#include <ydata.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <nfft3.h>
#include <fftw3.h>

/* Deal with different versions of NFFT3. */
#ifdef NFFT_DEFINE_MALLOC_API
# define MEMBER_NFFT_FLAGS flags
#else
# define NFFT_INT int
# define MEMBER_NFFT_FLAGS nfft_flags
#endif

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

static void make_nfft_plan(nfft_plan *plan, long rank,
                           const long inp_dims[],
                           const long ovr_dims[],
                           long num_nodes, const double *x[], double xscale,
                           double cutoff,
                           unsigned int nfft_flags,
                           unsigned int fftw_flags);

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
static long nthreads_index = -1L;

/* Default value for cutoff number (negative means not yet determined). */
static long default_cutoff = -1;

#define INIT_THREADS 0x1
#define INIT_OTHERS  0x2
#define INIT_BITS    (INIT_THREADS|INIT_OTHERS)
static unsigned int init_bits = INIT_BITS;
static int use_threads = 0;

#define INITIALIZE if ((init_bits & INIT_BITS) == 0) /* do nothing */; \
                   else initialize()

static void initialize(void)
{
  if ((init_bits & INIT_THREADS) != 0) {
    /* Clear the bit before because we have only one chance to do that. */
    init_bits &= ~INIT_THREADS;
#ifdef USE_THREADS
    if (fftw_init_threads() == 0) {
      use_threads = 0;
      y_error("initialization of FFTW threads failed");
    } else {
      use_threads = 1;
    }
#else
    use_threads = 0;
#endif
  }

  if ((init_bits & INIT_OTHERS) != 0) {
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
    SET_INDEX(nthreads);
    SET_INDEX(fftw_flags);
#undef SET_INDEX
    {
      /* Get default size of window. */
      nfft_plan p;
      nfft_init_1d(&p, 100, 1);
      default_cutoff = p.m;
      nfft_finalize(&p);
    }

    /* Clear the initialization bit.  In case of interrupts, must be done
       last. */
    init_bits &= ~INIT_OTHERS;
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
#define GET_NFFT_FLAGS(op)      ((op)->plan.MEMBER_NFFT_FLAGS)
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
  const NFFT_INT *inp_dims;
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
    const NFFT_INT *src = GET_OVR_DIMS(op);
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
    const NFFT_INT *src = GET_INP_DIMS(op);
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

/*---------------------------------------------------------------------------*/
/* BUILT-IN FUNCTIONS */

BUILTIN(version)(int argc)
{
  ypush_q(NULL)[0] = p_strcpy(VERSION);
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
  const double *ovr_fact = NULL;
  long          ovr_fact_ntot = 0;
  int           ovr_fact_rank = -1;
  double        ovr_fact_buf[MAX_RANK];

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
  int           flags = -1;
  int           nthreads = 1;
  unsigned int  nfft_flags, fftw_flags;
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
    } else if (index == nthreads_index) {
      --iarg;
      if (get_scalar_int(iarg, &nthreads) != SUCCESS || nthreads < 1) {
        y_error("invalid value for NTHREADS keyword");
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

  /* Set number of threads for FFTW. */
  if (nthreads > 1) {
#ifdef USE_THREADS
    fftw_plan_with_nthreads(nthreads);
#else
    y_warn("NFFT not compiled with support for multi-threaded FFTW");
#endif
  }

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
                           long num_nodes,
                           const double *x[],
                           double xscale,
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
  if ((plan->MEMBER_NFFT_FLAGS & PRE_ONE_PSI) != 0) {
    nfft_precompute_one_psi(plan);
  }
  return;

 integer_overflow:
  y_error("integer overflow (too many nodes or dimensions too large)");
}

#include "m3d_nfft.c"
