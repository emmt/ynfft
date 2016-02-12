/*
 * nfft.i --
 *
 * Yorick interface to NFFT (non-uniform fast Fourier transform).
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

if (is_func(plug_in)) plug_in, "yor_nfft";

extern nfft_version;
/* DOCUMENT nfft_version();
     Returns the version of NFFT plug-in as a string.
   SEE ALSO: nfft_new.
 */
local NFFT_PRE_PHI_HUT, NFFT_FG_PSI, NFFT_PRE_LIN_PSI, NFFT_PRE_FG_PSI;
local NFFT_PRE_PSI, NFFT_PRE_FULL_PSI, NFFT_SORT_NODES;
local NFFT_ESTIMATE, NFFT_MEASURE, NFFT_PATIENT, NFFT_EXHAUSTIVE;
extern nfft_new;
/* DOCUMENT op = nfft_new(dims, x);
         or op = nfft_new(n1, x1, n2, x2, ...);

     The function nfft_new() creates a new operator for nonequidistant fast
     Fourier transform (NFFT).  DIMS is the dimension list of the input evenly
     sampled arrays to transform, and X is an M-by-D real array which gives
     the coordinates of the nonequispaced nodes:

       X(j,t) = t-th coordinate of j-th node

     thus D is the number of dimensions and M the number of nonequispaced
     nodes and DIMS is:

       DIMS = [D, N1, N2, ..., ND]

     or

       DIMS = [N1, N2, ..., ND]

     with N1, N2, ..., ND, the lengths of the dimensions.  Alternately, the
     transform may be specified as: N1, X1, N2, X2, ... with:

       N1 = length of 1st dimension;
       X1 = 1st coordinates of the nonequispaced nodes;
       N2 = length of 2nd dimension;
       X2 = 2nd coordinates of the nonequispaced nodes;
       ...

     The coordinates of the nonequispaced nodes must be in the range
     [-1/2,1/2) and all dimensions N1, N2, ..., must be even.

     The new operator can be called as a function to apply the transform.  For
     instance:

       b = op(a)

     where A is a complex array of dimensions DIMS (see above) yields a
     complex vector of length M set with:

       b(j) = sum_k a(k) * exp(-2i*pi*<k,x_j>)

     for j = 1, ..., M, and where k is the multi-index to access the elements
     of A, and the "scalar product" <k,x_j> is:

       <k,x_j> = sum_t (k(t) - (1 + n(t)/2))*x(j,t)

     with x(j,t) the t-th coordinate of j-th node, n(t) the length of t-th
     axis and k(t) = 1, ..., n(t), the index along this axis (using Yorick
     conventions).  The transpose transform is obtained by:

       a = op(b, 1)

     The 1-D transform is:

       b(j) = sum_{k1=1}^{n1} a(k1)*exp(-2i*pi*(k1 - 1 - n1/2)*x(j));

     Operator object can be used as a structure to query its parameters:

       op.rank         = number of dimensions D;
       op.num_nodes    = number of nonequispaced nodes;
       op.nodes        = packed coordinates of nonequispaced nodes (NUM_NODES
                         vector if RANK=1, a NUM_NODES-by-RANK array else);
       op.inp_dims     = dimension list of input array;
       op.ovr_dims     = oversampled dimension list;
       op.ovr_fact     = oversampling factors (for each dimension);
       op.flags        = flags and options for NFFT and FFTW;
       op.nfft_flags   = flags for NFFT (for debug only);
       op.fftw_flags   = flags for FFTW (for debug only);
       op.cutoff       = size of window;
       op.nevals       = number of evaluations;

     The operator is approximated by:

       A ~ B.F.D

     where A is the exact transform, D is a diagonal matrix which perform the
     deconvolution step (division by the Fourier transform of the
     interpolation kernel), F is a discrete Fourier transform (computed by
     FFTW) and B is a sparse interpolation matrix.


   KEYWORDS:
     Keyword OVR_FACT (a scalar or a vector of length D) can be used to specify
     the oversampling factor.  By default, the oversampling factor is at least
     two.  The actual oversampling factor can be larger because dimensions are
     rounded up to powers of 2, 3 and 5.  OP.ovr_fact will gives the actual
     oversampling factor.

     Keyword OVR_DIMS can be set with a vector of integers [L1,L2, ...] or a
     dimension list [D,L1,L2,...] to specify the dimensions of the oversampled
     array.  Keywords OVR_FACT and OVR_DIMS are exclusive.

     Keyword CUTOFF specifies the cut-off of the window function used for the
     interpolation of the discrete Fourier samples. The number of neighbors
     taken into account for interpolating along a direction is 2*CUTOFF + 2.

     Keyword FLAGS can be set with a combination of:

       NFFT_PRE_PHI_HUT  - If this flag is set, the deconvolution step (the
                           multiplication with the diagonal matrix D) uses
                           precomputed values of the Fourier transformed window
                           function.

       NFFT_FG_PSI       - If this flag is set, the convolution step (the
                           multiplication with the sparse interpolation matrix
                           B) uses particular properties of the Gaussian window
                           function to trade multiplications for direct calls
                           to exponential function.

       NFFT_PRE_LIN_PSI  - If this flag is set, the convolution step (the
                           multiplication with the sparse interpolation matrix
                           B) uses linear interpolation from a lookup table of
                           equispaced samples of the window function instead of
                           exact values of the window function.

       NFFT_PRE_FG_PSI   - If this flag is set, the convolution step (the
                           multiplication with the sparse interpolation matrix
                           B) uses particular properties of the Gaussian window
                           function to trade multiplications for direct calls to
                           exponential function (the remaining direct calls are
                           precomputed).

       NFFT_PRE_PSI      - If this flag is set, the convolution step (the
                           multiplication with the sparse interpolation matrix
                           B) uses NUM_NODES*(2*CUTOFF + 2)*RANK precomputed
                           values of the window function.

       NFFT_PRE_FULL_PSI - If this flag is set, the convolution step (the
                           multiplication with the sparse interpolation matrix
                           B) uses NUM_NODES*(2*CUTOFF + 2)^RANK precomputed
                           values of the window function, in addition indices
                           of source and target vectors are stored.

       NFFT_SORT_NODES   - If set, the sampling nodes are internally sorted,
                           which can result in a performance increase.  This
                           has no other side effects for the user.

     plus at most one of the following values to specify the strategy for
     searching for a fast FFT in FFTW library:

       NFFT_ESTIMATE specifies that, instead of actual measurements of
                    different algorithms, a simple heuristic is used to pick a
                    (probably sub-optimal) plan quickly.

       NFFT_MEASURE tells FFTW to find an optimized plan by actually computing
                    several FFTs and measuring their execution time. Depending
                    on your machine, this can take some time (often a few
                    seconds).

       NFFT_PATIENT is like NFFT_MEASURE, but considers a wider range of
                    algorithms and often produces a "more optimal" plan
                    (especially for large transforms), but at the expense of
                    several times longer planning time (especially for large
                    transforms).

       NFFT_EXHAUSTIVE is like NFFT_PATIENT, but considers an even wider range
                    of algorithms, including many that we think are unlikely
                    to be fast, to produce the most optimal plan but with a
                    substantially increased planning time.

     The default flags are:

       (NFFT_PRE_PHI_HUT | NFFT_PRE_PSI | NFFT_ESTIMATE | (RANK > 1 ? NFFT_SORT_NODES : 0))


   SEE ALSO: xfft, mvmult.
 */

/* NFFT flags. */
NFFT_PRE_PHI_HUT    = (1n <<  0n);
NFFT_FG_PSI         = (1n <<  1n);
NFFT_PRE_LIN_PSI    = (1n <<  2n);
NFFT_PRE_FG_PSI     = (1n <<  3n);
NFFT_PRE_PSI        = (1n <<  4n);
NFFT_PRE_FULL_PSI   = (1n <<  5n);
NFFT_SORT_NODES     = (1n << 11n);

/* FFTW options. */
NFFT_ESTIMATE       = (1n << 20n);
NFFT_MEASURE        = (2n << 20n);
NFFT_PATIENT        = (3n << 20n);
NFFT_EXHAUSTIVE     = (4n << 20n);

extern nfft_indgen;
/* DOCUMENT nfft_indgen(len);
         or nfft_indgen(len, stp);
     Generate a vector of LEN elements correspong to NFFT input coordinate index
     frame.  If optional argument STP is missing an array of longs
     [-(LEN/2), 1 - (LEN/2), 2 - (LEN/2), ...] is returned; otherwise, an array of
     doubles [-(LEN/2), 1 - (LEN/2), 2 - (LEN/2), ...]*STP is returned.

   SEE ALSO: indgen.
 */

func nfft_full_matrix(op, mode)
/* DOCUMENT nfft_full_matrix(op);
         or nfft_full_matrix(op, mode);

     This function returns the coefficients of the NFFT operator OP as a plain
     complex array of dimensions OP.NUM_NODES by OP.INP_DIMS. If MODE is 0 or
     omitted, the coefficients are computed directly from the formula.  If
     MODE is 1, the coefficients are computed by applying the operator OP to
     each "vectors" of the canonical basis of the input space.  If MODE is 2,
     the coefficients are computed by applying the adjoint operator OP to each
     "vectors" of the canonical basis of the output space.

     This function is mostly needed for testing purposes.

   SEE ALSO: nfft_new.
 */
{
  PI = 3.1415926535897932384626433832795029;
  if (4*atan(1) != PI) error, "bad value for PI?";

  rank = op.rank;
  nodes = op.nodes;
  num_nodes = op.num_nodes;
  inp_dims = op.inp_dims;
  ovr_dims = op.ovr_dims;
  if (rank >= 1) {
    n1 = inp_dims(2);
    x1 = nfft_indgen(n1);
    u1 = nodes(,1);
  }
  if (rank >= 2) {
    n2 = inp_dims(3);
    x2 = nfft_indgen(n2);
    u2 = nodes(,2);
  }
  if (rank >= 3) {
    n3 = inp_dims(4);
    x3 = nfft_indgen(n3);
    u3 = nodes(,3);
  }
  a = array(complex, num_nodes, inp_dims);
  if (! mode) {
    /* Directly compute coefficients of operator. */
    if (rank == 1) {
      phi = 2*PI*u1*x1(-,);
    } else if (rank == 2) {
      phi = 2*PI*(u1*x1(-,) + u2*x2(-,-,));
    } else if (rank == 3) {
      phi = 2*PI*(u1*x1(-,) + u2*x2(-,-,) + u3*x3(-,-,-,));
    }
    a.re =  cos(phi);
    a.im = -sin(phi);
  } else if (mode == 1) {
    /* Apply the operator to the canonical basis. */
    x = array(complex, inp_dims);
    if (rank == 1) {
      for (i1 = 1; i1 <= n1; ++i1) {
        x(i1) = 1;
        a(, i1) = op(x);
        x(i1) = 0;
      }
    } else if (rank == 2) {
      for (i1 = 1; i1 <= n1; ++i1) {
        for (i2 = 1; i2 <= n2; ++i2) {
          x(i1, i2) = 1;
          a(, i1, i2) = op(x);
          x(i1, i2) = 0;
        }
      }
    } else if (rank == 3) {
      for (i1 = 1; i1 <= n1; ++i1) {
        for (i2 = 1; i2 <= n2; ++i2) {
          for (i3 = 1; i3 <= n3; ++i3) {
            x(i1, i2, i3) = 1;
            a(, i1, i2, i3) = op(x);
            x(i1, i2, i3) = 0;
          }
        }
      }
    }
  } else if (mode == 2) {
    /* Apply the adjoint operator to the canonical basis. */
    y = array(complex, num_nodes);
    for (j = 1; j <= num_nodes; ++j) {
      y(j) = 1;
      a(j,..) = conj(op(y,1));
      y(j) = 0;
    }
  } else {
    error, "invalid method for computing the full matrix";
  }
  return a;
}

extern nfft_mira3d_new;
/* DOCUMENT H = nfft_mira3d_new(u, v, w, pixelsize, nx, ny, wlist,
                                complex_meas=);

   U, V and W are the spatial frequency coordinates and wavelength they must
   all be of the same size and in compatible units.

   WLIST = list of model wavelengths (must be in ascending order).

   COMPLEX_MEAS keyword can be used to specify whether the measurements model
   is an array composed of complex number or pairs of reals. (default:
   COMPLEX_MEAS = 0).

   SEE ALSO: nfft_new.
 */
