/*
 * nfft-tests.i --
 *
 * Suite of tests for NFFT Yorick plugin.
 *
 *-----------------------------------------------------------------------------
 *
 * Copyright (C) 2012, Éric Thiébaut <eric.thiebaut@univ-lyon1.fr>
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

func _nfft_test_init
{
  if (! is_func(nfft_new)) {
    here = current_include();
    index = strfind("/", here, back=1n)(2);
    if (index < 0) error, "bad path";
    dir = strpart(here, 1:index);
    prev = plug_dir(dir);
    include, dir+"nfft.i", 1;
    plug_dir, prev;
  }
}

func _nfft_test_info(name, a, b, threshold)
{
  r = abs(a - b);
  i = where(r);
  e = array(1.0, dimsof(r));
  if (is_array(i)) {
    e(i) = 1.0/max(abs(a(i)), abs(b(i)));
  }
  e *= r;
  if (is_void(threshold)) threshold = 1e-6;
  failure = (max(e) > threshold);
  write, format = "%s: %s\n", name, (failure ? "FAILURE" : "SUCCESS");
  write, format = "  - absolute error: max. = %.1e, RMS = %.1e\n", max(r), sqrt(avg(r*r));
  //write, format = "  - relative error: max. = %.1e, RMS = %.1e\n", max(e), sqrt(avg(e*e));
}
errs2caller, _nfft_test_info;

func _nfft_test_newline
{
  write, format="%s", "\n";
}

func nfft_test_clone(f, cutoff=, flags=, ovr_dims=, ovr_fact=)
{
  if (is_void(cutoff)) cutoff = f.cutoff;
  if (is_void(flags)) flags = f.flags;
  if (is_void(ovr_fact)) {
    if (is_void(ovr_dims)) ovr_dims = f.ovr_dims(2:0);
    return nfft_new(f.inp_dims, f.nodes, cutoff=cutoff, flags=flags, ovr_dims=ovr_dims);
  } else {
    if (! is_void(ovr_dims)) error, "keywords OVR_DIMS and OVR_FACT are exclusive";
    return nfft_new(f.inp_dims, f.nodes, cutoff=cutoff, flags=flags, ovr_fact=ovr_fact);
  }
}

func nfft_test
{
  PI = 3.1415926535897932384626433832795029;
  if (4*atan(1) != PI) error, "bad value for PI?";
  METER = 1.0;
  MICRON = 1E-6*METER;
  ARCSEC = PI/(180*3600);
  PIXSCALE = 1e-3*ARCSEC;
  BMAX = 180.0*METER;
  LAMBDA = 1.2*MICRON;

  FLAGS1 = (NFFT_PRE_PHI_HUT | NFFT_PRE_PSI | NFFT_ESTIMATE);
  FLAGS2 = (FLAGS1 | NFFT_SORT_NODES);

  num_nodes = 42;
  n1 = 20;   // input size along 1st axis
  n2 = 22;   // input size along 2nd axis
  n3 = 12;   // input size along 3rd axis

  u1 = (random(num_nodes) - 0.5)*(BMAX/LAMBDA);
  u2 = (random(num_nodes) - 0.5)*(BMAX/LAMBDA);
  u3 = (random(num_nodes) - 0.5)*(BMAX/LAMBDA);

  // 1-D test
  f = nfft_new(n1, u1*PIXSCALE);
  a0 = nfft_full_matrix(f, 0);
  a1 = nfft_full_matrix(f, 1);
  a2 = nfft_full_matrix(f, 2);
  _nfft_test_info, "1-D transform (A0 vs. A1)", a0, a1;
  _nfft_test_info, "1-D transform (A1 vs. A2)", a1, a2;
  _nfft_test_newline;

  if (anyof(f.nodes != u1*PIXSCALE)) error, "nodes have changed";
  f1 = nfft_test_clone(f, flags=FLAGS1);
  if (f1.flags != FLAGS1) error, "flags have changed";
  f2 = nfft_test_clone(f, flags=FLAGS2);
  if (f2.flags != FLAGS2) error, "flags have changed";
  if (anyof(f2.nodes != f.nodes)) {
    write, format="WARNING - %s\n", "nodes have changed with NFFT_SORT_NODES";
  }
  a11 = nfft_full_matrix(f1, 1);
  if (anyof(a11 != a1)) error, "matrix coefficients have changed";
  a12 = nfft_full_matrix(f1, 2);
  if (anyof(a12 != a2)) error, "matrix coefficients have changed";
  a21 = nfft_full_matrix(f2, 1);
  if (anyof(a21 != a1)) error, "matrix coefficients have changed";
  a22 = nfft_full_matrix(f2, 2);
  if (anyof(a22 != a2)) error, "matrix coefficients have changed";

  // 2-D test
  f = nfft_new(n1, u1*PIXSCALE, n2, u2*PIXSCALE);
  a0 = nfft_full_matrix(f, 0);
  a1 = nfft_full_matrix(f, 1);
  a2 = nfft_full_matrix(f, 2);
  _nfft_test_info, "2-D transform (A0 vs. A1)", a0, a1;
  _nfft_test_info, "2-D transform (A1 vs. A2)", a1, a2;
  _nfft_test_newline;

  if (anyof(f.nodes != [u1,u2]*PIXSCALE)) error, "nodes have changed";
  f1 = nfft_test_clone(f, flags=FLAGS1);
  if (f1.flags != FLAGS1) error, "flags have changed";
  f2 = nfft_test_clone(f, flags=FLAGS2);
  if (f2.flags != FLAGS2) error, "flags have changed";
  if (anyof(f2.nodes != f.nodes)) {
    write, format="WARNING - %s\n", "nodes have changed with NFFT_SORT_NODES";
  }
  a11 = nfft_full_matrix(f1, 1);
  if (anyof(a11 != a1)) error, "matrix coefficients have changed";
  a12 = nfft_full_matrix(f1, 2);
  if (anyof(a12 != a2)) error, "matrix coefficients have changed";
  a21 = nfft_full_matrix(f2, 1);
  if (anyof(a21 != a1)) error, "matrix coefficients have changed";
  a22 = nfft_full_matrix(f2, 2);
  if (anyof(a22 != a2)) error, "matrix coefficients have changed";

  // 3-D test
  f = nfft_new(n1, u1*PIXSCALE, n2, u2*PIXSCALE, n3, u3*PIXSCALE);
  a0 = nfft_full_matrix(f, 0);
  a1 = nfft_full_matrix(f, 1);
  a2 = nfft_full_matrix(f, 2);
  _nfft_test_info, "3-D transform (A0 vs. A1)", a0, a1;
  _nfft_test_info, "3-D transform (A1 vs. A2)", a1, a2;
  _nfft_test_newline;

  if (anyof(f.nodes != [u1,u2,u3]*PIXSCALE)) error, "nodes have changed";
  f1 = nfft_test_clone(f, flags=FLAGS1);
  if (f1.flags != FLAGS1) error, "flags have changed";
  f2 = nfft_test_clone(f, flags=FLAGS2);
  if (f2.flags != FLAGS2) error, "flags have changed";
  if (anyof(f2.nodes != f.nodes)) {
    write, format="WARNING - %s\n", "nodes have changed with NFFT_SORT_NODES";
  }
  a11 = nfft_full_matrix(f1, 1);
  if (anyof(a11 != a1)) error, "matrix coefficients have changed";
  a12 = nfft_full_matrix(f1, 2);
  if (anyof(a12 != a2)) error, "matrix coefficients have changed";
  a21 = nfft_full_matrix(f2, 1);
  if (anyof(a21 != a1)) error, "matrix coefficients have changed";
  a22 = nfft_full_matrix(f2, 2);
  if (anyof(a22 != a2)) error, "matrix coefficients have changed";
}

_nfft_test_init;
if (batch()) {
  nfft_test;
}
