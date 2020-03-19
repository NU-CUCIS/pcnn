#
# Copyright (C) 2020, Northwestern University
# See COPYRIGHT notice in top-level directory.
#

AC_INIT([pcnn], [1.0],[],[pcnn],[])
AC_CONFIG_SRCDIR([model.h])
AC_CONFIG_AUX_DIR([./scripts])
AM_INIT_AUTOMAKE([foreign])
AC_PROG_CC([mpicxx])
AC_PROG_CXX([mpicxx])

# OpenCV option
AC_ARG_WITH([opencv],
   [AS_HELP_STRING([--with-opencv=/path/to/hdf5], [Specify OpenCV installation path(s):
    --with-opencv=INC,LIB for include directory and library directory separated by a comma
    --with-opencv=DIR for directory containing include/ and lib/ subdirectories
   ])], [
   case $withval in
     *,*)
        opencv_inc="`echo $withval |cut -f1 -d,`"
        opencv_lib="`echo $withval |cut -f2 -d, -s`"
        ;;
     *)
        if test -n "$withval"; then
          opencv_inc="$withval/include"
          opencv_lib="$withval/lib64"
        fi
        ;;
   esac
   if test "x$opencv_inc" != x ; then
      if test "x$CPPFLAGS" = x ; then
         CPPFLAGS="-I$opencv_inc"
      elif ! echo "${CPPFLAGS}" | ${EGREP} -q -w -- "-I$opencv_inc" ; then
         # append only if not already appear in CPPFLAGS
         CPPFLAGS="$CPPFLAGS -I$opencv_inc"
      fi
   fi
   if test "x$opencv_lib" != x ; then
      if test "x$LDFLAGS" = x ; then
         LDFLAGS="-L$opencv_lib"
      elif ! echo "${LDFLAGS}" | ${EGREP} -q -w -- "-L$opencv_lib" ; then
         # append only if not already appear in LDFLAGS
         LDFLAGS="$LDFLAGS -L$opencv_lib"
      fi
   fi
])

AC_LANG_PUSH(C++)
AC_CHECK_HEADER([opencv2/opencv.hpp], [have_opencv=yes], [have_opencv=no])
if test "x$have_opencv" = xno ; then
      AC_MSG_ERROR([
      -----------------------------------------------------------------------
      Missing OpenCV-header file 'opencv2/opencv.hpp' required to build PCNN.
      Use configure command-line option --with-opencv=/path/to/opencv_install_dir
      to specify the location of OpenCV installation. Abort.
      -----------------------------------------------------------------------])
fi

AC_MSG_CHECKING([whether OpenCV version is 3.x])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[
#include "opencv2/opencv.hpp"
#if (CV_MAJOR_VERSION < 3)
#error OpenCV version is 2.x
#endif
]])], [opencv_3x=yes], [opencv_3x=no])
AC_MSG_RESULT([$opencv_3x])
AM_CONDITIONAL([OPENCV3X], [test "x$opencv_3x" = "xyes"])

AC_LANG_POP(C++)

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
