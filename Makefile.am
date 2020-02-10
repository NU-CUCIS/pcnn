#
# Copyright (C) 2020, Northwestern University
# See COPYRIGHT notice in top-level directory.
#

AM_CPPFLAGS = -I ${MKLROOT}/include -DUSE_MKL -D_GNU_SOURCE -O2
AM_LDFLAGS = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl \
			 -lopencv_core -lopencv_imgproc -lopencv_highgui
if OPENCV3X
AM_LDFLAGS += -lopencv_imgcodecs
endif

noinst_PROGRAMS = pcnn
pcnn_SOURCES = main.c model.c arch.c record.c \
			   conv.c full.c pool.c frame.c comm.c \
               relu.c loss.c batch_norm.c residual.c \
               upsample.c transform.c \
               ffbp_allreduce.c ffbp_multistep.c \
               util.cpp feeder.cpp

EXTRA_DIST = model.h arch.h record.h conv.h full.h pool.h frame.h comm.h \
             relu.h loss.h batch_norm.h residual.h upsample.h transform.h \
             ffbp_allreduce.h ffbp_multistep.h \
             util.h feeder.h \
             COPYRIGHT \
             README.md \
             AUTHORS.md
