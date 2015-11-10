#!/usr/bin/perl -pi
#
# Does basic conversion of MAGMA kernels from v1 to v2.
# 1) Kernel launches get stream from queue.
# 2) Old references to magma_stream use magmablasGetQueue() to get current queue.
#
# This does not attempt to write _q versions of functions,
# e.g., magmablas_ztrsm_q and magmablas_ztrsm.
#
# @author Mark Gates

s/<<<(\S)/<<< $1/;
s/queue *>>>/queue->cuda_stream() >>>/;
s/magma_stream/magmablasGetQueue()/;
