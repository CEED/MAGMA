/**
 *
 * @file quarkos-hwloc.c
 *
 *  This file handles the mapping from pthreads calls to windows threads
 *  QUARK is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 *  Note : this file is a copy of plasmaos-hwloc.c for use of Quark alone.
 *
 * @version 2.5.0
 * @author Piotr Luszczek
 * @author Mathieu Faverge
 * @date 2010-11-15
 *
 **/

#include <hwloc.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef QUARK_HWLOC

static hwloc_topology_t quark_topology = NULL; /* Topology object */
static volatile int     quark_nbr = 0;

void quark_topology_init()
{
    pthread_mutex_lock(&mutextopo);
    if (!topo_initialized) {
        /* Allocate and initialize topology object.  */
        hwloc_topology_init(&quark_topology);

        /* Perform the topology detection.  */
        hwloc_topology_load(quark_topology);

        /* Get the number of cores (We don't want to use HyperThreading */
        sys_corenbr = hwloc_get_nbobjs_by_type(quark_topology, HWLOC_OBJ_CORE);

        topo_initialized = 1;
    }
    quark_nbr++;
    pthread_mutex_unlock(&mutextopo);
}

void quark_topology_finalize()
{
    pthread_mutex_lock(&mutextopo);
    quark_nbr--;
    if ((topo_initialized ==1) && (quark_nbr == 0)) {
        /* Destroy tpology */
        hwloc_topology_destroy(quark_topology);

        topo_initialized = 0;
    }
    pthread_mutex_unlock(&mutextopo);
}

/**
 This routine will set affinity for the calling thread that has rank 'rank'.
 Ranks start with 0.

 If there are multiple instances of QUARK then affinity will be wrong: all ranks 0
 will be pinned to core 0.

 Also, affinity is not resotred when QUARK_Finalize() is called.
 */
int quark_setaffinity(int rank) {
    hwloc_obj_t      obj;      /* Hwloc object    */ 
    hwloc_cpuset_t   cpuset;   /* HwLoc cpuset    */

    if (!topo_initialized) {
        /* quark_error("quark_setaffinity", "Topology not initialized"); */
        return -1;
    }

    /* Get last one.  */
    obj = hwloc_get_obj_by_type(quark_topology, HWLOC_OBJ_CORE, rank);
    if (!obj)
        return -1;

    /* Get a copy of its cpuset that we may modify.  */
    /* Get only one logical processor (in case the core is SMT/hyperthreaded).  */
#if !defined(HWLOC_BITMAP_H)
    cpuset = hwloc_cpuset_dup(obj->cpuset);
    hwloc_cpuset_singlify(cpuset);
#else
    cpuset = hwloc_bitmap_dup(obj->cpuset);
    hwloc_bitmap_singlify(cpuset);
#endif

    /* And try to bind ourself there.  */
    if (hwloc_set_cpubind(quark_topology, cpuset, HWLOC_CPUBIND_THREAD)) {
        char *str = NULL;
#if !defined(HWLOC_BITMAP_H)
        hwloc_cpuset_asprintf(&str, obj->cpuset);
#else
        hwloc_bitmap_asprintf(&str, obj->cpuset);
#endif
        printf("Couldn't bind to cpuset %s\n", str);
        free(str);
        return -1;
    }

    /* Get the number at Proc level ( We don't want to use HyperThreading ) */
    rank = obj->children[0]->os_index;

    /* Free our cpuset copy */
#if !defined(HWLOC_BITMAP_H)
    hwloc_cpuset_free(cpuset);
#else
    hwloc_bitmap_free(cpuset);
#endif

    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* QUARK_HAS_COMPLEX */
