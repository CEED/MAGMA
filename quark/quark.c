/* **************************************************************************** */
/**
 * @file quark.c
 *
 * QUARK (QUeuing And Runtime for Kernels) provides a runtime
 * enviornment for the dynamic execution of precedence-constrained
 * tasks.
 *
 * QUARK is a software package provided by Univ. of Tennessee,
 * Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 2.5.0
 * @author Asim YarKhan
 * @date 2010-11-15
 *
 */

/* Define a group for Doxygen documentation */
/**
 * @defgroup QUARK QUARK: QUeuing And Runtime for Kernels
 *
 * These functions are available from the QUARK library for the
 * scheduling of kernel routines.
 */

/* Define a group for Doxygen documentation */
/**
 * @defgroup QUARK_Unsupported QUARK: Unsupported functions
 *
 * These functions are used by internal QUARK and PLASMA developers to
 * obtain very specific behavior, but are unsupported and may have
 * unexpected results.
 */

/**
 * @defgroup QUARK_Depreciated QUARK: Depreciated Functions
 *
 * These functions have been depreciated and will be removed in a
 * future release.
 */

/* **************************************************************************** */
/*
  Summary of environment flags:

  Change the window size (default should be checked in the code)
  export QUARK_UNROLL_TASKS_PER_THREAD=num

  Enable WAR avoidance (false dependency handling) (default=0 off)
  export QUARK_WAR_DEPENDENCIES_ENABLE=1

  Enable DAG generation (default=0 off)
  export QUARK_DOT_DAG_ENABLE=1
*/
/* **************************************************************************** */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

#ifndef inline
#define inline __inline
#endif

#if defined( _WIN32 ) || defined( _WIN64 )
#  define fopen(ppfile, name, mode) fopen_s(ppfile, name, mode)
#  define strdup _strdup
#  include "quarkwinthread.h"
#else
#  define fopen(ppfile, name, mode) *ppfile = fopen(name, mode)
#  include <pthread.h>
#endif

#ifdef TRUE
#undef TRUE
#endif

#ifdef FALSE
#undef FALSE
#endif

#include "icl_list.h"
#include "icl_hash.h"
#include "bsd_queue.h"
#include "bsd_tree.h"
#include "quark.h"
#include "quark_unpack_args.h"
#include "quark_trace.h"

#ifdef DBGQUARK
#include <time.h>
#include <sys/time.h>
#endif /* DBGQUARK */

#ifndef ULLONG_MAX
# define ULLONG_MAX 18446744073709551615ULL
#endif

typedef enum { ALLOCATED_ONLY, NOTREADY, QUEUED, RUNNING, DONE, CANCELLED } task_status;
typedef enum { FALSE, TRUE } bool;
typedef enum { WORKER_SLEEPING, WORKER_NOT_SLEEPING } worker_status;

struct quark_s {
    pthread_mutex_t quark_mutex;
    int low_water_mark;
    int high_water_mark;
    int num_threads;              /* number of threads */
    struct worker_s **worker;     /* array of workers [num_threads] */
    int *coresbind;               /* array of indices where to bind workers [num_threads] */
    /* volatile  */int list_robin;      /* round-robin list insertion index */
    volatile bool start;          /* start flag */
    volatile bool all_tasks_queued; /* flag */
    volatile long long num_tasks; /* number of tasks queued */
    icl_hash_t *task_set;
    pthread_mutex_t task_set_mutex;
    icl_hash_t *address_set;      /* hash table of addresses */
    pthread_mutex_t address_set_mutex;    /* hash table access mutex */
    pthread_attr_t thread_attr;   /* threads' attributes */
    volatile int num_queued_tasks;
    pthread_mutex_t num_queued_tasks_mutex;
    pthread_cond_t num_queued_tasks_cond;
    int war_dependencies_enable;
    int dot_dag_enable;
    int dot_dag_was_setup;
    int queue_before_computing;
#define tasklevel_width_max_level 5000
    int tasklevel_width[tasklevel_width_max_level];
    pthread_mutex_t dot_dag_mutex;
    pthread_mutex_t completed_tasks_mutex;
    struct completed_tasks_head_s *completed_tasks;
    volatile int completed_tasks_size;
};

struct Quark_sequence_s {
    volatile int status;
    pthread_mutex_t sequence_mutex;
    struct ll_list_head_s *tasks_in_sequence;
};

typedef struct worker_s {
    pthread_mutex_t worker_mutex;
    pthread_t thread_id;
    int rank;
    struct task_priority_tree_head_s *ready_list;
    volatile int ready_list_size;
    Quark_Task *current_task_ptr;
    Quark *quark_ptr;
    volatile int finalize;       /* termination flag */
    volatile int executing_task;
    int set_to_manual_scheduling;
    pthread_cond_t worker_must_awake_cond;
    int status;
} Worker;

typedef struct quark_task_s {
    pthread_mutex_t task_mutex;
    void (*function) (Quark *);    /* task function pointer */
    volatile task_status status; /* Status of task; NOTREADY, READY; QUEUED; DONE */
    volatile int num_dependencies_remaining; /* number of dependencies remaining to be fulfilled */
    icl_list_t *args_list;        /* list of arguments (copies of scalar values and pointers) */
    icl_list_t *dependency_list;  /* list of dependencies */
    icl_list_t *scratch_list;        /* List of scratch space information and their sizes */
    volatile struct dependency_s *locality_preserving_dep; /* Try to run task on core that preserves the locality of this dependency */
    unsigned long long taskid; /* An identifier, used only for generating DAGs */
    unsigned long long tasklevel; /* An identifier, used only for generating DAGs */
    int lock_to_thread;
    unsigned char *lock_to_thread_mask;
    char *task_label;            /* Label for this task, used in dot_dag generation */
    char *task_color;            /* Color for this task, used in dot_dag generation */
    int priority;                    /* Is this a high priority task */
    Quark_Sequence *sequence;
    struct ll_list_node_s *ptr_to_task_in_sequence; /* convenience pointer to this task in the sequence */
    int task_thread_count;                /* Num of threads required by task */
    int task_thread_count_outstanding; /* Num of threads required by task */
    int thread_set_to_manual_scheduling; /* enable or disable work stealing in the thread that runs this task */
    volatile int threadid;      /* Index of the thread calling the function GetRankInTask in parallel tasks */
    int executed_on_threadid;   /* Track which thread executes this task */
} Task;

typedef struct dependency_s {
    struct quark_task_s *task; /* pointer to parent task containing this dependency */
    void *address;              /* address of data */
    int size;                   /* Size of dependency data */
    quark_direction_t direction; /* direction of this dependency, INPUT, INOUT, OUTPUT */
    bool locality; /* Priority of this dependency; more like data locality */
    bool accumulator; /* Tasks depending on this may be reordered, they accumulate results */
    int data_region; /* Different regions may be specified for dependencies; uses bitmask of 8 bits */
    bool gatherv; /* Tasks depending on this may be run in parallel, assured by the programmer */
    struct address_set_node_s *address_set_node_ptr; /* convenience pointer to address_set_node */
    icl_list_t *address_set_waiting_deps_node_ptr; /* convenience pointer to address_set_node waiting_deps node */
    icl_list_t *task_args_list_node_ptr; /* convenience ptr to the task->args_list [node] to use for WAR address updates */
    icl_list_t *task_dependency_list_node_ptr; /* convenience ptr to the task->dependency_list [node] */
    /* volatile */ bool ready;        /* Data dependency is ready */
} Dependency;

typedef struct scratch_s {
    void *ptr;                  /* address of scratch space */
    int size;                   /* Size of scratch data */
    icl_list_t *task_args_list_node_ptr; /* convenience ptr to the task->args_list [node] */
} Scratch;

typedef struct address_set_node_s {
    void *address; /* copy of key to the address_set - pointer to the data */
    int size;            /* data object size */
    /* volatile */ int last_thread; /* last thread to use this data - for scheduling/locality */
    icl_list_t *waiting_deps;    /* list of dependencies waiting for this data */
    /* volatile */ bool delete_data_at_address_when_node_is_deleted; /* used when data is copied in order to handle false dependencies  */
    unsigned long long last_writer_taskid; /* used for generating DOT DAGs */
    unsigned long long last_writer_tasklevel; /* used for tracking critical depth */
    unsigned long long last_reader_or_writer_taskid; /* used for generating DOT DAGs */
    unsigned long long last_reader_or_writer_tasklevel; /* used for tracking critical depth */
    pthread_mutex_t asn_mutex;
} Address_Set_Node;

/* Data structure for a list containing long long int values.  Used to
 * track task ids in sequences of tasks, so that the tasks in a
 * sequence can be controlled */
typedef struct ll_list_node_s {
    long long int val;
    LIST_ENTRY( ll_list_node_s ) ll_entries;
} ll_list_node_t;
LIST_HEAD(ll_list_head_s, ll_list_node_s);
typedef struct ll_list_head_s ll_list_head_t;

typedef struct completed_tasks_node_s {
    Task *task;
    int workerid;
    TAILQ_ENTRY( completed_tasks_node_s ) ctn_entries;
} completed_tasks_node_t;
TAILQ_HEAD( completed_tasks_head_s, completed_tasks_node_s );
typedef struct completed_tasks_head_s completed_tasks_head_t;

/* Tree (red-black) structure for keeping a priority list of
 * executable tasks */
typedef struct task_priority_tree_node_s {
    int priority;
    Task *task;
    RB_ENTRY( task_priority_tree_node_s ) n_entry;
} task_priority_tree_node_t;
RB_HEAD( task_priority_tree_head_s, task_priority_tree_node_s );
typedef struct task_priority_tree_head_s task_priority_tree_head_t;
static int compare_task_priority_tree_nodes( task_priority_tree_node_t *n1, task_priority_tree_node_t *n2 )
{
    int diff = n2->priority - n1->priority;
    return diff;
}
/* Generate red-black tree functions */
RB_PROTOTYPE_STATIC( task_priority_tree_head_s, task_priority_tree_node_s, n_entry, compare_task_priority_tree_nodes )
RB_GENERATE_STATIC( task_priority_tree_head_s, task_priority_tree_node_s, n_entry, compare_task_priority_tree_nodes )


/* **************************************************************************** */
/**
 * Local function prototypes, declared static so they are not
 * available outside the scope of this file.
 */
static Task *quark_task_new();
static void *quark_task_delete( Quark *quark, Task *task);
static Worker *quark_worker_new(Quark *quark, int rank);
static void quark_worker_delete(Worker *worker);
static inline int quark_worker_find_next_assignable( Quark *quark );
static void quark_insert_task_dependencies(Quark * quark, Task * task);
static void quark_check_and_queue_ready_task( Quark *quark, Task *task, int worker_rank );
static void quark_work_set_affinity_and_call_main_loop(Worker *worker);
static long long quark_work_main_loop(Worker *worker);
static Scratch *quark_scratch_new( void *arg_ptr, int arg_size, icl_list_t *task_args_list_node_ptr);
static void quark_scratch_allocate( Task *task );
static void quark_scratch_deallocate( Task *task );
static void quark_worker_remove_completed_task_enqueue_for_later_processing(Quark *quark, Task *task, int worker_rank);
static void quark_remove_completed_task_and_check_for_ready(Quark *quark, Task *task, int worker_rank);
static void quark_process_completed_tasks(Quark *quark);
static void quark_address_set_node_free( void* data );
static inline void quark_fatal_error(const char *func_name, char* msg_text);
static void quark_address_set_node_wait(Quark *quark, Address_Set_Node *address_set_node);
static Task *quark_set_task_flags_in_task_structure( Quark *quark, Task *task, Quark_Task_Flags *task_flags );
static void quark_avoid_war_dependencies( Quark *quark, Address_Set_Node *asn_old, Task *parent_task );

/* **************************************************************************** */
/**
 * Mutex wrappers for tracing/timing purposes.  Makes it easier to
 * profile the costs of these pthreads routines.
 */
inline static int pthread_mutex_lock_address_set(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_address_set", strerror(rv)); } return rv; }
/* inline static int pthread_mutex_trylock_address_set(pthread_mutex_t *mtx) { int rv;  rv=pthread_mutex_trylock( mtx ); return rv; } */
inline static int pthread_mutex_unlock_address_set(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_unlock( mtx ))!=0) { quark_fatal_error("pthread_mutex_unlock_address_set", strerror(rv)); } return rv; }

inline static int pthread_mutex_lock_ready_list(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_ready_list", strerror(rv)); } return rv; }
inline static int pthread_mutex_trylock_ready_list(pthread_mutex_t *mtx) { int rv;  rv=pthread_mutex_trylock( mtx ); return rv; }
inline static int pthread_mutex_unlock_ready_list(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_unlock( mtx ))!=0) { quark_fatal_error("pthread_mutex_unlock_ready_list", strerror(rv)); } return rv; }

inline static int pthread_mutex_lock_task(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_task", strerror(rv)); } return rv; }
inline static int pthread_mutex_unlock_task(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_unlock( mtx ))!=0) { quark_fatal_error("pthread_mutex_unlock_task", strerror(rv)); } return rv; }

inline static int pthread_mutex_lock_atomic_add(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_atomic_add", strerror(rv)); } return rv; }
inline static int pthread_mutex_lock_atomic_set(pthread_mutex_t *mtx)  { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_atomic_set", strerror(rv)); } return rv; }
/* inline static int pthread_mutex_lock_atomic_get(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_atomic_get", strerror(rv)); } return rv; } */
inline static int pthread_mutex_unlock_atomic(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_unlock( mtx ))!=0) { quark_fatal_error("pthread_mutex_unlock_atomic", strerror(rv)); } return rv; }

inline static int pthread_mutex_lock_wrap(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_wrap", strerror(rv)); } return rv; }
inline static int pthread_mutex_unlock_wrap(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_unlock( mtx ))!=0) { quark_fatal_error("pthread_mutex_unlock_wrap", strerror(rv)); } return rv; }

inline static int pthread_mutex_lock_completed_tasks(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_lock( mtx ))!=0) { quark_fatal_error("pthread_mutex_lock_completed_tasks", strerror(rv)); } return rv; }
inline static int pthread_mutex_trylock_completed_tasks(pthread_mutex_t *mtx) { int rv;  rv=pthread_mutex_trylock( mtx ); return rv; }
inline static int pthread_mutex_unlock_completed_tasks(pthread_mutex_t *mtx) { int rv;  if ((rv=pthread_mutex_unlock( mtx ))!=0) { quark_fatal_error("pthread_mutex_unlock_completed_tasks", strerror(rv)); } return rv; }
/* inline static int pthread_cond_wait_ready_list( pthread_cond_t *cond, pthread_mutex_t *mtx ) { int rv; if ((rv=pthread_cond_wait( cond, mtx))!=0) { quark_fatal_error("pthread_cond_wait_ready_list", strerror(rv)); } return rv; } */
inline static int pthread_cond_wait_wrap( pthread_cond_t *cond, pthread_mutex_t *mtx ) { int rv; if ((rv=pthread_cond_wait( cond, mtx))!=0) { quark_fatal_error("pthread_cond_wait_wrap", strerror(rv)); } return rv; }


/* **************************************************************************** */

/* If dags are to be generated, setup file name and pointer and
 * various macros.  This assumes that the fprintf function is thread
 * safe.  */
static char *quark_task_default_label = " ";
static char *quark_task_default_color = "white";
#define DEPCOLOR "black"
#define DEPCOLOR_R_FIRST "black"
#define DEPCOLOR_W_FIRST "black"
#define DEPCOLOR_RAR "black"
#define DEPCOLOR_WAW "black"
#define DEPCOLOR_RAW "black"
#define DEPCOLOR_WAR "red"
#define DEPCOLOR_GATHERV "green"
#define DOT_DAG_FILENAME "dot_dag_file.dot"
FILE *dot_dag_file = NULL;
#define dot_dag_print_edge( quark, parentid, parent_level, childid, child_level, color) \
    if ( quark->dot_dag_enable ) {                                      \
        pthread_mutex_lock_wrap( &quark->dot_dag_mutex );               \
        if ( parentid>0 ) fprintf(dot_dag_file, "t%llu->t%llu [color=\"%s\"];\n", (parentid), (childid), (color)); \
        fflush(dot_dag_file);                                           \
        child_level = (parent_level+1 <= child_level ? child_level : parent_level+1 ); \
        pthread_mutex_unlock_wrap( &quark->dot_dag_mutex );             \
    }

/* **************************************************************************** */
/**
 * Define a macro to add to value.  Setup to shorten some code by
 * locking/unlocking a mutex around the operation.
 */
#define quark_atomic_add( pval, addvalue, pmutex ) {                       \
        pthread_mutex_lock_atomic_add(pmutex); pval += addvalue; pthread_mutex_unlock_atomic(pmutex); \
        /* pval += addvalue; */                                          \
    }


/* **************************************************************************** */
/**
 * Define a macro to set a value.  Setup to shorten some code by
 * locking/unlocking a mutex around the operation.
 */
#define quark_atomic_set( pval, setvalue, pmutex ) {                       \
        pthread_mutex_lock_atomic_set(pmutex); pval = setvalue; pthread_mutex_unlock_atomic(pmutex); \
        /* pval = setvalue; */                                          \
    }

/* **************************************************************************** */
/**
 * Define a macro get a value.  Setup to shorten some code by
 * locking/unlocking a mutex around the operation.  Can disable the
 * mutex for performance, since the exact current value of the
 * variable is not needed.  The variables used here are declared
 * volatile, and can lag behind the real value without a loss of
 * accuracy.
 */
#define quark_atomic_get( retval, pval, pmutex ) {                      \
        /* pthread_mutex_lock_atomic_get(pmutex); retval = pval; pthread_mutex_unlock_atomic(pmutex);  */\
        retval = pval; \
    }


/***************************************************************************//**
 *
 *  Unrecoverable errors.
 * @param[in] func_name
 *          Function location where warning occurred
 * @param[in] msg_text
 *          Warning message to display.
 *
 ******************************************************************************/
static void quark_fatal_error(const char *func_name, char* msg_text)
{
    fprintf(stderr, "QUARK_FATAL_ERROR: %s(): %s\n", func_name, msg_text);
    abort();
    exit(0);
}

/***************************************************************************//**
 *
 * Warning messages
 * @param[in] func_name
 *          Function location where warning occurred
 * @param[in] msg_text
 *          Warning message to display.
 *
 ******************************************************************************/
void quark_warning(const char *func_name, char* msg_text)
{
    fprintf(stderr, "QUARK_WARNING: %s(): %s\n", func_name, msg_text);
}

/* **************************************************************************** */
/**
 * Allocate memory, failing
 */
static inline void *quark_malloc(size_t size)
{
    void *mem = malloc(size);
    if ( mem == NULL ) quark_fatal_error( "malloc", "memory allocation failed" );
    return mem;
}

/* **************************************************************************** */
/**
 * Initialize the task data structure
 */
static Task *quark_task_new()
{
    static unsigned long long taskid = 1;
    Task *task = (Task *)quark_malloc(sizeof(Task));
    task->function = NULL;
    task->num_dependencies_remaining = 0;
    task->args_list = icl_list_new();
    if ( task->args_list == NULL) quark_fatal_error( "quark_task_new", "Allocating arg list" );
    task->dependency_list = icl_list_new();
    if ( task->dependency_list == NULL) quark_fatal_error( "quark_task_new", "Allocating dependency list" );
    task->locality_preserving_dep = NULL;
    task->scratch_list = icl_list_new();
    if ( task->scratch_list == NULL) quark_fatal_error( "quark_task_new", "Allocating scratch list" );
    if ( taskid >= ULLONG_MAX) quark_fatal_error( "quark_task_new", "Task id > ULLONG_MAX, too many tasks" );
    task->taskid = taskid++;
    task->tasklevel = 0;
    pthread_mutex_init( &task->task_mutex, NULL );
    task->ptr_to_task_in_sequence = NULL;
    task->sequence = NULL;
    task->priority = QUARK_TASK_MIN_PRIORITY;
    task->task_label = quark_task_default_label;
    task->task_color = quark_task_default_color;
    task->lock_to_thread = -1;
    task->lock_to_thread_mask = NULL;
    task->task_thread_count = 1;
    task->thread_set_to_manual_scheduling = -1;
    task->threadid = 0;
    task->status = ALLOCATED_ONLY;
    task->executed_on_threadid = -1;
    return task;
}


/* **************************************************************************** */
/**
 * Remove the task from any quark data structures.  Note, this may
 * also occur before the task is added to any global data structures,
 * if the sequence is deleted.
 */
static void *quark_task_delete(Quark *quark, Task *task)
{
    /* task is not just allocated, it has been inserted and may have other references to it */
    if ( task->status!=ALLOCATED_ONLY ) {
        quark_trace_deltask();
        pthread_mutex_lock_wrap( &quark->task_set_mutex );
        icl_hash_delete( quark->task_set, &task->taskid, NULL, NULL );
        quark->num_tasks--;
        pthread_mutex_lock_task( &task->task_mutex );
        pthread_mutex_unlock_wrap( &quark->task_set_mutex );
    }
    if ( task->task_color!=NULL && task->task_color!=quark_task_default_color ) free(task->task_color); 
    if ( task->task_label!=NULL && task->task_label!=quark_task_default_label ) free(task->task_label); 
    if ( task->lock_to_thread_mask!=NULL ) free(task->lock_to_thread_mask); 
    icl_list_destroy(task->args_list, free);
    icl_list_destroy(task->dependency_list, free);
    icl_list_destroy(task->scratch_list, free);
    if ( task->status!=ALLOCATED_ONLY ) {
        if ( task->ptr_to_task_in_sequence != NULL ) {
            pthread_mutex_lock_wrap( &task->sequence->sequence_mutex );
            LIST_REMOVE( task->ptr_to_task_in_sequence, ll_entries );
            pthread_mutex_unlock_wrap( &task->sequence->sequence_mutex );
            free( task->ptr_to_task_in_sequence );
        }
        pthread_mutex_unlock_task( &task->task_mutex );
    }
    pthread_mutex_destroy( &task->task_mutex );
    free( task );
    task = NULL;
    return task;
}

/* **************************************************************************** */
/**
 * Return the rank of a thread.
 *
 * @param[in] quark
 *         The scheduler's main data structure.
 * @return
 *          The rank of the calling thread
 * @ingroup QUARK
 */
int QUARK_Thread_Rank(Quark *quark)
{
    pthread_t self_id = pthread_self();
    int  i;
    for (i=0; i<quark->num_threads; i++)
        if (pthread_equal(quark->worker[i]->thread_id, self_id))
            return i;
    return -1;
}

/* **************************************************************************** */
/**
 * Return a pointer to the argument list being processed by the
 * current task and worker.
 *
 * @param[in] quark
 *         The scheduler's main data structure.
 * @return
 *          Pointer to the current argument list (icl_list_t *)
 * @ingroup QUARK
 */
void *QUARK_Args_List(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    return (void *)curr_task->args_list;
}

/* **************************************************************************** */
/**
 * Return the rank of a thread inside a parallel task.
 *
 * @param[in] quark
 *         The scheduler's main data structure.
 * @return
 *          Pointer to the current argument list (icl_list_t *)
 * @ingroup QUARK
 */
/* FIXME This is working but could be more efficient.  Depends on the
 * function being called only once by each instance in a
 * multi-threaded task */
int QUARK_Get_RankInTask(Quark *quark)
{
    int local_rank = 0;
    int global_rank = QUARK_Thread_Rank(quark);
    Task *curr_task = quark->worker[global_rank]->current_task_ptr;

    pthread_mutex_lock_wrap( &curr_task->task_mutex );
    local_rank = curr_task->threadid;
    curr_task->threadid++;
    pthread_mutex_unlock_wrap( &curr_task->task_mutex );

    return local_rank;
}

/* **************************************************************************** */
/**
 * Return a pointer to the next argument.  The variable last_arg
 * should be NULL on the first call, then each subsequent call will
 * use last_arg to get the the next argument. The argument list is
 * not actually popped, it is preserved intact.
 *
 * @param[in] args_list
 *         Pointer to the current arguments
 * @param[in,out] last_arg
 *         Pointer to the last argument; should be NULL on the first call
 * @return
 *          Pointer to the next argument
 * @ingroup QUARK
 */
void *QUARK_Args_Pop( void *args_list, void **last_arg)
{
    icl_list_t *args = (icl_list_t *)args_list;
    icl_list_t *node = (icl_list_t *)*last_arg;
    void *arg = NULL;
    if ( node == NULL ) {
        node = icl_list_first( args );
        if (node!=NULL) arg = node->data;
    } else {
        node = icl_list_next( args, node );
        if (node!=NULL) arg = node->data;
    }
    *last_arg = node;
    return arg;
}

/* **************************************************************************** */
/**
 * Well known hash function: Fowler/Noll/Vo - 32 bit version
 */
static inline unsigned int fnv_hash_function( void *key, int len )
{
    unsigned char *p = key;
    unsigned int h = 2166136261u;
    int i;
    for ( i = 0; i < len; i++ )
        h = ( h * 16777619 ) ^ p[i];
    return h;
}

/* **************************************************************************** */
/**
 * Hash function to map addresses, cut into "long" size chunks, then
 * XOR. The result will be matched to hash table size using mod in the
 * hash table implementation
 */
static inline unsigned int address_hash_function(void *address)
{
    int len = sizeof(void *);
    unsigned int hashval = fnv_hash_function( &address, len );
    return hashval;
}

/* **************************************************************************** */
/**
 * Adress compare function for hash table */
static inline int address_key_compare(void *addr1, void *addr2)
{
    return (addr1 == addr2);
}

/* **************************************************************************** */
/**
 * Hash function for unsigned long longs (used for taskid)
 */
static inline unsigned int ullong_hash_function( void *key )
{
    int len = sizeof(unsigned long long);
    unsigned int hashval = fnv_hash_function( key, len );
    return hashval;
}
/* **************************************************************************** */
/**
 * Compare unsigned long longs for hash keys (used for taskid)
 */
static inline int ullong_key_compare( void *key1, void *key2  )
{
    return ( *(unsigned long long*)key1 == *(unsigned long long*)key2 );
}

/* **************************************************************************** */
/**
 * Find the next worker thread that can have a task assigned to it.
 * Need to skip the manually scheduled threads, since the master
 * cannot assign work to them.  Abort if there is no such thread.
 */
static inline int quark_worker_find_next_assignable( Quark *quark )
{
    int id = quark->list_robin;
    quark->list_robin = ((quark->list_robin + 1) % quark->num_threads);
    return id;
}

/* **************************************************************************** */
/**
 * Duplicate the argument, allocating a memory buffer for it
 */
static inline char *arg_dup(char *arg, int size)
{
    char *argbuf = (char *) quark_malloc(size);
    memcpy(argbuf, arg, size);
    return argbuf;
}

/* **************************************************************************** */
/**
 * Allocate and initialize a dependency structure
 */
static inline Dependency *dependency_new(void *addr, long long size, quark_direction_t dir, bool loc, Task *task, bool accumulator, bool gatherv, int data_region, icl_list_t *task_args_list_node_ptr)
{
    Dependency *dep = (Dependency *) quark_malloc(sizeof(Dependency));
    dep->task = task;
    dep->address = addr;
    dep->size = size;
    dep->direction = dir;
    dep->locality = loc;
    dep->accumulator = accumulator;
    dep->data_region = data_region;
    dep->gatherv = gatherv;
    dep->address_set_node_ptr = NULL; /* convenience ptr, filled later */
    dep->address_set_waiting_deps_node_ptr = NULL; /* convenience ptr, filled later */
    dep->task_args_list_node_ptr = task_args_list_node_ptr; /* convenience ptr for WAR address updating */
    dep->task_dependency_list_node_ptr = NULL; /* convenience ptr */
    dep->ready = FALSE;
    /* For the task, track the dependency to be used to do locality
     * preservation; by default, use first output dependency.  */
    if ( dep->locality )
        task->locality_preserving_dep = dep;
    else if ( (task->locality_preserving_dep == NULL) && ( dep->direction==OUTPUT || dep->direction==INOUT) )
        task->locality_preserving_dep = dep;
    return dep;
}

/* **************************************************************************** */
/**
 * Allocate and initialize a worker structure
 */
static Worker *quark_worker_new(Quark *quark, int rank)
{
    Worker *worker = (Worker *) quark_malloc(sizeof(Worker));
    worker->thread_id = pthread_self();
    pthread_mutex_init( &worker->worker_mutex, NULL );
    worker->rank = rank;
    worker->ready_list = quark_malloc(sizeof(task_priority_tree_head_t));
    RB_INIT( worker->ready_list );
    worker->ready_list_size = 0;
    /* convenience pointer to the real args for the task  */
    worker->current_task_ptr = NULL;
    worker->quark_ptr = quark;
    worker->finalize = FALSE;
    worker->executing_task = FALSE;
    worker->set_to_manual_scheduling = FALSE;
    pthread_cond_init( &worker->worker_must_awake_cond, NULL );
    worker->status = WORKER_NOT_SLEEPING;
    return worker;
}

/* **************************************************************************** */
/**
 * Cleanup and free worker data structures.
 */
static void quark_worker_delete(Worker * worker)
{
    task_priority_tree_node_t *node, *nxt;
    /* Destroy the workers priority queue, if there is still anything there */
    for ( node = RB_MIN( task_priority_tree_head_s, worker->ready_list ); node != NULL; node = nxt) {
        nxt = RB_NEXT( task_priority_tree_head_s, worker->ready_list, node );
        RB_REMOVE( task_priority_tree_head_s, worker->ready_list, node );
        free(node);
    }
    free( worker->ready_list );
    pthread_mutex_destroy(&worker->worker_mutex);
    free(worker);
}

/* **************************************************************************** */
/**
 * The task requires scratch workspace, which will be allocated if
 * needed.  This records the scratch requirements.
 */
static Scratch *quark_scratch_new( void *arg_ptr, int arg_size, icl_list_t *task_args_list_node_ptr )
{
    Scratch *scratch = (Scratch *)quark_malloc(sizeof(Scratch));
    scratch->ptr = arg_ptr;
    scratch->size = arg_size;
    scratch->task_args_list_node_ptr = task_args_list_node_ptr;
    return(scratch);
}

/* **************************************************************************** */
/**
 * Allocate any needed scratch space;
 */
static void quark_scratch_allocate( Task *task )
{
    icl_list_t *scr_node;
    for (scr_node = icl_list_first( task->scratch_list );
         scr_node != NULL && scr_node->data != NULL;
         scr_node = icl_list_next(task->scratch_list, scr_node)) {
        Scratch *scratch = (Scratch *)scr_node->data;
        if ( scratch->ptr == NULL ) {
            /* Since ptr is null, space is to be allocted and attached */
            if ( scratch->size <= 0 ) quark_fatal_error( "quark_scratch_allocate", "scratch->size <= 0 " );
            void *scratchspace = quark_malloc( scratch->size );
            *(void **)scratch->task_args_list_node_ptr->data = scratchspace;
        }
    }
}

/* **************************************************************************** */
/**
 * Deallocate any scratch space.
 */
static void quark_scratch_deallocate( Task *task )
{
    icl_list_t *scr_node;
    for (scr_node = icl_list_first( task->scratch_list );
         scr_node != NULL && scr_node->data!=NULL;
         scr_node = icl_list_next(task->scratch_list, scr_node)) {
        Scratch *scratch = (Scratch *)scr_node->data;
        if ( scratch->ptr == NULL ) {
            /* If scratch had to be allocated, free it */
            free(*(void **)scratch->task_args_list_node_ptr->data);
        }
    }
}

/* **************************************************************************** */
/**
 * Called by the master thread.  This routine does not do thread
 * management, so it can be used with a larger libarary.  Allocate and
 * initialize the scheduler data stuctures for the master and
 * num_threads worker threads.
 *
 * @param[in] num_threads
 *          Number of threads to be used (1 master and rest compute workers).
 * @return
 *          Pointer to the QUARK scheduler data structure.
 * @ingroup QUARK
 */
Quark *QUARK_Setup(int num_threads)
{
    int i = 0;
    Quark *quark = (Quark *) quark_malloc(sizeof(Quark));
    /* Used to tell master when to act as worker */
    int quark_unroll_tasks_per_thread = quark_getenv_int("QUARK_UNROLL_TASKS_PER_THREAD", 50);
    int quark_unroll_tasks = quark_getenv_int("QUARK_UNROLL_TASKS", quark_unroll_tasks_per_thread * num_threads);
    quark->war_dependencies_enable = quark_getenv_int("QUARK_WAR_DEPENDENCIES_ENABLE", 0);
    quark->queue_before_computing = quark_getenv_int("QUARK_QUEUE_BEFORE_COMPUTING", 0);
    quark->dot_dag_enable = quark_getenv_int("QUARK_DOT_DAG_ENABLE", 0);
    //if ( quark->dot_dag_enable ) quark->queue_before_computing = 1;
    if ( quark->queue_before_computing==1 || quark_unroll_tasks==0 ) {
        quark->high_water_mark = (int)(INT_MAX - 1);
        quark->low_water_mark = (int)(quark->high_water_mark);
    } else {
        quark->low_water_mark = (int)(quark_unroll_tasks);
        quark->high_water_mark = (int)(quark->low_water_mark + quark->low_water_mark*0.25);
    }
    quark->num_queued_tasks = 0;
    pthread_mutex_init( &quark->num_queued_tasks_mutex, NULL );
    pthread_cond_init( &quark->num_queued_tasks_cond, NULL );
    quark->num_threads = num_threads;
    quark->list_robin = 0;
    pthread_mutex_init( &quark->quark_mutex, NULL );
    quark->start = FALSE;
    quark->all_tasks_queued = FALSE;
    quark->num_tasks = 0;
    quark->task_set = icl_hash_create( 0x1<<12, ullong_hash_function, ullong_key_compare );
    pthread_mutex_init( &quark->task_set_mutex, NULL );
    /* Create hash table to hold addresses */
    quark->address_set = icl_hash_create( 0x01<<12, address_hash_function, address_key_compare);
    pthread_mutex_init( &quark->address_set_mutex, NULL );
    /* To handle completed tasks */
    quark->completed_tasks = quark_malloc(sizeof(completed_tasks_head_t));
    TAILQ_INIT( quark->completed_tasks );
    pthread_mutex_init( &quark->completed_tasks_mutex, NULL );
    quark->completed_tasks_size = 0;
    /* Setup workers */
    quark->worker = (Worker **) quark_malloc(num_threads * sizeof(Worker *));
    /* The structure for the 0th worker will be used by the master */
    quark->worker[0] = quark_worker_new(quark, 0);
    quark->worker[0]->thread_id = pthread_self();
    quark->dot_dag_was_setup = 0;
    if ( quark->dot_dag_enable ) QUARK_DOT_DAG_Enable( quark, 1 );
    /* Launch workers; first create the structures */
    for(i = 1; i < num_threads; i++)
        quark->worker[i] = quark_worker_new(quark, i);
    /* Threads can start as soon as they want */
    quark->start = TRUE;
    return quark;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Allocate and initialize the scheduler
 * data stuctures and spawn worker threads.  Used when this scheduler
 * is to do all the thread management.
 *
 * @param[in] num_threads
 *          Number of threads to be used (1 master and rest compute workers).
 *          If num_threads < 1, first try environment variable QUARK_NUM_THREADS
 *          or use use num_threads = number of cores
 * @return
 *          Pointer to the QUARK data structure.
 * @ingroup QUARK
 */
Quark *QUARK_New(int num_threads)
{
    int i, nthrd;
    /* Init number of cores and topology */
    quark_topology_init();
    /* Get number of threads */
    if ( num_threads < 1 ) {
        nthrd = quark_get_numthreads();
        if ( nthrd == -1 ) nthrd = 1;
    } else {
        nthrd = num_threads;
    }
    /* Create scheduler data structures for master and workers */
    Quark *quark = QUARK_Setup(nthrd);
    /* Get binding informations */
    quark->coresbind = quark_get_affthreads();
    /* Setup thread attributes */
    pthread_attr_init(&quark->thread_attr);
    /* pthread_setconcurrency(quark->num_threads); */
    pthread_attr_setscope(&quark->thread_attr, PTHREAD_SCOPE_SYSTEM);
    /* Then start the threads, so that workers can scan the structures easily */
    for(i = 1; i < nthrd; i++) {
        int rc = pthread_create(&quark->worker[i]->thread_id, &quark->thread_attr, (void *(*)(void *))quark_work_set_affinity_and_call_main_loop, quark->worker[i]);
        if ( rc != 0 ) quark_fatal_error ( " QUARK_New", "Could not create threads properly" );
    }
    quark_setaffinity( quark->coresbind[0] );
    return quark;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Wait for all the tasks to be
 * completed, then return.  The worker tasks will NOT exit from their
 * work loop.
 *
 * @param[in,out] quark
 *         The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Barrier(Quark * quark)
{
    long long num_tasks = 1;
    /* Force queue_before_computing to be OFF!! */
    quark->queue_before_computing = FALSE;
    quark->all_tasks_queued = TRUE;
    do {
        quark_process_completed_tasks(quark);
        num_tasks = quark_work_main_loop( quark->worker[0] );
#ifdef QUARK_WITH_VALGRIND
        /* Asim: maybe you can have a signal right here ? */
        pthread_yield();
#endif
    } while ( num_tasks > 0 );
    /* FIXME Since address_set_nodes are not cleaned as the code progresses, they are freed here */
    if ( quark->dot_dag_enable ) {
        /* If dag generation is enabled, reset level counters */
        unsigned long long tasklevel = 0;
        for ( tasklevel=1; tasklevel<tasklevel_width_max_level; tasklevel++ ) 
            if ( quark->tasklevel_width[tasklevel] == 0 ) 
                break;
        tasklevel = tasklevel -1;
        int tmpint; icl_entry_t* tmpent; void *kp, *dp;
        icl_hash_foreach(quark->address_set, tmpint, tmpent, kp, dp) {
            Address_Set_Node *address_set_node = (Address_Set_Node *)dp;
            address_set_node->last_writer_tasklevel = tasklevel;
            address_set_node->last_reader_or_writer_tasklevel = tasklevel;
        }
        fprintf(dot_dag_file, "// QUARK_Barrier reached: level=%llu \n", tasklevel );
    } else {
        /* If NO dag generation, cleanup memory */
        icl_hash_destroy( quark->address_set, NULL, quark_address_set_node_free );
        quark->address_set = icl_hash_create( 0x01<<12, address_hash_function, address_key_compare);
    }
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Wait for all the
 * tasks to be completed, then return.  The worker tasks will also
 * exit from their work loop at this time.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Waitall(Quark * quark)
{
    int i;
    Worker *worker;
    QUARK_Barrier( quark );
    /* Tell each worker to exit the work_loop; master handles himself */
    for (i=1; i<quark->num_threads; i++) {
        worker = quark->worker[i];
        DBGPRINTF("Wkr %d [ %d ] setting finalize\n", worker->rank, worker->ready_list_size );
        quark_atomic_set( worker->finalize, TRUE, &worker->worker_mutex );
    }
    pthread_mutex_lock_wrap( &quark->num_queued_tasks_mutex );
    for (i=0; i<quark->num_threads; i++)
        pthread_cond_signal( &quark->worker[i]->worker_must_awake_cond );
    pthread_mutex_unlock_wrap( &quark->num_queued_tasks_mutex );
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Free all QUARK data structures, this
 * assumes that all usage of QUARK is completed.  This interface does
 * not manage, delete or close down the worker threads.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Free(Quark * quark)
{
    int i;
    QUARK_Waitall(quark);
    /* Write the level matching/forcing information */
    QUARK_DOT_DAG_Enable( quark, 0 );
    /* Destroy hash tables, workers and other data structures */
    for (i = 1; i < quark->num_threads; i++)
        quark_worker_delete( quark->worker[i] );
    quark_worker_delete( quark->worker[0] );
    if (quark->worker) free(quark->worker);
    if (quark->completed_tasks) free(quark->completed_tasks);
    icl_hash_destroy( quark->address_set, NULL, quark_address_set_node_free );
    icl_hash_destroy( quark->task_set, NULL, NULL );
    pthread_mutex_destroy(&quark->address_set_mutex);
    pthread_mutex_destroy(&quark->quark_mutex);
    free(quark);
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Wait for all tasks to complete, then
 * join/end the worker threads, and clean up all the data structures.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @ingroup QUARK
 */
void QUARK_Delete(Quark * quark)
{
    void *exitcodep = NULL;
    int   i;
    /* Wait for all tasks to complete */
    QUARK_Waitall( quark );
    /* Wait for workers to quit and join threads */
    for (i = 1; i < quark->num_threads; i++)
        pthread_join(quark->worker[i]->thread_id, &exitcodep);
    pthread_attr_destroy( &quark->thread_attr );
    /* Destroy specific structures */
    if (quark->coresbind) free(quark->coresbind);
    quark_topology_finalize();
    /* Destroy hash tables, workers and other data structures */
    QUARK_Free( quark );
}

/* **************************************************************************** */
/**
 * Use the task_flags data structure to set various items in the task
 * (priority, lock_to_thread, color, labels, etc ).
 */
static Task *quark_set_task_flags_in_task_structure( Quark *quark, Task *task, Quark_Task_Flags *task_flags )
{
    if ( task_flags ) {
        if ( task_flags->task_priority ) task->priority = task_flags->task_priority;
        if ( task_flags->task_lock_to_thread >= 0 ) task->lock_to_thread = task_flags->task_lock_to_thread;
        if ( task_flags->task_lock_to_thread_mask ) {
            int sizeofthreadmask = ( quark->num_threads%8==0 ? quark->num_threads/8 : quark->num_threads/8 + 1);
            if ( task->lock_to_thread_mask==NULL ) task->lock_to_thread_mask = quark_malloc( sizeofthreadmask );
            memcpy( task->lock_to_thread_mask, task_flags->task_lock_to_thread_mask, sizeofthreadmask );
        }
        if ( task_flags->task_color && quark->dot_dag_enable ) task->task_color = strdup(task_flags->task_color);
        if ( task_flags->task_label && quark->dot_dag_enable ) task->task_label = strdup(task_flags->task_label);
        if ( task_flags->task_sequence ) task->sequence = task_flags->task_sequence;
        if ( task_flags->task_thread_count > 1 ) task->task_thread_count = task_flags->task_thread_count;
        if ( task_flags->thread_set_to_manual_scheduling >= 0 ) task->thread_set_to_manual_scheduling = task_flags->thread_set_to_manual_scheduling;
    }
    return task;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  This is used in argument packing, to
 * create an initial task data structure.  Arguments can be packed
 * into this structure, and it can be submitted later.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] function
 *          The function (task) to be executed by the scheduler
 * @param[in] task_flags
 *          Flags to specify task behavior
 * @ingroup QUARK
 */
Task *QUARK_Task_Init(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags )
{
    Task *task = quark_task_new();
    task->function = function;
    quark_set_task_flags_in_task_structure( quark, task, task_flags );
    return task;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  This is used in argument packing, to
 * pack/add arguments to a task data structure.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in,out] task
 *          The task data struture to hold the arguments
 * @param[in] arg_size
 *          Size of the argument in bytes (0 cannot be used here)
 * @param[in] arg_ptr
 *          Pointer to data or argument
 * @param[in] arg_flags
 *          Flags indicating argument usage and various decorators
 *          INPUT, OUTPUT, INOUT, VALUE, NODEP, SCRATCH
 *          LOCALITY, ACCUMULATOR, GATHERV
 *          TASK_COLOR, TASK_LABEL (special decorators for VALUE)
 *          e.g., arg_flags    INPUT | LOCALITY | ACCUMULATOR
 *          e.g., arg_flags    VALUE | TASK_COLOR
 * @ingroup QUARK
 */
void QUARK_Task_Pack_Arg( Quark *quark, Task *task, int arg_size, void *arg_ptr, int arg_flags )
{
    int value_mask;
    bool arg_locality, accumulator, gatherv;
    int data_region;
    icl_list_t *task_args_list_node_ptr;
    Scratch *scratcharg;
    // extract information from the flags
    quark_direction_t arg_direction = (quark_direction_t) (arg_flags & QUARK_DIRECTION_BITMASK);
    switch ( arg_direction ) {
    case VALUE:
        /* If argument is a value; Copy the contents to the argument buffer */
        value_mask = ( arg_flags & QUARK_VALUE_FLAGS_BITMASK );
        if ( value_mask==0 ) {
            icl_list_append(task->args_list, arg_dup(arg_ptr, arg_size));
        } else if ( (arg_flags & TASK_PRIORITY) != 0 ) {
            task->priority = *((int *)arg_ptr);
        } else if ( (arg_flags & TASK_LOCK_TO_THREAD) != 0 ) {
            task->lock_to_thread = *((int *)arg_ptr);
        } else if ( (arg_flags & TASK_THREAD_COUNT) != 0 ) {
            task->task_thread_count = *((int *)arg_ptr);
        } else if ( (arg_flags & TASK_SEQUENCE) != 0 ) {
            task->sequence = *((Quark_Sequence **)arg_ptr);
        } else if ( (arg_flags & THREAD_SET_TO_MANUAL_SCHEDULING) != 0 ) {
            task->thread_set_to_manual_scheduling = *((int *)arg_ptr);
        } else if ( (arg_flags & TASK_COLOR) != 0 ) {
            if ( quark->dot_dag_enable ) {
                task->task_color = arg_dup(arg_ptr, arg_size);
            }
        } else if ( (arg_flags & TASK_LABEL) != 0 ) {
            if ( quark->dot_dag_enable ) {
                task->task_label = arg_dup(arg_ptr, arg_size);
            }
        }
        break;
    case NODEP:
        icl_list_append(task->args_list, arg_dup((char *) &arg_ptr, sizeof(char *)));
        break;
    case SCRATCH:
        task_args_list_node_ptr = icl_list_append(task->args_list, arg_dup((char *) &arg_ptr, sizeof(char *)));
        scratcharg = quark_scratch_new( arg_ptr, arg_size, task_args_list_node_ptr);
        icl_list_append( task->scratch_list, scratcharg );
        break;
    case INPUT:
    case OUTPUT:
    case INOUT:
    default:
        task_args_list_node_ptr = icl_list_append(task->args_list, arg_dup((char *) &arg_ptr, sizeof(char *)));
        arg_locality = (bool) ((arg_flags & LOCALITY) != 0 );
        accumulator = (bool) ((arg_flags & ACCUMULATOR) != 0 );
        gatherv = (bool) ((arg_flags & GATHERV) != 0 );
        if ( (arg_flags & QUARK_REGION_BITMASK) != 0 )
            data_region = (arg_flags & QUARK_REGION_BITMASK);
        else
            data_region = QUARK_REGION_ALL;
        // DBGPRINTF("Adding dependency arg_flags %x arg_direction %d data_region %x\n", arg_flags, arg_direction, data_region);
        Dependency *dep = dependency_new(arg_ptr, arg_size, arg_direction, arg_locality, task, accumulator, gatherv, data_region, task_args_list_node_ptr);
        /* Insert dependency in order of address; uses simple resource ordering to avoid deadlock situations  */
        icl_list_t *ptr = NULL;
        icl_list_t *task_dependency_list_node_ptr = NULL;
        for (ptr = icl_list_last(task->dependency_list); ptr != NULL; ptr = icl_list_prev(task->dependency_list, ptr)) {
            Dependency *ptrdep = (Dependency *)ptr->data;
            if (ptrdep->address > dep->address ) {
                task_dependency_list_node_ptr = icl_list_insert( task->dependency_list, ptr, dep );
                break;
            }
        }
        if ( ptr==NULL) task_dependency_list_node_ptr = icl_list_append( task->dependency_list, dep );
        dep->task_dependency_list_node_ptr = task_dependency_list_node_ptr;
        task->num_dependencies_remaining++;
        break;
    }
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Add a new task to the scheduler,
 * providing the data pointers, sizes, and dependency information.
 * This function provides the main user interface for the user to
 * write data-dependent algorithms.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in,out] task
 *          The packed task structure that already has all the
 *          arguments associated with the function
 * @return
 *          A long, long integer which can be used to refer to
 *          this task (e.g. for cancellation)
 * @ingroup QUARK
 */
unsigned long long QUARK_Insert_Task_Packed(Quark * quark, Task *task )
{
    long long num_tasks = -1;
    unsigned long long taskid = task->taskid;
    Quark_Sequence *sequence;
    quark_trace_event_start(INSERT_TASK_PACKED);
    task->task_thread_count_outstanding = task->task_thread_count;
    /* Track sequence information if it is provided */
    if ( task->sequence ) {
        sequence = task->sequence;
        pthread_mutex_lock_wrap( &sequence->sequence_mutex );
        if ( task->sequence->status == QUARK_ERR ) {
            /* If the sequence is cancelled or has error, return at once */
            task->function = NULL;
            pthread_mutex_unlock_wrap( &sequence->sequence_mutex );
            quark_task_delete( quark, task );
            return QUARK_ERR;
        } else {
            /* Otherwise insert this task into sequence */
            ll_list_node_t *entry = quark_malloc(sizeof(ll_list_node_t));
            entry->val = task->taskid;
            ll_list_head_t *headp = task->sequence->tasks_in_sequence;
            LIST_INSERT_HEAD( headp, entry, ll_entries );
            pthread_mutex_unlock_wrap( &task->sequence->sequence_mutex );
            /* Keep pointer to task in sequence so it can be deleted when task completes */
            task->ptr_to_task_in_sequence = entry;
        }
    }
    task->status = NOTREADY;
    quark_trace_addtask();
    /* Save the task in task_set, indexed by its taskid */
    pthread_mutex_lock_wrap( &quark->task_set_mutex );
    icl_hash_insert( quark->task_set, &task->taskid, task );
    quark->all_tasks_queued = FALSE;
    num_tasks = quark->num_tasks++;
    pthread_mutex_unlock_wrap( &quark->task_set_mutex );
    DBGPRINTF("Wkr %d [ %d ] Inserted %lld [ %d  %d ] into task set [ %lld ]\n", QUARK_Thread_Rank(quark), quark->worker[0]->ready_list_size, task->taskid, task->priority, task->task_thread_count, quark->num_tasks );
    /* Insert the task in the address hash, locking access to the address set hash */
    quark_insert_task_dependencies( quark, task );
    /* Check and see if task is ready for execution */
    pthread_mutex_lock_task( &task->task_mutex );
    quark_check_and_queue_ready_task( quark, task, -1 );
    pthread_mutex_unlock_task( &task->task_mutex );

    quark_trace_event_end();    

    /* If conditions are right, task insertion blocks and master
     * works; this will return when num_tasks becomes less than
     * low_water_mark */
    quark_process_completed_tasks(quark);

    while ( (quark->high_water_mark>0) && (num_tasks>=quark->high_water_mark) ) {
        num_tasks = quark_work_main_loop(quark->worker[0]);
        quark_process_completed_tasks(quark);
    }
    return taskid;
}

/* **************************************************************************** */
/**
 * Called by the master thread.  Add a new task to the scheduler,
 * providing the data pointers, sizes, and dependency information.
 * This function provides the main user interface for the user to
 * write data-dependent algorithms.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] function
 *          The function (task) to be executed by the scheduler
 * @param[in] task_flags
 *          Flags to specify task behavior
 * @param[in] ...
 *          Triplets of the form, ending with 0 for arg_size.
 *            arg_size, arg_ptr, arg_flags where
 *          arg_size: int: Size of the argument in bytes (0 cannot be used here)
 *          arg_ptr: pointer: Pointer to data or argument
 *          arg_flags: int: Flags indicating argument usage and various decorators
 *            INPUT, OUTPUT, INOUT, VALUE, NODEP, SCRATCH
 *            LOCALITY, ACCUMULATOR, GATHERV
 *            TASK_COLOR, TASK_LABEL (special decorators for VALUE)
 *            e.g., arg_flags    INPUT | LOCALITY | ACCUMULATOR
 *            e.g., arg_flags    VALUE | TASK_COLOR
 * @return
 *          A long, long integer which can be used to refer to
 *          this task (e.g. for cancellation)
 * @ingroup QUARK
 */
unsigned long long QUARK_Insert_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...)
{
    va_list varg_list;
    int arg_size;
    unsigned long long taskid;

    quark_trace_event_start(INSERT_TASK);
    Task *task = QUARK_Task_Init(quark, function, task_flags);
    /* For each argument */
    va_start(varg_list, task_flags);
    while( (arg_size = va_arg(varg_list, int)) != 0) {
        void *arg_ptr = va_arg(varg_list, void *);
        int arg_flags = va_arg(varg_list, int);
        QUARK_Task_Pack_Arg( quark, task, arg_size, arg_ptr, arg_flags );
    }
    va_end(varg_list);

    quark_trace_event_end();
    taskid = QUARK_Insert_Task_Packed( quark, task );

    return taskid;
}

/* **************************************************************************** */
unsigned long long QUARK_Execute_Task_Packed( Quark * quark, Quark_Task *task )
{
    /* For each dependency, wait till it is synced and empty */
    icl_list_t *dep_node;
    for (dep_node = icl_list_first(task->dependency_list);
         dep_node != NULL &&  dep_node->data!=NULL;
         dep_node = icl_list_next(task->dependency_list, dep_node)) {
        Dependency *dep = (Dependency *)dep_node->data;
        pthread_mutex_lock_address_set( &quark->address_set_mutex );
        Address_Set_Node *address_set_node = (Address_Set_Node *)icl_hash_find( quark->address_set, dep->address );
        pthread_mutex_unlock_address_set( &quark->address_set_mutex );
        if ( address_set_node != NULL ) {
            dep->address_set_node_ptr = address_set_node;
            quark_address_set_node_wait( quark, address_set_node );
        }
    }

    int thread_rank = QUARK_Thread_Rank(quark);
    Worker *worker = quark->worker[thread_rank];
    if ( task->function == NULL ) {
        /* This can occur if the task is cancelled */
        task->status = CANCELLED;
    } else {
        /* Call the task */
        task->status = RUNNING;
        worker->current_task_ptr = task;
        quark_scratch_allocate( task );
        task->function( quark );
        quark_scratch_deallocate( task );
        worker->current_task_ptr = NULL;
        task->status = DONE;
    }

    /* There is no real taskid to be returned, since the task has been deleted */
    return( 0 );
}

/* **************************************************************************** */
/**
 * Run this task in the current thread, at once, without scheduling.
 * This is an unsupported function that can be used by developers for
 * testing.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] function
 *          The function (task) to be executed by the scheduler
 * @param[in] task_flags
 *          Flags to specify task behavior
 * @param[in] ...
 *          Triplets of the form, ending with 0 for arg_size.
 *            arg_size, arg_ptr, arg_flags where
 *          arg_size: int: Size of the argument in bytes (0 cannot be used here)
 *          arg_ptr: pointer: Pointer to data or argument
 *          arg_flags: int: Flags indicating argument usage and various decorators
 *            INPUT, OUTPUT, INOUT, VALUE, NODEP, SCRATCH
 *            LOCALITY, ACCUMULATOR, GATHERV
 *            TASK_COLOR, TASK_LABEL (special decorators for VALUE)
 *            e.g., arg_flags    INPUT | LOCALITY | ACCUMULATOR
 *            e.g., arg_flags    VALUE | TASK_COLOR
 * @return
 *           Error value 0 since the task is run at once and there is no need for a task handle.
 * @ingroup QUARK_Unsupported
 */
unsigned long long QUARK_Execute_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...)
{
    va_list varg_list;
    int arg_size;

    Task *task = QUARK_Task_Init(quark, function, task_flags);

    va_start(varg_list, task_flags);
    // For each argument
    while( (arg_size = va_arg(varg_list, int)) != 0) {
        void *arg_ptr = va_arg(varg_list, void *);
        int arg_flags = va_arg(varg_list, int);
        QUARK_Task_Pack_Arg( quark, task, arg_size, arg_ptr, arg_flags );
    }
    va_end(varg_list);

    QUARK_Execute_Task_Packed( quark, task );

    /* Delete the task data structures */
    icl_list_destroy(task->args_list, free);
    icl_list_destroy(task->dependency_list, free);
    icl_list_destroy(task->scratch_list, free);
    pthread_mutex_destroy(&task->task_mutex);
    free(task);

    /* There is no real taskid to be returned, since the task has been deleted */
    return( 0 );
}

/* **************************************************************************** */
/**
 * Called by any thread.  Cancel a task that is in the scheduler.
 * This works by simply making the task a NULL task.  The scheduler
 * still processes all the standard dependencies for this task, but
 * when it is time to run the actual function, the scheduler does
 * nothing.
 *
 * @param[in,out] quark
 *          The scheduler's main data structure.
 * @param[in] taskid
 *          The taskid returned by a QUARK_Insert_Task
 * @return 1 on success.
 * @return -1 if the task cannot be found (may already be done and removed).
 * @return -2 if the task is aready running, done, or cancelled.
 * @ingroup QUARK
 */
int QUARK_Cancel_Task(Quark *quark, unsigned long long taskid)
{
    pthread_mutex_lock_wrap( &quark->task_set_mutex );
    Task *task = icl_hash_find( quark->task_set, &taskid );
    if ( task == NULL ) {
        pthread_mutex_unlock_wrap( &quark->task_set_mutex );
        return -1;
    }
    pthread_mutex_lock_task( &task->task_mutex );
    if ( task->status==RUNNING || task->status==DONE || task->status==CANCELLED ) {
        pthread_mutex_unlock_task( &task->task_mutex );
        pthread_mutex_unlock_wrap( &quark->task_set_mutex );
        return -2;
    }
    task->function = NULL;
    pthread_mutex_unlock_task( &task->task_mutex );
    pthread_mutex_unlock_wrap( &quark->task_set_mutex );
    return 1;
}

/* **************************************************************************** */
/**
 * Allocate and initialize address_set_node structure.  These are
 * inserted into the hash table.
 */
static Address_Set_Node *quark_address_set_node_new( void* address, int size )
{
    Address_Set_Node *address_set_node = (Address_Set_Node *)quark_malloc(sizeof(Address_Set_Node));
    address_set_node->address = address;
    address_set_node->size = size;
    address_set_node->last_thread = -1;
    address_set_node->waiting_deps = icl_list_new();
    if ( address_set_node->waiting_deps == NULL )
        quark_fatal_error( "quark_address_set_node_new", "Problem creating icl_list_new" );
    address_set_node->delete_data_at_address_when_node_is_deleted = FALSE;
    address_set_node->last_writer_taskid = 0;
    address_set_node->last_writer_tasklevel = 0;
    address_set_node->last_reader_or_writer_taskid = 0;
    address_set_node->last_reader_or_writer_tasklevel = 0;
    pthread_mutex_init( &address_set_node->asn_mutex, NULL );
    return address_set_node;
}

/* **************************************************************************** */
/**
 * Free address set node structures.
 */
static void quark_address_set_node_free( void* data )
{
    Address_Set_Node *address_set_node = (Address_Set_Node *)data;
    icl_list_destroy( address_set_node->waiting_deps, free );
    pthread_mutex_destroy( &address_set_node->asn_mutex );
    free (address_set_node );
}

/* **************************************************************************** */
/**
 * Queue ready tasks on a worker node, either using locality
 * information or a round robin scheme.  If the task uses multiple
 * threads (task_thread_count > 1) then put the task into multiple
 * worker queues.  Called when a task is initally inserted by the
 * master thread (from QUARK_Insert_Task_Packed), and when a task
 * finishes and subsequent tasks have dependencies satisfied (by all
 * threads via quark_work_main_loop_check_for_task). The
 * task->task_mutex must be locked when this is called.
 */
static void quark_check_and_queue_ready_task( Quark *quark, Task *task, int worker_rank )
{
    int worker_thread_id = -1;
    Worker *worker = NULL;
    int assigned_thread_count = 0;
    int first_worker_tid = -1;
    int i = 0;
    int wtid = 0;

    /* Quick return */
    if ( task->num_dependencies_remaining > 0 || task->status == QUEUED || task->status == RUNNING || task->status == DONE ) {
        return;
    }
    task->status = QUEUED;
    /* Assign task to thread.  Locked tasks get sent to appropriate
     * thread.  Locality tasks should have be correctly placed.  Tasks
     * without either should have the original round robin thread
     * assignment */
    if ( task->lock_to_thread >= 0 ) {
        worker_thread_id = task->lock_to_thread % quark->num_threads;
    }
    if ( worker_thread_id<0 && task->locality_preserving_dep != NULL ) {
        int test_thread_id = -1;
        if ( pthread_mutex_lock_address_set( &quark->address_set_mutex ) == 0 ) {
            Address_Set_Node *address_set_node = (Address_Set_Node *)icl_hash_find( quark->address_set, task->locality_preserving_dep->address );
            if ( address_set_node != NULL )
                /* The asn_mutex may already be locked, so it should not be locked here.  However it should not matter if we get an older version of the thread_id variable */
                test_thread_id = address_set_node->last_thread % quark->num_threads;
            pthread_mutex_unlock_address_set( &quark->address_set_mutex );
        }
        if (( test_thread_id >= 0 )
            /* test_thread_id is not set to manual scheduling */
            && ( quark->worker[test_thread_id]->set_to_manual_scheduling==FALSE )
            /* task is not locked to mask; OR it is stealable by test_thread_id */
            && ( task->lock_to_thread_mask==NULL || QUARK_Bit_Get( task->lock_to_thread_mask,test_thread_id)==1 ))
            worker_thread_id = test_thread_id;
    }
    /* Any unassiged tasks use round-robin on assignable workers to choose a worker thread */
    if ( worker_thread_id < 0 ) {
        for ( i=0; i<quark->num_threads; i++ ) {
            int test_thread_id = quark_worker_find_next_assignable( quark );
            if (( test_thread_id >= 0 )
                /* test_thread_id is not set to manual scheduling */
                && ( quark->worker[test_thread_id]->set_to_manual_scheduling==FALSE )
                /* task is not locked to mask; OR it is stealable by test_thread_id */
                && ( task->lock_to_thread_mask==NULL || QUARK_Bit_Get( task->lock_to_thread_mask,test_thread_id)==1 )) {
                worker_thread_id = test_thread_id;
                break;
            }
        }
    }

    /* Throw an error if for some reason no worker_thread could be found */
    if ( worker_thread_id < 0 )
        quark_fatal_error( "quark_check_and_queue_ready_task", "Task could not be assigned to any thread" );

    /* Parallel tasks using less than the total number of threads are not using thread 0 */
    if ( task->task_thread_count > quark->num_threads )
        quark_fatal_error( "quark_check_and_queue_ready_task", "Task requests more threads than available" );
    if ( ( task->task_thread_count > 1 )
         && ( quark->num_threads > task->task_thread_count )
         && ( worker_thread_id == 0 ))
      worker_thread_id++;

    first_worker_tid = worker_thread_id;
    while ( assigned_thread_count < task->task_thread_count) {
        worker = quark->worker[worker_thread_id];
        /* Create a new entry for the ready list */
        task_priority_tree_node_t *new_task_tree_node = quark_malloc(sizeof(task_priority_tree_node_t));
        new_task_tree_node->priority = task->priority;
        new_task_tree_node->task = task;
        /* Insert new entry into the ready list */
        if ( pthread_mutex_lock_ready_list( &worker->worker_mutex )==0 ) {
            RB_INSERT( task_priority_tree_head_s, worker->ready_list, new_task_tree_node );
            worker->ready_list_size++;
            pthread_mutex_unlock_ready_list(&worker->worker_mutex );
            quark_trace_addtask2worker(quark->worker[worker_thread_id]->thread_id);
        }
        assigned_thread_count++;
        /* DBGPRINTF("Wkr %d [ %d ] was_assigned tid %lld [ %d  %d/%d ] assigned by task %d\n", worker_thread_id, worker->ready_list_size, task->taskid, task->priority, assigned_thread_count, task->task_thread_count, QUARK_Thread_Rank(quark) ); */
        /* Set worker to manual scheduling at the time the task is
         * assigned to the worker.  Only the master should be changing
         * this variable, so it does not require locking. */
        if ( task->thread_set_to_manual_scheduling == 0 ) worker->set_to_manual_scheduling = FALSE;
        else if ( task->thread_set_to_manual_scheduling == 1 ) worker->set_to_manual_scheduling = TRUE;
        /* Wake up worker, or if it is already awake, some other sleeping worker */
        if ( pthread_mutex_lock_wrap( &quark->num_queued_tasks_mutex ) == 0 ) {
            quark->num_queued_tasks++;
            for ( wtid=worker_thread_id; ; ) {
                if ( quark->worker[wtid]->status == WORKER_SLEEPING ) {
                    pthread_cond_signal( &quark->worker[wtid]->worker_must_awake_cond );
                    break;
                }
                wtid = ( wtid + 1 ) % quark->num_threads;
                if ( wtid==worker_thread_id ) break;
            }
            pthread_mutex_unlock_wrap( &quark->num_queued_tasks_mutex );
        }
        if ( assigned_thread_count < task->task_thread_count ) {
            /* NOTE This is a special case, multi-threaded tasks do scheduling strangely */
            for ( worker_thread_id = (worker_thread_id+1) % quark->num_threads;
                  worker_thread_id != first_worker_tid;
                  worker_thread_id = (worker_thread_id+1) % quark->num_threads) {
                if (/* worker_thread_id is not set to manual scheduling */
                    ( quark->worker[worker_thread_id]->set_to_manual_scheduling==FALSE )
                    /* task is not locked to mask; OR it is stealable by worker_thread_id */
                    && ( task->lock_to_thread_mask==NULL || QUARK_Bit_Get( task->lock_to_thread_mask,worker_thread_id)==1 )) {
                    break;
                }
            }
            if ( worker_thread_id == first_worker_tid ) 
                quark_fatal_error("quark_check_and_queue_ready_task", "Not enough workers for mutithreaded task" );
        }
    }
}

/* **************************************************************************** */
/**
 * Routine to avoid false (WAR write-after-read) dependencies by
 * making copies of the data.  Check if there are suffient INPUTS in
 * the beginning of a address dependency followed by a OUTPUT or an
 * INOUT (data<-RRRRW).  If so, make a copy of the data, adjust the
 * pointers of the read dependencies to point to the new copy
 * (copy<-RRRR and data<-W) and send to workers if the tasks are
 * ready.  The copy can be automacally freed when all the reads are
 * done.  The write can proceed at once.  The asn_old->asn_mutex is
 * already locked when this is called.
 */
/* FIXME This entire routine needs to be redone!!  It does not work properly */
static void quark_avoid_war_dependencies( Quark *quark, Address_Set_Node *asn_old, Task *parent_task )
{
    /* Quick return if this is not enabled */
    if ( !quark->war_dependencies_enable ) return;

    /* Figure out if there are enough input dependencies to make this worthwhile */
    int count_initial_input_deps = 0;
    bool output_dep_reached = FALSE;
    int quark_num_queued_tasks;
    quark_atomic_get( quark_num_queued_tasks, quark->num_queued_tasks, &quark->num_queued_tasks_mutex );
    double avg_queued_tasks_per_thread = (double)quark_num_queued_tasks/(double)quark->num_threads;
    double avg_tasks_per_thread = (double)quark->num_tasks/(double)quark->num_threads;
    int min_input_deps;
    icl_list_t *dep_node_old;

    /* This stuff is still under development.... */
    if ( avg_queued_tasks_per_thread < 0.4 ) min_input_deps = 1;
    else if ( avg_queued_tasks_per_thread < 0.75 ) min_input_deps = 6;
    else if ( avg_queued_tasks_per_thread < 0.90 ) min_input_deps = 7;
    else if ( avg_queued_tasks_per_thread < 1.20 ) min_input_deps = 10;
    else if ( avg_queued_tasks_per_thread > 1.80 ) min_input_deps = 2000;
    else if ( avg_tasks_per_thread < (double)quark->low_water_mark/(double)quark->num_threads/2 ) min_input_deps = 2000;
    else min_input_deps = (int)(7 + 27 * avg_queued_tasks_per_thread);

    /* Override computed value using environment variable */
    min_input_deps = quark_getenv_int( "QUARK_AVOID_WAR_WHEN_NUM_WAITING_READS", min_input_deps );

    /* Scan thru initial deps, make sure they are inputs and that there
     * are enough of them to make data copying worthwhile */
    for (dep_node_old=icl_list_first(asn_old->waiting_deps);
         dep_node_old!=NULL;
         dep_node_old=icl_list_next(asn_old->waiting_deps, dep_node_old)) {
        Dependency *dep = (Dependency *)dep_node_old->data;
        Task *task = dep->task;
        if ( dep->direction==INPUT && task->status==NOTREADY  ) {
            count_initial_input_deps++;
        } else if ( (dep->direction==OUTPUT || dep->direction==INOUT) && task->status!=DONE ) {
            output_dep_reached = TRUE;
            break;
        }
    }

    /* if ( count_initial_input_deps>=quark->min_input_deps_to_avoid_war_dependencies && output_dep_reached ) { */
    if ( count_initial_input_deps>=min_input_deps && output_dep_reached ) {
        icl_list_t *dep_node_asn_old;
        Address_Set_Node *asn_new;
        /* Allocate and copy data */
        void *datacopy = quark_malloc( asn_old->size );
        /* Still need to track the allocated memory in datacopies TODO */
        /* quark->mem_allocated_to_war_dependency_data += asn_old->size; */
        memcpy( datacopy, asn_old->address, asn_old->size );
        /* Create address set node, attach to hash, and set it to clean up when done */
        asn_new = quark_address_set_node_new( datacopy, asn_old->size );
        asn_new->delete_data_at_address_when_node_is_deleted = TRUE;

        /* Update task dependences to point to this new data */
        /* Grab input deps from the old list, copy to new list, delete, then repeat */
        for ( dep_node_asn_old=icl_list_first(asn_old->waiting_deps);
              dep_node_asn_old!=NULL;  ) {
            icl_list_t *dep_node_asn_old_to_be_deleted = NULL;
            Dependency *dep = (Dependency *)dep_node_asn_old->data;
            Task *task = dep->task;
            if ( dep->direction==INPUT && task->status==NOTREADY ) {
                dep_node_asn_old_to_be_deleted = dep_node_asn_old;
                icl_list_t *dep_node_new = icl_list_append( asn_new->waiting_deps, dep );
                /* In the args list, set the arg pointer to the new datacopy address */
                *(void **)dep->task_args_list_node_ptr->data = datacopy;
                dep->address = asn_new->address;
                dep->address_set_node_ptr = asn_new;
                dep->address_set_waiting_deps_node_ptr = dep_node_new;
                if (dep->ready == FALSE) { /* dep->ready will always be FALSE */
                    dep->ready = TRUE;
                    dot_dag_print_edge( quark, parent_task->taskid, parent_task->tasklevel, task->taskid, task->tasklevel, DEPCOLOR );
                    pthread_mutex_lock_task( &task->task_mutex );
                    task->num_dependencies_remaining--;
                    quark_check_and_queue_ready_task( quark, task, -1 );
                    pthread_mutex_unlock_task( &task->task_mutex );
                }
            } else if ( (dep->direction==OUTPUT || dep->direction==INOUT) && task->status!=DONE ) {
                /* Once we return from this routine, this dep dependency will be processed */
                break;
            }
            dep_node_asn_old = icl_list_next(asn_old->waiting_deps, dep_node_asn_old);
            if (dep_node_asn_old_to_be_deleted!=NULL) {
                icl_list_delete(asn_old->waiting_deps, dep_node_asn_old_to_be_deleted, NULL);
            }
        }
        /* Insert the constructed asn_new into the address_set */
        pthread_mutex_lock_wrap( &quark->address_set_mutex );
        icl_hash_insert( quark->address_set, asn_new->address, asn_new );
        pthread_mutex_unlock_wrap( &quark->address_set_mutex );
    }
}

/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Sweeps through a sequence of GATHERV dependencies from the
 * beginning, and enables them all. Assumes asn_mutex is locked.
 */
static void quark_address_set_node_initial_gatherv_check_and_launch(Quark *quark, Address_Set_Node *address_set_node, Dependency *completed_dep, int worker_rank)
{
    icl_list_t *next_dep_node;
    Task *completed_task = completed_dep->task;
    for ( next_dep_node=icl_list_first(address_set_node->waiting_deps);
          next_dep_node!=NULL && next_dep_node->data != NULL;
          next_dep_node=icl_list_next(address_set_node->waiting_deps, next_dep_node) ) {
        Dependency *next_dep = (Dependency *)next_dep_node->data;
        /* Break when we run out of GATHERV output dependencies */
        if ( next_dep->gatherv==FALSE ) break;
        if ( next_dep->direction!=OUTPUT && next_dep->direction!=INOUT ) break;
        if ( next_dep->data_region != completed_dep->data_region ) break;
        Task *next_task = next_dep->task;
        /* Update next_dep ready status */
        if ( next_dep->ready == FALSE ) {
            /* Record the locality information with the task data structure */
            // if ( next_dep->locality ) next_task->locality_preserving_dep = worker_rank;
            /* Mark the next dependency as ready since we have GATHERV flag */
            next_dep->ready = TRUE;
            dot_dag_print_edge( quark, completed_task->taskid, completed_task->tasklevel, next_task->taskid, next_task->tasklevel, DEPCOLOR_GATHERV );
            pthread_mutex_lock_task( &next_task->task_mutex );
            next_task->num_dependencies_remaining--;
            /* If the dep status became true check related task, and put onto ready queues */
            quark_check_and_queue_ready_task( quark, next_task, worker_rank );
            pthread_mutex_unlock_task( &next_task->task_mutex );
        }
    }
}

/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Sweeps through a sequence of ACCUMULATOR tasks from the
 * beginning and prepends one at the beginning if only one (chained)
 * dependency remaining. This does not actually lauch the prepended
 * task, it depends on another function to do that. Assumes
 * asn_mutex is locked.
 */
static void quark_address_set_node_accumulator_find_prepend(Quark *quark, Address_Set_Node *address_set_node)
{
    icl_list_t *dep_node = NULL;
    Dependency *first_dep = NULL;
    icl_list_t *first_ready_dep_node = NULL;
    icl_list_t *last_ready_dep_node = NULL;
    icl_list_t *swap_node = NULL;
    int acc_dep_count = 0;

    /* FOR each ACCUMULATOR task waiting at the beginning of address_set_node  */
    for (dep_node = icl_list_first(address_set_node->waiting_deps);
         dep_node != NULL;
         dep_node = icl_list_next( address_set_node->waiting_deps, dep_node )) {
        Dependency *dependency = (Dependency *)dep_node->data;
        /* IF not an ACCUMULATOR dependency - break */
        if (dependency->accumulator == FALSE) break;
        Task *task = dependency->task;
        /* Scan through list keeping first, first_ready, last_ready, last */
        if ( first_dep==NULL ) first_dep = (Dependency *)dep_node->data;
        if ( task->num_dependencies_remaining==1 ) {
            if (first_ready_dep_node==NULL) first_ready_dep_node = dep_node;
            last_ready_dep_node = dep_node;
        }
        /* If the data_region changes, break */
        if ( dependency->data_region != first_dep->data_region ) break;
        acc_dep_count++;
    }

    /* Choose and move chosen ready node to the front of the list */
    /* Heuristic: Flip-flop between first-ready and last-ready.
     * Tested (always first, always last, flip-flop first/last) but
     * there was always a bad scenario.  If perfect loop orders are
     * provided (e.g. Choleky inversion test) then this will not make
     * performance worse.  If bad loops are provided, this will
     * improve performance, though not to the point of perfect
     * loops.  */
    if (acc_dep_count % 2 == 0 ) {
        if ( last_ready_dep_node!=NULL ) swap_node = last_ready_dep_node;
    } else {
        if ( first_ready_dep_node != NULL ) swap_node = first_ready_dep_node;
    }
    if ( swap_node != NULL ) {
        Dependency *dependency = (Dependency *)swap_node->data;
        /* Move to front of the address_set_node waiting_deps list (if not already there) */
        if ( swap_node!=icl_list_first(address_set_node->waiting_deps) ) {
            icl_list_t *tmp_swap_node = icl_list_prepend( address_set_node->waiting_deps, dependency );
            dependency->address_set_waiting_deps_node_ptr = tmp_swap_node;
            icl_list_delete( address_set_node->waiting_deps, swap_node, NULL );
        }
        /* Lock the dependency in place by setting ACC to false now */
        dependency->accumulator = FALSE;
    }
}


/* **************************************************************************** */
/** Note address_set_node->asn_mutex is locked when this is called;
 */
#if 0
static void quark_address_set_node_delete( Quark *quark, Address_Set_Node *address_set_node )
{
    return;
    /* FIXME; Currently does not free as soon as possible */
    if ( quark->dot_dag_enable == 0 ) {
        if ( icl_list_first( address_set_node->waiting_deps )==NULL ) {
            pthread_mutex_lock_address_set( &quark->address_set_mutex );
            icl_hash_delete( quark->address_set, address_set_node->address, NULL, NULL );
            /* Free data if it was allocted as a WAR data copy */
            if ( address_set_node->delete_data_at_address_when_node_is_deleted == TRUE )
                free( address_set_node->address );
            icl_list_destroy( address_set_node->waiting_deps, free );
            pthread_mutex_unlock_wrap( &address_set_node->asn_mutex );
            pthread_mutex_destroy( &address_set_node->asn_mutex );
            free( address_set_node );
            pthread_mutex_unlock_address_set( &quark->address_set_mutex );
        } else {
            pthread_mutex_unlock_wrap( &address_set_node->asn_mutex );
        }
    }
}
#endif

/* **************************************************************************** */
/**
 * Called by the master insert task dependencies into the hash table.
 * Any tasks that are ready to run are queued.
 */
static void quark_insert_task_dependencies(Quark * quark, Task * task)
{
    icl_list_t *task_dep_p = NULL; /* task dependency list pointer */

    /* For each task dependency list pointer */
    for (task_dep_p = icl_list_first(task->dependency_list);
         task_dep_p != NULL;
         task_dep_p = icl_list_next(task->dependency_list, task_dep_p)) {
        Dependency *dep = (Dependency *) task_dep_p->data;
        /* Lookup address in address_set hash, add it if it does not exist */
        pthread_mutex_lock_address_set( &quark->address_set_mutex );
        Address_Set_Node *address_set_node = (Address_Set_Node *)icl_hash_find( quark->address_set, dep->address );
        /* If not found, create a new address set node and add it to the hash */
        if ( address_set_node == NULL ) {
            address_set_node = quark_address_set_node_new( dep->address, dep->size );
            icl_hash_insert( quark->address_set, address_set_node->address, address_set_node );
        }
        /* Convenience shortcut pointer so that we don't have to hash again */
        dep->address_set_node_ptr = address_set_node;
        pthread_mutex_unlock_address_set( &quark->address_set_mutex );

        /* Lock the address_set_node and manipulate it */
        if ( pthread_mutex_lock_wrap( &address_set_node->asn_mutex ) == 0 ) {

            /* Add the dependency to the list of waiting dependencies on this address set node */
            icl_list_t *curr_dep_node = icl_list_append( address_set_node->waiting_deps, dep );
            /* Convenience shortcut pointer so we don't have to scan the waiting dependencies */
            dep->address_set_waiting_deps_node_ptr = curr_dep_node;
            /* Handle the case that the a single task makes multiple dependencies on the same data address */
            /* e.g. func( A11:IN, A11:INOUT, A11:OUT, A11:IN, A22:OUT )  */
            icl_list_t *prev_dep_node = icl_list_prev( address_set_node->waiting_deps, curr_dep_node);
            if ( prev_dep_node != NULL ) {
                Dependency *prev_dep = (Dependency *)prev_dep_node->data;
                Task *prev_task = prev_dep->task;
                if ( prev_task->taskid == task->taskid ) {
                    pthread_mutex_lock_task( &task->task_mutex );
                    DBGPRINTF( "task t%lld [label=\"%s %lld\" multiple dependencies on address %p];\n", task->taskid, task->task_label, task->taskid, dep->address );
                    /* The curr dependency will updated using the ordering INPUT < OUTPUT < INOUT  */
                    /* When the scheduler checks the front of the dependency list, it will find the correct dep setting */
                    dep->direction = (dep->direction > prev_dep->direction ? dep->direction : prev_dep->direction );
                    dep->data_region = (dep->data_region | prev_dep->data_region );
                    if ( prev_dep->ready == FALSE ) {
                        prev_dep->ready = TRUE;
                        task->num_dependencies_remaining--;
                    }
                    /* Remove the redundent dependency from waiting deps and from the task */
                    icl_list_delete( address_set_node->waiting_deps, prev_dep_node, NULL );
                    icl_list_delete( task->dependency_list, prev_dep->task_dependency_list_node_ptr, NULL );
                    /* Update the prev_dep_node ptr since it has changed */
                    prev_dep_node = icl_list_prev( address_set_node->waiting_deps, curr_dep_node);
                    pthread_mutex_unlock_task( &task->task_mutex );
                }
            }

            /* This will avoid WAR dependencies if possible: if enabled, and
             * the current dependency is a write, and there were only reads
             * earlier (input>1, output+inout=1) */
            if ( dep->direction==OUTPUT || dep->direction==INOUT ) {
                quark_avoid_war_dependencies( quark, address_set_node, task );
            }

            /* The following code decides whether the dep is ready or not */
            if ( dep->direction==INOUT || dep->direction==OUTPUT ) {
                /* If output, and previous dep exists, then ready=false */
                if ( prev_dep_node != NULL ) {
                    dep->ready = FALSE;
                } else {
                    dep->ready = TRUE;
                    dot_dag_print_edge( quark, address_set_node->last_writer_taskid, address_set_node->last_writer_tasklevel, task->taskid, task->tasklevel, DEPCOLOR_W_FIRST );
                    quark_atomic_add( task->num_dependencies_remaining, -1, &task->task_mutex );
                }
            } else if ( dep->direction == INPUT ) {
                if ( prev_dep_node != NULL ) {
                    /* If input, and previous dep is a read that is ready, then ready=true */
                    Dependency *prev_dep = (Dependency *)prev_dep_node->data;
                    if ( prev_dep->direction==INPUT && prev_dep->ready==TRUE ) {
                        dep->ready = TRUE;
                        dot_dag_print_edge( quark, address_set_node->last_writer_taskid, address_set_node->last_writer_tasklevel, task->taskid, task->tasklevel, DEPCOLOR_RAR );
                        quark_atomic_add( task->num_dependencies_remaining, -1, &task->task_mutex );
                    } else {
                        dep->ready = FALSE;
                    }
                } else {
                    /* Input, but no previous node (is first), so ready   */
                    dep->ready = TRUE;
                    dot_dag_print_edge( quark, address_set_node->last_writer_taskid, address_set_node->last_writer_tasklevel, task->taskid, task->tasklevel, DEPCOLOR_R_FIRST );
                    quark_atomic_add( task->num_dependencies_remaining, -1, &task->task_mutex );
                }
            }
            pthread_mutex_unlock_wrap( &address_set_node->asn_mutex );
        }
    }
}

/* **************************************************************************** */
/**
 * This function is called by a thread when it wants to start working.
 * This is used in a system that does its own thread management, so
 * each worker thread in that system must call this routine to get the
 * worker to participate in computation.
 *
 * @param[in,out] quark
 *          The main data structure.
 * @param[in] thread_rank
 *          The rank of the thread.
 * @ingroup QUARK
 */
void QUARK_Worker_Loop(Quark *quark, int thread_rank)
{
    quark->worker[thread_rank]->thread_id = pthread_self();
    quark_work_main_loop( quark->worker[thread_rank] );
}


/* **************************************************************************** */
/**
 * Called when spawning the worker thread to set affinity to specific
 * core and then call the main work loop.  This function is used
 * internally, when the scheduler spawns and manages the threads.  If
 * an external driver is using the scheduler (e.g. PLASMA) then it
 * does the thread management and any affinity must be set in the
 * external driver.
 */
static void quark_work_set_affinity_and_call_main_loop(Worker *worker)
{
    Quark *quark = worker->quark_ptr;
    int thread_rank = QUARK_Thread_Rank(quark);
    quark_setaffinity( quark->coresbind[thread_rank] );
    quark_work_main_loop( quark->worker[thread_rank] );
    return;
}


/* **************************************************************************** */
static Task *quark_work_main_loop_check_for_task( Quark *quark, Worker *worker, int worker_rank )
{
    Worker *worker_victim;
    task_priority_tree_node_t *task_priority_tree_node;
    Task *task = NULL;
    int ready_list_victim = worker_rank;
    int worker_finalize = FALSE;
    //int completed_tasks_size;
    int quark_num_queued_tasks = 0;

    /* Loop while looking for tasks */
    quark_atomic_get( worker_finalize, worker->finalize, &worker->worker_mutex );
    /* worker_finalize = worker->finalize; */
    while ( task==NULL && !worker->finalize ) {

        /* FIXME Tuning these statement is important to performance at small tile sizes */
        if ( worker_rank==0 ) quark_process_completed_tasks(quark);
        // else if ( worker->ready_list_size==0 ) quark_process_completed_tasks(quark);
        else if ( worker_rank%10==1 && worker->ready_list_size==0 && quark->completed_tasks_size>(20) ) quark_process_completed_tasks(quark);
        else if (quark->completed_tasks_size>=1 ) quark_process_completed_tasks(quark); //added 
        //else if ( completed_tasks_size>1 ) quark_process_completed_tasks(quark);
        // else quark_process_completed_tasks(quark);

        worker_victim = quark->worker[ready_list_victim];

        /* DBGPRINTF("Wkr %d [ %d ] looking at queue %d [ %d ]\n", worker->rank, worker->ready_list_size, ready_list_victim, worker_victim->ready_list_size ); */
        if ( worker_rank==ready_list_victim ) {
            if ( pthread_mutex_lock_ready_list( &worker_victim->worker_mutex  ) == 0 ) {
                task_priority_tree_node = RB_MIN( task_priority_tree_head_s, worker_victim->ready_list );
                if ( task_priority_tree_node != NULL ) {
                    task = task_priority_tree_node->task;
                    RB_REMOVE( task_priority_tree_head_s, worker_victim->ready_list, task_priority_tree_node );
                    free( task_priority_tree_node );
                    worker_victim->ready_list_size--;
                }
                pthread_mutex_unlock_ready_list( &worker_victim->worker_mutex );
            }
        } else if ( worker_rank!=ready_list_victim && worker_victim->executing_task==TRUE && worker_victim->ready_list_size>0 ) {
            if ( pthread_mutex_trylock_ready_list( &worker_victim->worker_mutex ) == 0) {
                if ( worker_victim->executing_task==TRUE && worker_victim->ready_list_size>0 ) { /* victim has at least so many tasks */
                    /* DBGPRINTF("Wkr %d [ %d ] Got lock for queue %d [ %d ]\n", worker->rank, worker->ready_list_size, ready_list_victim, worker_victim->ready_list_size ); */
                    //task_priority_tree_node = RB_MAX( task_priority_tree_head_s, worker_victim->ready_list );
                    task_priority_tree_node = RB_MIN( task_priority_tree_head_s, worker_victim->ready_list ); //modif: allow steal at the head 
                    if ( task_priority_tree_node != NULL ) {
                        Task *task_to_steal = task_priority_tree_node->task;
                        if ( task_to_steal->lock_to_thread == -1 /* task not locked, so steal OK */
                            /* && ( task_to_steal->task_thread_count == 1 )*/ //modif: allow steal multithreaded task /* We don't steal // task */
                             /* && !( ( worker_rank == 0 ) &&                       /\* worker 0 is allowed to steal // task *\/ */
                             /*       ( task_priority_tree_node->task->task_thread_count > 1 ) &&            /\* only if there is just enough thread  *\/ */
                             /*       ( quark->num_threads > task_priority_tree_node->task->task_thread_count ) ) ) */
                             && ( task_to_steal->lock_to_thread_mask==NULL /* task is not locked to mask, so steal OK */
                                  || QUARK_Bit_Get(task_to_steal->lock_to_thread_mask,worker_rank)==1 )) /* OR task is locked to mask, but steal by worker_rank is OK */
                        {
                            task = task_to_steal;
                            DBGPRINTF("Wkr %d [ %d ] Stealing tid %lld %p [ %d %d ] from thread %d [ %d ]\n", worker->rank, worker->ready_list_size, task->taskid, task->function, task->priority, task->task_thread_count, ready_list_victim, worker_victim->ready_list_size );
                            //printf("Wkr %d [ %d ] Stealing tid %lld %p [ %d %d ] from thread %d [ %d ]\n", worker->rank, worker->ready_list_size, task->taskid, task->function, task->priority, task->task_thread_count, ready_list_victim, worker_victim->ready_list_size ); //added
                            RB_REMOVE( task_priority_tree_head_s, worker_victim->ready_list, task_priority_tree_node );
                            free( task_priority_tree_node );
                            worker_victim->ready_list_size--;
                            quark_trace_deltask2worker(quark->worker[ready_list_victim]->thread_id);
                            quark_trace_addtask2worker(quark->worker[worker_rank]->thread_id);
                        }
                    }
                }
                pthread_mutex_unlock_ready_list( &worker_victim->worker_mutex );
            }
        }
        /* If no task found */
        if ( task == NULL ) {
            /* If there are no tasks, wait for a task to be introduced, then check own queue first */
            /* If this worker is allowed to do work stealing, then move and check the next victim queue */
            if ( worker->set_to_manual_scheduling == FALSE )
                ready_list_victim = (ready_list_victim + 1) % quark->num_threads;
            /* Break for master when a scan of all queues is finished and no tasks were found or no work is available */
            if ( worker_rank==0 && ( ready_list_victim==0 || quark->num_queued_tasks==0 ) ) return NULL;
            /* Grab some high level counters */
            quark_atomic_get( quark_num_queued_tasks, quark->num_queued_tasks, &quark->num_queued_tasks_mutex );
            quark_atomic_get( worker_finalize, worker->finalize, &worker->worker_mutex );
            /* Wait for work */
            if ( quark_num_queued_tasks==0 && worker_rank!=0 ) {
                pthread_mutex_lock_wrap( &quark->num_queued_tasks_mutex );
                quark_num_queued_tasks = quark->num_queued_tasks;
                worker_finalize = worker->finalize;
                DBGPRINTF("Wkr %d [ %d ] Goes to sleep\n", worker->rank, worker->ready_list_size );
                while ( quark_num_queued_tasks==0 && !worker_finalize ) {
                    worker->status = WORKER_SLEEPING;
                    pthread_cond_wait_wrap( &quark->worker[worker_rank]->worker_must_awake_cond, &quark->num_queued_tasks_mutex );
                    quark_num_queued_tasks = quark->num_queued_tasks;
                    worker_finalize = worker->finalize;
                }
                worker->status = WORKER_NOT_SLEEPING;
                DBGPRINTF("Wkr %d [ %d ] Wakes up\n", worker->rank, worker->ready_list_size );
                pthread_mutex_unlock_wrap( &quark->num_queued_tasks_mutex );
                DBGPRINTF("Wkr %d [ %d ] Unlock quark->num_queued_tasks_mutex\n", worker->rank, worker->ready_list_size );
            }
        }
        quark_atomic_get( worker_finalize, worker->finalize, &worker->worker_mutex );
    }
    DBGPRINTF("Wkr %d [ %d ] found a task and is returning with it or got a finalize\n", worker->rank, worker->ready_list_size );
    return task;
}


/* **************************************************************************** */
/**
 * Called by the workers (and master) to continue executing tasks
 * until some exit condition is reached.
 */
static long long quark_work_main_loop(Worker *worker)
{
    Quark *quark = worker->quark_ptr;
    Task *task = NULL;
    long long num_tasks = -1;
    /*int worker_finalize = FALSE;*/

    /* Busy wait while not ready */
    do {} while ( !quark->start );
    int worker_rank = QUARK_Thread_Rank(quark);

    /* Queue all tasks before running; this can be enabled via environment */
    if ( quark->queue_before_computing && worker_rank==0 && !quark->all_tasks_queued ) return quark->num_tasks;
    while ( quark->queue_before_computing && worker_rank!=0 && !quark->all_tasks_queued ) { /* busy loop */ }

    /* Master never does work; this line for debugging use  */
    /* if (worker_rank == 0) return; */
    /* DBGPRINTF("Wkr %d [ %d ] Starting main loop\n", worker->rank, worker->ready_list_size); */
    /* quark_atomic_get( worker_finalize, worker->finalize, &worker->worker_mutex ); */
    while ( worker->finalize == FALSE ) {
        /* Repeatedly try to find a task, first trying my own ready list,
         * then trying to steal from someone else */
        task = quark_work_main_loop_check_for_task( quark, worker, worker_rank );

        /* EXECUTE THE TASK IF FOUND */
        if ( task!=NULL ) {
            DBGPRINTF("Wkr %d [ %d ] Found a task\n", worker->rank, worker->ready_list_size);
            int sequence_status = 0;
            if ( task->sequence!=NULL ) {
                pthread_mutex_lock_wrap( &task->sequence->sequence_mutex );
                sequence_status = task->sequence->status;
                pthread_mutex_unlock_wrap( &task->sequence->sequence_mutex );
            }
            pthread_mutex_lock_task( &task->task_mutex );
            if ( (sequence_status==QUARK_ERR) || (task->function==NULL) ) { /* cancelled */
                DBGPRINTF("Wkr %d [ %d ] Task was cancelled %lld\n", worker->rank, worker->ready_list_size, task->taskid);
                task->status = CANCELLED;
                pthread_mutex_unlock_task( &task->task_mutex );
            } else { /* Call the task */
                quark_atomic_set( worker->executing_task, TRUE, &worker->worker_mutex );
                task->status = RUNNING;
                quark_scratch_allocate( task );
                pthread_mutex_unlock_task( &task->task_mutex );
                worker->current_task_ptr = task;
                quark_trace_deltask2worker(quark->worker[worker_rank]->thread_id);
#ifdef DBGQUARK
                struct timeval tstart; gettimeofday( &tstart, NULL );
#endif /* DBGQUARK */
                /* THIS IS THE ACTUAL CALL TO EXECUTE THE FUNCTION */
                task->function( quark );
#ifdef DBGQUARK
                struct timeval tend; gettimeofday( &tend, NULL );
                struct timeval tresult; timersub( &tend, &tstart, &tresult );
                DBGPRINTF("Wkr %d [ %d ] Did tid %lld %p [ %d %d ] %f\n", worker->rank, worker->ready_list_size, task->taskid, task->function, task->priority, task->task_thread_count, (double)tresult.tv_sec + (double)tresult.tv_usec/1000000.0 );
#endif /* DBGQUARK */
                pthread_mutex_lock_task( &task->task_mutex );
                quark_scratch_deallocate( task );
                task->executed_on_threadid = worker_rank;
                task->status = DONE;
                pthread_mutex_unlock_task( &task->task_mutex );
                quark_atomic_set( worker->executing_task, FALSE, &worker->worker_mutex );
            }
            /* Put the task into a queue for later processing */
            quark_worker_remove_completed_task_enqueue_for_later_processing(quark, task, worker_rank);
        }
        /* Break if master */
        if ( worker_rank==0 ) break;
        /* quark_atomic_get( worker_finalize, worker->finalize, &worker->worker_mutex ); */
    }
    /* DBGPRINTF("Wkr %d [ %d ] Leaving main loop with finalize %d\n", worker->rank, worker->ready_list_size, worker->finalize); */
    /* Worker has exited loop; ready for next time this worker is activated */
    quark_atomic_set( worker->finalize, FALSE, &worker->worker_mutex );
    /* Get the num_tasks in the system and return it */
    quark_atomic_get( num_tasks, quark->num_tasks, &quark->task_set_mutex );
    DBGPRINTF("Wkr %d [ %d ] Exiting main work loop with num_tasks %lld\n", worker->rank, worker->ready_list_size, num_tasks );
    return num_tasks;
}


/* **************************************************************************** */
/**
 * Called by the control program.  Creates a new sequence data
 * structure and returns it.  This can be used to put a sequence of
 * tasks into a group and cancel that group if an error condition
 * occurs.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @return  Pointer to the newly created sequence structure.
 * @ingroup QUARK
 */
Quark_Sequence *QUARK_Sequence_Create( Quark *quark )
{
    Quark_Sequence *sequence = quark_malloc(sizeof(Quark_Sequence));
    DBGPRINTF("Wkr %d [ %d ] In seq create\n", QUARK_Thread_Rank(quark), quark->worker[0]->ready_list_size );
    sequence->status = QUARK_SUCCESS;
    pthread_mutex_init( &sequence->sequence_mutex, NULL );
    ll_list_head_t *head = quark_malloc(sizeof(ll_list_head_t));
    LIST_INIT(head);
    sequence->tasks_in_sequence = head;
    return sequence;
}

/* **************************************************************************** */
/**
 * Can be called by any thread.  Cancels all the remaining tasks in a
 * sequence using QUARK_Cancel_Task and changes the state so that
 * future tasks belonging to that sequence are ignored.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in,out] sequence
 *          Pointer to a sequence data structure
 * @return 0 (QUARK_SUCCESS) on success
 * @return -1 (QUARK_ERR) on failure
 * @ingroup QUARK
 */
int QUARK_Sequence_Cancel( Quark *quark, Quark_Sequence *sequence )
{
    int retval;
    if ( quark==NULL || sequence==NULL ) return QUARK_ERR;
    pthread_mutex_lock_wrap( &sequence->sequence_mutex );
    if ( sequence->status != QUARK_SUCCESS ) {
        /* sequence already cancelled */
        retval = QUARK_SUCCESS;
    } else {
        sequence->status = QUARK_ERR;
        ll_list_node_t *np;
        LIST_FOREACH( np, sequence->tasks_in_sequence, ll_entries ) {
            long long int taskid = np->val;
            /* Find taskid, make function NULL */
            QUARK_Cancel_Task( quark, taskid );
            /* Task node is removed from sequence when it finishes and is
             * deleted; or when sequence is destroyed */
        }
        retval = QUARK_SUCCESS;
    }
    pthread_mutex_unlock_wrap( &sequence->sequence_mutex );
    return retval;
}

/* **************************************************************************** */
/**
 * Called by the control program.  Cancels all the remaining tasks in
 * a sequence using QUARK_Cancel_Task and deletes the sequence data
 * structure.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in,out] sequence
 *          Pointer to a sequence data structure
 * @return A NULL pointer; which can be used to reset the sequence structure
 * @ingroup QUARK
 */
Quark_Sequence *QUARK_Sequence_Destroy( Quark *quark, Quark_Sequence *sequence )
{
    DBGPRINTF("Wkr %d [ %d ] In seq destroy \n", QUARK_Thread_Rank(quark), quark->worker[0]->ready_list_size );
    if ( quark==NULL || sequence==NULL ) return NULL;
    if ( !LIST_EMPTY( sequence->tasks_in_sequence )) {
        if ( QUARK_Sequence_Cancel( quark, sequence ) != QUARK_SUCCESS ) return NULL;
        if ( QUARK_Sequence_Wait( quark, sequence ) != QUARK_SUCCESS ) return NULL;
    }
    /* Dont need to remove tasks in sequence, should have been removed by sequence_wait */
    free( sequence->tasks_in_sequence );
    sequence->tasks_in_sequence = NULL;
    pthread_mutex_destroy( &sequence->sequence_mutex );
    free( sequence );
    return NULL;
}

/* **************************************************************************** */
/**
 * Called by the control program.  Returns when all the tasks in a
 * sequence have completed.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in,out] sequence
 *          Pointer to a sequence structure
 * @return  0 on success
 * @return  -1 on failure
 * @ingroup QUARK
 */
int QUARK_Sequence_Wait( Quark *quark, Quark_Sequence *sequence )
{
    if ( quark==NULL || sequence==NULL) return QUARK_ERR;
    int myrank = QUARK_Thread_Rank( quark );
    while ( !LIST_EMPTY( sequence->tasks_in_sequence ) ) {
        quark_process_completed_tasks( quark );
        quark_work_main_loop( quark->worker[myrank] );
    }
    return QUARK_SUCCESS;
}

/* **************************************************************************** */
/**
 * For the current thread, in the current task being executed, return
 * the task's sequence value.  This is the value provided when the
 * task was Task_Inserted into a sequence.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @return Pointer to sequence data structure
 * @ingroup QUARK
 */
Quark_Sequence *QUARK_Get_Sequence(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    return (Quark_Sequence *)curr_task->sequence;
}

/* **************************************************************************** */
/**
 * For the current thread, in the current task being executed, return
 * the task's priority value.  This is the value provided when the
 * task was Task_Inserted.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @return priority of the task
 * @ingroup QUARK_Depreciated
 */
int QUARK_Get_Priority(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    return curr_task->priority;
}

/* **************************************************************************** */
/**
 * For the current thread, in the current task being executed, return
 * the task label.  This is the value that was optionally provided
 * when the task was Task_Inserted.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @return Pointer to null-terminated label string
 * @return NULL if there is no label
 * @ingroup QUARK_Depreciated
 */
char *QUARK_Get_Task_Label(Quark *quark)
{
    Task *curr_task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    return (char *)curr_task->task_label;
}


/* **************************************************************************** */
/**
 * When a task is completed, queue it for further handling by another
 * process.
 */
static void quark_worker_remove_completed_task_enqueue_for_later_processing(Quark *quark, Task *task, int worker_rank)
{
    int threads_remaining_for_this_task = -1;
    pthread_mutex_lock_task( &task->task_mutex );
    threads_remaining_for_this_task = --task->task_thread_count_outstanding;
    pthread_mutex_unlock_task( &task->task_mutex );
    if ( threads_remaining_for_this_task == 0 ) {
        completed_tasks_node_t *node = quark_malloc(sizeof(completed_tasks_node_t));
        node->task = task;
        node->workerid = worker_rank;
        pthread_mutex_lock_completed_tasks( &quark->completed_tasks_mutex );
        TAILQ_INSERT_TAIL( quark->completed_tasks, node, ctn_entries );
        quark->completed_tasks_size++;
        pthread_mutex_unlock_completed_tasks( &quark->completed_tasks_mutex );
    }
}

/* **************************************************************************** */
/**
 * Handle the queue of completed tasks.
 */
static void quark_process_completed_tasks( Quark *quark )
{
    int completed_tasks_size;
    Task *task;
    int workerid = -1;
    quark_atomic_get( completed_tasks_size, quark->completed_tasks_size, &quark->completed_tasks_mutex );
    if ( completed_tasks_size==0 ) return;
    quark_trace_event_start(PROCESS_COMPLETED_TASKS);
    do {
        task = NULL;
        if ( pthread_mutex_trylock_completed_tasks( &quark->completed_tasks_mutex ) == 0 ) {
            completed_tasks_node_t *completed_task_node = TAILQ_FIRST( quark->completed_tasks );
            if ( completed_task_node!= NULL ) {
                TAILQ_REMOVE( quark->completed_tasks, completed_task_node, ctn_entries );
                quark->completed_tasks_size--;
                task = completed_task_node->task;
                workerid = completed_task_node->workerid;
                free( completed_task_node );
            }
            pthread_mutex_unlock_completed_tasks( &quark->completed_tasks_mutex );
        }
        if ( task != NULL )
            quark_remove_completed_task_and_check_for_ready( quark, task, workerid );
    } while ( task!=NULL );
    quark_trace_event_end();
}

/* **************************************************************************** */
/**
 * Called by a worker each time a task is removed from an address set
 * node.  Sweeps through a sequence of initial dependencies on an
 * address, and launches any that are ready to go. Uses and tracks the
 * regions of each data tile that are to be accessed for reads and
 * writes.  Assumes asn_mutex is locked.
 */
static void quark_address_set_node_initial_check_and_launch( Quark *quark, Address_Set_Node *address_set_node, Dependency *completed_dep, int worker_rank )
{
    int read_data_region = 0;
    int write_data_region = 0;
    icl_list_t *dep_node = NULL;
    int keep_processing_more_nodes = 1;

    for ( dep_node=icl_list_first( address_set_node->waiting_deps );
          dep_node!=NULL && keep_processing_more_nodes==1;
          dep_node=icl_list_next( address_set_node->waiting_deps, dep_node )) {
        Dependency *dep = (Dependency *)dep_node->data;
        Task *task = dep->task;
        /* NOTE Skip CANCELLED and DONE tasks */
        if ( task->status==CANCELLED || task->status==DONE ) continue;
        switch ( dep->direction ) {
        case INPUT:
            if ( (dep->data_region & write_data_region) == 0 ) {
                if ( dep->ready==FALSE ) {
                    dep->ready = TRUE;
                    pthread_mutex_lock_task( &task->task_mutex );
                    task->num_dependencies_remaining--;
                    quark_check_and_queue_ready_task( quark, task, worker_rank );
                    pthread_mutex_unlock_task( &task->task_mutex );
                    dot_dag_print_edge( quark, completed_dep->task->taskid, completed_dep->task->tasklevel, task->taskid, task->tasklevel, DEPCOLOR_RAW );
                }
            }
            read_data_region = read_data_region | dep->data_region;
            break;
        case OUTPUT:
        case INOUT:
            if ( ((dep->data_region & write_data_region)==0) && ((dep->data_region & read_data_region)==0) ) {
                if ( dep->ready==FALSE ) {
                    dep->ready = TRUE;
                    pthread_mutex_lock_task( &task->task_mutex );
                    task->num_dependencies_remaining--;
                    quark_check_and_queue_ready_task( quark, task, worker_rank );
                    pthread_mutex_unlock_task( &task->task_mutex );
                    if ( quark->dot_dag_enable ) {
                        if ( completed_dep->direction==INPUT ) {
                            dot_dag_print_edge( quark, completed_dep->task->taskid, completed_dep->task->tasklevel, task->taskid, task->tasklevel, DEPCOLOR_WAR );
                        } else {
                            dot_dag_print_edge( quark, completed_dep->task->taskid, completed_dep->task->tasklevel, task->taskid, task->tasklevel, DEPCOLOR_WAW );
                        }
                    }
                }
            } /* else keep_processing_more_nodes = 0; */
            write_data_region = write_data_region | dep->data_region;
            if ( write_data_region==QUARK_REGION_ALL )
                keep_processing_more_nodes = 0;
            break;
        case VALUE:
        case NODEP:
        case SCRATCH:
        default:
            DBGPRINTF("Unexpected dependency direction (not INPUT, OUTPUT, INOUT)\n");
            break;
        }
    }
}
/* **************************************************************************** */
/**
 * Handle a single completed task, finding its children and putting
 * the children that are ready to go (all dependencies satisfied) into
 * worker ready queues.
 */
static void quark_remove_completed_task_and_check_for_ready(Quark *quark, Task *task, int worker_rank)
{
    if ( quark->dot_dag_enable ) {
        pthread_mutex_lock_wrap( &quark->dot_dag_mutex );
        //if (task->tasklevel < 1) task->tasklevel=1;
        fprintf(dot_dag_file, "t%llu [fillcolor=\"%s\",label=\"%s\",style=filled]; // %llu %d %p %d %llu \n",
                task->taskid, task->task_color, task->task_label, task->taskid, task->priority, task->sequence, task->task_thread_count, task->tasklevel);
        /* Track the width of each task level */
        quark->tasklevel_width[task->tasklevel]++;
        /* fprintf(dot_dag_file, "// critical-path depth %ld \n", task->tasklevel ); */
        fprintf(dot_dag_file, "{rank=same;%llu;t%llu};\n", task->tasklevel, task->taskid );
        pthread_mutex_unlock_wrap( &quark->dot_dag_mutex );
    }

    /* For each dependency in the task that was completed */
    icl_list_t *dep_node;
    for (dep_node = icl_list_first(task->dependency_list);
         dep_node != NULL &&  dep_node->data!=NULL;
         dep_node = icl_list_next(task->dependency_list, dep_node)) {
        Dependency  *dep = (Dependency *)dep_node->data;
        Address_Set_Node *address_set_node = dep->address_set_node_ptr;

        if ( pthread_mutex_lock_wrap( &address_set_node->asn_mutex )==0 ) {
            /* Mark the address/data as having been written by worker_rank  */
            if ( dep->direction==OUTPUT || dep->direction==INOUT )
                address_set_node->last_thread = worker_rank;
            /* Update dag generation information */
            if ( quark->dot_dag_enable ) {
                if ( dep->direction==OUTPUT || dep->direction==INOUT ) {
                    /* Track last writer and level, needed when this structure becomes empty */
                    address_set_node->last_writer_taskid = task->taskid;
                    address_set_node->last_writer_tasklevel = task->tasklevel;
                }
                address_set_node->last_reader_or_writer_taskid = task->taskid;
                address_set_node->last_reader_or_writer_tasklevel = task->tasklevel;
            }
            /* Check the address set node to avoid WAR dependencies */
            if ( (quark->war_dependencies_enable) &&
                 (dep->direction==OUTPUT || dep->direction==INOUT) )
                quark_avoid_war_dependencies( quark, address_set_node, task );
            /* Remove competed dependencies from address_set_node waiting_deps list */
            icl_list_delete( address_set_node->waiting_deps, dep->address_set_waiting_deps_node_ptr, NULL );
            /* If dependencies are waiting ... */
            if ( icl_list_first(address_set_node->waiting_deps) != NULL ) {
                /* Handle any initial GATHERV dependencies */
                quark_address_set_node_initial_gatherv_check_and_launch(quark, address_set_node, dep, worker_rank);
                /* Prepend any initial accumulater dependency that is ready to go */
                quark_address_set_node_accumulator_find_prepend( quark, address_set_node );
                /* Initial input and or output */
                quark_address_set_node_initial_check_and_launch( quark, address_set_node, dep, worker_rank );
                pthread_mutex_unlock_wrap( &address_set_node->asn_mutex );
            } else { /* if ( icl_list_first(address_set_node->waiting_deps) == NULL ) { */
                pthread_mutex_unlock_wrap( &address_set_node->asn_mutex );
                /* FIXME the address set node is not actually deleted */
                // quark_address_set_node_delete( quark, address_set_node );
            }
        }
    }
    DBGPRINTF("Wkr %d [ %d ] deleting task %lld\n", worker_rank, quark->worker[worker_rank]->ready_list_size, task->taskid );
    task = quark_task_delete(quark, task);
    quark_atomic_add( quark->num_queued_tasks, -1, &quark->num_queued_tasks_mutex );
}

/* **************************************************************************** */
/**
 * Set various task level flags.  This flag data structure is then
 * provided when the task is created/inserted.  Each flag can take a
 * value which is either an integer or a pointer.
 *
 *          Select from one of the flags:
 *          TASK_PRIORITY : an integer (0-MAX_INT)
 *          TASK_LOCK_TO_THREAD : an integer for the thread number
 *          TASK_LOCK_TO_THREAD_MASK : a pointer to a bitmask where task can run
 *          TASK_LABEL : a string pointer (NULL terminated) for the label
 *          TASK_COLOR :  a string pointer (NULL terminated) for the color.
 *          TASK_SEQUENCE : takes pointer to a Quark_Sequence structure
 *          THREAD_SET_TO_MANUAL_SCHEDULING: boolean integer {0,1} setting thread to manual (1) or automatic (0) scheduling
 *
 * @param[in,out] task_flags
 *          Pointer to a Quark_Task_Flags structure
 * @param[in] flag
 *          One of the flags listed above
 * @param[in] val
 *          A integer or a pointer value for the flag ( uses the intptr_t )
 * @return Pointer to the updated Quark_Task_Flags structure
 * @ingroup QUARK
 */
Quark_Task_Flags *QUARK_Task_Flag_Set( Quark_Task_Flags *task_flags, int flag, intptr_t val )
{
    switch (flag)  {
    case TASK_PRIORITY:
        task_flags->task_priority = (int)val;
        break;
    case TASK_LOCK_TO_THREAD:
        task_flags->task_lock_to_thread = (int)val;
        break;
    case TASK_LOCK_TO_THREAD_MASK:
        task_flags->task_lock_to_thread_mask = (unsigned char *)val;
        break;
    case TASK_LABEL:
        task_flags->task_label = (char *)val;
        break;
    case TASK_COLOR:
        task_flags->task_color = (char *)val;
        break;
    case TASK_SEQUENCE:
        task_flags->task_sequence = (Quark_Sequence *)val;
        break;
    case TASK_THREAD_COUNT:
        task_flags->task_thread_count = (int)val;
        break;
    case THREAD_SET_TO_MANUAL_SCHEDULING:
        task_flags->thread_set_to_manual_scheduling = (int)val;
        break;
    }
    return task_flags;
}

/* **************************************************************************** */
/**
 * Get the value of various task level flags.  Each returned value can
 * be either an integer or a pointer (intptr type).
 *
 *          Select from one of the flags:
 *          TASK_PRIORITY : an integer (0-MAX_INT)
 *          TASK_LOCK_TO_THREAD : an integer for the thread number
 *          TASK_LOCK_TO_THREAD_MASK : a pointer to a bitmask where task can run
 *          TASK_LABEL : a string pointer (NULL terminated) for the label
 *          TASK_COLOR :  a string pointer (NULL terminated) for the color.
 *          TASK_SEQUENCE : pointer to a Quark_Sequence structure
 *          THREAD_SET_TO_MANUAL_SCHEDULING: boolean integer {0,1} setting thread to manual (1) or automatic (0) scheduling
 *
 * @param[in] quark
 *          Pointer to the scheduler data structure
 * @param[in] flag
 *          One of the flags shown above.
 * @return Intptr type giving the value of the flag; -9 on error
 * @ingroup QUARK
 */
intptr_t QUARK_Task_Flag_Get( Quark* quark, int flag )
{
    Task *task = quark->worker[QUARK_Thread_Rank(quark)]->current_task_ptr;
    switch (flag)  {
    case TASK_PRIORITY:
        return (intptr_t)task->priority;
        break;
    case TASK_LOCK_TO_THREAD:
        return (intptr_t)task->lock_to_thread;
        break;
    case TASK_LOCK_TO_THREAD_MASK:
        return (intptr_t)task->lock_to_thread_mask;
        break;
    case TASK_LABEL:
        return (intptr_t)task->task_label;
        break;
    case TASK_COLOR:
        return (intptr_t)task->task_color;
        break;
    case TASK_SEQUENCE:
        return (intptr_t)task->sequence;
        break;
    case TASK_THREAD_COUNT:
        return (intptr_t)task->task_thread_count;
        break;
    case THREAD_SET_TO_MANUAL_SCHEDULING:
        return (intptr_t)task->thread_set_to_manual_scheduling;
        break;
    default:
        return -9;
        break;
    }
}

/* **************************************************************************** */
/**
 * Enable and disable DAG generation in QUARK.  Only to be called at
 * the task insertion level by the master thread.  Can be called
 * multiple times to enable and disable DAG generation during the
 * runtime.  For the output to make sense, this MUST be preceeded by a
 * sync operation such as QUARK_Barrier.
 *
 * @param[in,out] quark
 *          Pointer to the scheduler data structure
 * @param[in] enable
 *          Integer: 1 = enable DAG generation; otherwise disable
 * @ingroup QUARK
 */
void QUARK_DOT_DAG_Enable( Quark *quark, int enable )
{
    int i;
    if ( enable==1 ) {
        if ( !quark->dot_dag_was_setup ) {
            quark->high_water_mark = (int)(INT_MAX - 1);
            quark->low_water_mark = (int)(quark->high_water_mark);
            /* global FILE variable */
            if ( dot_dag_file == NULL ) fopen( &dot_dag_file, DOT_DAG_FILENAME, "w" ); 
            else fopen( &dot_dag_file, DOT_DAG_FILENAME, "a" );
            fprintf(dot_dag_file, "digraph G { size=\"10,7.5\"; center=1; orientation=portrait; \n");
            pthread_mutex_init( &quark->dot_dag_mutex, NULL );
            fprintf(dot_dag_file, "%d [style=\"invis\"]\n", 0);
            /* Reset tasklevel information */
            for (i=0; i<tasklevel_width_max_level; i++ )
                quark->tasklevel_width[i] = 0;
            /* Reset the address set nodes information */    
            int tmpint;
            icl_entry_t* tmpent;
            void *kp, *dp;
            icl_hash_foreach(quark->address_set, tmpint, tmpent, kp, dp) {
                Address_Set_Node *address_set_node = (Address_Set_Node *)dp;
                address_set_node->last_writer_taskid = 0;
                address_set_node->last_writer_tasklevel = 0;
                address_set_node->last_reader_or_writer_taskid = 0;
                address_set_node->last_reader_or_writer_tasklevel = 0;
            }
            /* quark->dot_dag_was_setup is used to indicate that the
             * dot_dag_file needs to be finalized */
            quark->dot_dag_was_setup = 1;
            quark->dot_dag_enable = 1;
        }
    } else {
        if ( quark->dot_dag_was_setup ) {
            for (i=1; i<tasklevel_width_max_level && quark->tasklevel_width[i]!=0; i++ ) {
                fprintf(dot_dag_file, "%d [label=\"%d:%d\"]\n", i, i, quark->tasklevel_width[i] );
                fprintf(dot_dag_file, "%d->%d [style=\"invis\"];\n", i-1, i );
            }
            fprintf(dot_dag_file, "} // close graph\n");
            fprintf(dot_dag_file, "// ---------------------- \n");
            fprintf(dot_dag_file, "\n\n");
            fclose( dot_dag_file );
            pthread_mutex_destroy( &quark->dot_dag_mutex );
            quark->dot_dag_was_setup = 0;
        }
        quark->dot_dag_enable = 0;
    }
}

/* **************************************************************************** */
/**
 * Called internally, from the master thread.  Wait/work till the
 * address_set_node is empty, that is, it has no more dependencies
 * waiting to use it.  If this is called by something other than the
 * master thread, we may have problems.  Originally created for use is
 * via the QUARK_Execute_Task function.
 */
static void quark_address_set_node_wait( Quark *quark, Address_Set_Node *address_set_node )
{
    int this_asn_still_has_tasks = 1;
    int myrank = QUARK_Thread_Rank( quark );
    while ( this_asn_still_has_tasks ) {
        pthread_mutex_lock_wrap( &address_set_node->asn_mutex );
        if ( icl_list_first(address_set_node->waiting_deps) == NULL ) 
            this_asn_still_has_tasks = 0;
        pthread_mutex_unlock_wrap( &address_set_node->asn_mutex );
        if ( this_asn_still_has_tasks ) {
            quark_process_completed_tasks( quark );
            quark_work_main_loop( quark->worker[myrank] );
        }
    }
}
/* **************************************************************************** */
