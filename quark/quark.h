/**
 *
 * @file quark.h
 *
 * Dynamic scheduler functions
 *
 * PLASMA is a software package provided by Univ. of Tennessee,
 * Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.5.0
 * @author Asim YarKhan
 * @date 2010-11-15
 *
 **/

#ifndef QUARK_H
#define QUARK_H

#include <limits.h>
#include <stdio.h>

#if defined( _WIN32 )
  /* This must be included before INPUT is defined below, otherwise we
     have a name clash/problem  */
  #include <windows.h>
  #include <limits.h>
#else
  #include <inttypes.h>
#endif

#include "quark_unpack_args.h"

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

#ifdef DBGQUARK
/* #define DBGPRINTF(str, ...) { fprintf(stderr, "%s:%d: [%s] " str, __FILE__, __LINE__, __FUNCTION__, __VA_ARGS__); } */
#define DBGPRINTF(...) { fprintf(stderr, __VA_ARGS__); }
#else
#define DBGPRINTF(...) if (0) {};
#endif

#define QUARK_SUCCESS 0
#define QUARK_ERR -1
#define QUARK_ERR_UNEXPECTED -1
#define QUARK_ERR_NOT_SUPPORTED -2

/* A bitmask of 8 bits to to hold region markers  */
#define QUARK_REGION_BITMASK 0x0000FF
#define QUARK_REGION_ALL 0x0FF
typedef enum { QUARK_REGION_0=1<<0, QUARK_REGION_1=1<<1, QUARK_REGION_2=1<<2, QUARK_REGION_3=1<<3,
               QUARK_REGION_4=1<<4, QUARK_REGION_5=1<<5, QUARK_REGION_6=1<<6, QUARK_REGION_7=1<<7 } quark_data_region_t;
typedef enum { QUARK_REGION_L=QUARK_REGION_0|QUARK_REGION_1|QUARK_REGION_2,
               QUARK_REGION_D=QUARK_REGION_3|QUARK_REGION_4,
               QUARK_REGION_U=QUARK_REGION_5|QUARK_REGION_6|QUARK_REGION_7 } quark_ldu_region_t;

/* Data items can be: */
/*  INPUT, OUTPUT, INOUT:  these data items create dependencies */
/*  VALUE:  these data items get copied over */
/*  NODEP:  these data items get copied over, and are not used for dependencies */
/*  SCRATCH:  these data items can be allocated (and deallocted) by the scheduler when tasks execute  */
#define QUARK_DIRECTION_BITMASK 0x000F00
typedef enum { QINPUT=0x100, OUTPUT=0x200, INOUT=0x300, VALUE=0x400, NODEP=0x500, SCRATCH=0x600} quark_direction_t;
#define INPUT 0x100

#define QUARK_VALUE_FLAGS_BITMASK 0xFFF000

/* Data locality flag; ie keep data on the same core if possible */
#define LOCALITY ( 1 << 12 )
#define NOLOCALITY 0x00

/* A data address with a sequence of ACCUMULATOR dependencies will allow the related tasks to be reordered  */
#define ACCUMULATOR ( 1 << 13 )
#define NOACCUMULATOR 0x00

/* A data address with a sequence of GATHERV dependencies will allow the related tasks to be run in parallel  */
#define GATHERV ( 1 << 14 )
#define NOGATHERV 0x00

/* The following are task level flags, that can be either provided as additional arguments to the task, or via SET functions */
/* The task label; should be provided as a null terminated string */
#define TASK_LABEL ( 1 << 15 )
#define TASKLABEL TASK_LABEL    /* depreciated label */
/* The task color; should be provided as a null terminated string */
#define TASK_COLOR ( 1 << 16 )
#define TASKCOLOR TASK_COLOR    /* depreciated label */
/* The priority of the task, provided as an integer */
#define TASK_PRIORITY ( 1 << 17 )
/* Lock the task to a specific thread number (0 ... NTHREADS-1), provided as an integer */
#define TASK_LOCK_TO_THREAD ( 1 << 18 )
/* The sequence pointer to be associated with the task, provided as a pointer */
#define TASK_SEQUENCE ( 1 << 19 )
/* An integere for the number of threads require */
#define TASK_THREAD_COUNT ( 1 << 20 )
/* The thread that runs this task should have manual scheduling enabled (1) or disabled (0) */
#define THREAD_SET_TO_MANUAL_SCHEDULING ( 1 << 21 )
/* Lock the task to a thead mask (0 ... NTHREADS-1) bits long, provided as a character array (byte array) */
#define TASK_LOCK_TO_THREAD_MASK ( 1 << 22 )

/* The range for priority values */
#define QUARK_TASK_MIN_PRIORITY 0
#define QUARK_TASK_MAX_PRIORITY INT_MAX

/* Definition of structure holding scheduler information */
typedef struct quark_s Quark;

/* Structure holding task information */
typedef struct quark_task_s Quark_Task;

/* Create a type for setting task flags */
struct quark_task_flags_s {
    int task_priority;
    int task_lock_to_thread;
    char *task_color;
    char *task_label;
    void *task_sequence;
    int task_thread_count;
    int thread_set_to_manual_scheduling;
    unsigned char *task_lock_to_thread_mask;
};

typedef struct quark_task_flags_s Quark_Task_Flags;
/* Static initializer for Quark_Task_Flags_t */
#define Quark_Task_Flags_Initializer { (int)0, (int)-1, (char *)NULL, (char *)NULL, (void *)NULL, (int)1, (int)-1, (unsigned char *)NULL }

/* Setup scheduler data structures, assumes threads are managed seperately */
Quark *QUARK_Setup(int num_threads);

/* Setup scheduler data structures, spawn worker threads, start the workers working  */
Quark *QUARK_New(int num_threads);

/* Add a task, called by the master process (thread_rank 0)  */
unsigned long long QUARK_Insert_Task(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...);

/* Main work loop, called externally by everyone but the master
 * (master manages this internally to the insert_task and waitall
 * routines). Each worker thread can call work_main_loop( quark,
 * thread_rank), where thread rank is 1...NUMTHREADS ) */
void QUARK_Worker_Loop(Quark *quark, int thread_rank);

/* Finish work and return.  Workers do not exit */
void QUARK_Barrier(Quark * quark);

/* Just wait for current tasks to complete, the scheduler and
 * strutures remain as is... should allow for repeated use of the
 * scheduler.  The workers return from their loops.*/
void QUARK_Waitall(Quark * quark);

/* Delete scheduler, shutdown threads, finish everything, free structures */
void QUARK_Delete(Quark * quark);

/* Free scheduling data structures */
void QUARK_Free(Quark * quark);

/* Cancel a specific task */
int QUARK_Cancel_Task(Quark *quark, unsigned long long taskid);

/* Returns a pointer to the list of arguments, used when unpacking the
   arguments; Returna a pointer to icl_list_t, so icl_list.h will need
   bo included if you use this function */
void *QUARK_Args_List(Quark *quark);

/* Returns the rank of a thread in a parallel task */
int QUARK_Get_RankInTask(Quark *quark);

/* Return a pointer to an argument.  The variable last_arg should be
   NULL on the first call, then each subsequent call will used
   last_arg to get the the next argument. */
void *QUARK_Args_Pop( void *args_list, void **last_arg);

/* Utility function returning rank of the current thread */
int QUARK_Thread_Rank(Quark *quark);

/* Packed task interface */
/* Create a task data structure to hold arguments */
Quark_Task *QUARK_Task_Init(Quark * quark, void (*function) (Quark *), Quark_Task_Flags *task_flags );

/* Add (or pack) the arguments into a task data structure (make sure of the correct order) */
void QUARK_Task_Pack_Arg( Quark *quark, Quark_Task *task, int arg_size, void *arg_ptr, int arg_flags );

/* Insert the packed task data strucure into the scheduler for execution */
unsigned long long QUARK_Insert_Task_Packed(Quark * quark, Quark_Task *task );

/* Unsupported function for debugging purposes; execute task AT ONCE */
unsigned long long QUARK_Execute_Task_Packed( Quark *quark, Quark_Task *task );

/* Unsupported function for debugging purposes; execute task AT ONCE */
unsigned long long QUARK_Execute_Task( Quark *quark, void (*function) (Quark *), Quark_Task_Flags *task_flags, ...);

/* Get the label (if any) associated with the current task; used for printing and debugging  */
char *QUARK_Get_Task_Label(Quark *quark);

/* Method for setting task flags */
Quark_Task_Flags *QUARK_Task_Flag_Set( Quark_Task_Flags *flags, int flag, intptr_t val );

/* Type for task sequences */
typedef struct Quark_sequence_s Quark_Sequence;

/* Create a seqeuence structure, to hold sequences of tasks */
Quark_Sequence *QUARK_Sequence_Create( Quark *quark );

/* Called by worker, cancel any pending tasks, and mark sequence so that it does not accept any more tasks */
int QUARK_Sequence_Cancel( Quark *quark, Quark_Sequence *sequence );

/* Destroy a sequence structure, cancelling any pending tasks */
Quark_Sequence *QUARK_Sequence_Destroy( Quark *quark, Quark_Sequence *sequence );

/* Wait for a sequence of tasks to complete */
int QUARK_Sequence_Wait( Quark *quark, Quark_Sequence *sequence );

/* Get the sequence information associated the current task/worker, this was provided when the tasks was created */
Quark_Sequence *QUARK_Get_Sequence(Quark *quark);

/* Get the priority associated the current task/worker */
int QUARK_Get_Priority(Quark *quark);

/* Get information associated the current task and worker thread;
 * Callable from within a task, since it works on the currently
 * executing task */
intptr_t QUARK_Task_Flag_Get( Quark *quark, int flag );

/* Enable and disable DAG generation via API.  Only makes sense after
 * a sync, such as QUARK_Barrier. */
void QUARK_DOT_DAG_Enable( Quark *quark, int boolean_value );

/* Get the number_th bit in a bitset (unsigned char *); useful for QUARK_LOCK_TO_THREAD_MASK flag */
static inline int QUARK_Bit_Get(unsigned char *set, int number)
{
    set += number / 8;
    return (*set & (1 << (7-(number % 8)))) != 0; /* 0 or 1       */
}

/* Set the number_th bit in a bitset (unsigned char *) to value (0 or 1); useful for QUARK_LOCK_TO_THREAD_MASK flag */
static inline void QUARK_Bit_Set(unsigned char *set, int number, int value)
{
    set += number / 8;
    if (value)
        *set =  (unsigned char)(*set | (1 << (7-(number % 8)))); /* set bit      */
    else    *set = (unsigned char)(*set & ( ~(1 << (7-(number % 8))))); /* clear bit    */
}

/* The following are QUARK internal functions, and will be moved to a
 * separate header file in the future.  Please do not use these
 * functions outside of the QUARK source code.  */
void quark_warning(const char *func_name, char* msg_text);
void quark_topology_init();
void quark_topology_finalize();
int quark_setaffinity(int rank);
int quark_unsetaffinity();
int quark_yield();
int quark_get_numthreads();
int *quark_get_affthreads();
int quark_getenv_int(char* name, int defval);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif                          /* QUARK.H */
