/**
 * @file quark_trace.h
 *
 * QUARK (QUeuing And Runtime for Kernels) provides a runtime
 * enviornment for the dynamic execution of precedence-constrained
 * tasks.
 *
 * QUARK is a software package provided by Univ. of Tennessee,
 * Univ. of California Berkeley and Univ. of Colorado Denver.
 *
 * @version 2.5.0
 * @author Mathieu Faverge (2010-11-15)
 * @author Vijay Joshi (2013-05-15)
 * @author Asim YarKhan (2013-05-15)
 * @date 2010-11-15
 *
 */
#ifndef _QUARK_TRACE_H_
#define _QUARK_TRACE_H_

enum quark_ev_code_e {
    QUARK_STOP,
    QUARK_TASK,
    QUARK_TASKW,
    QUARK_INSERT_TASK,
    QUARK_INSERT_TASK_PACKED,
    QUARK_PROCESS_COMPLETED_TASKS,
    QUARK_NBMAX_EVENTS,
};

#ifdef TRACE_QUARK

#include <eztrace.h>
#include <ev_codes.h>

#define QUARK_EVENTS_ID    0x0020
#define QUARK_MASK_EVENTS  0x0fff
#define QUARK_PREFIX       (QUARK_EVENTS_ID << NB_BITS_EVENTS)
#define FUT_QUARK(event)   (QUARK_PREFIX | QUARK_##event)

#define quark_trace_addtask() EZTRACE_EVENT1(FUT_QUARK(TASK),  1)
#define quark_trace_deltask() EZTRACE_EVENT1(FUT_QUARK(TASK), -1)
#define quark_trace_addtask2worker(__tid) EZTRACE_EVENT2(FUT_QUARK(TASKW), __tid,  1)
#define quark_trace_deltask2worker(__tid) EZTRACE_EVENT2(FUT_QUARK(TASKW), __tid, -1)

#define quark_trace_event_start(ev_code) EZTRACE_EVENT0(FUT_QUARK(ev_code));
#define quark_trace_event_end() EZTRACE_EVENT0(FUT_QUARK(STOP));

#else

#define quark_trace_addtask()
#define quark_trace_deltask()
#define quark_trace_addtask2worker(__tid)
#define quark_trace_deltask2worker(__tid)

#define quark_trace_event_start(ev_code) 
#define quark_trace_event_end()

#endif

#endif /* _QUARK_TRACE_H_ */
