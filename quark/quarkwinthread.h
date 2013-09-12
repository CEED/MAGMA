/**
 *
 * @file quarkwinthread.h
 *
 *  This file handles the mapping from pthreads calls to windows threads
 *  QUARK is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver
 *
 * @version 2.5.0
 * @author Piotr Luszczek
 * @date 2010-11-15
 *
 * Note : this file is a copy of a PLASMA file for use of QUARK in a
 * standalone library updated by Asim YarKhan
 **/
#ifndef PLASMWINTHREAD_H
#define PLASMWINTHREAD_H

#include <windows.h>

/*
typedef struct pthread_s {
  HANDLE Hth;
  unsigned IDth;
  void *(*Fth) (void *);
} pthread_t;
*/
typedef struct pthread_s {
  HANDLE hThread;
  unsigned int uThId;
} pthread_t;

typedef HANDLE pthread_mutex_t;
typedef int pthread_mutexattr_t;
typedef int pthread_attr_t;
typedef int pthread_condattr_t;

typedef struct pthread_cond_s {
  HANDLE hSem;
  HANDLE hEvt;
  CRITICAL_SECTION cs;
  int waitCount; /* waiting thread counter */
} pthread_cond_t;

typedef int pthread_attr_t;

#define PTHREAD_MUTEX_INITIALIZER ((pthread_mutex_t) -1)

#define PTHREAD_SCOPE_SYSTEM 1

#define QUARK_DLLPORT
#define QUARK_CDECL __cdecl

QUARK_DLLPORT pthread_t QUARK_CDECL pthread_self(void);
QUARK_DLLPORT int QUARK_CDECL pthread_mutex_init(pthread_mutex_t *mutex, const pthread_mutexattr_t * attr);
QUARK_DLLPORT int QUARK_CDECL pthread_mutex_destroy(pthread_mutex_t *mutex);
QUARK_DLLPORT int QUARK_CDECL pthread_mutex_lock(pthread_mutex_t *mutex);
QUARK_DLLPORT int QUARK_CDECL pthread_mutex_trylock(pthread_mutex_t *mutex);
QUARK_DLLPORT int QUARK_CDECL pthread_mutex_unlock(pthread_mutex_t *mutex);
QUARK_DLLPORT int QUARK_CDECL pthread_attr_init(pthread_attr_t *attr);
QUARK_DLLPORT int QUARK_CDECL pthread_attr_destroy(pthread_attr_t *attr);
QUARK_DLLPORT int QUARK_CDECL pthread_attr_setscope(pthread_attr_t *attr, int scope);
QUARK_DLLPORT int QUARK_CDECL pthread_create(pthread_t *tid, const pthread_attr_t *attr, void *(*start) (void *), void *arg);
QUARK_DLLPORT int QUARK_CDECL pthread_cond_init(pthread_cond_t *cond, const pthread_condattr_t *attr);
QUARK_DLLPORT int QUARK_CDECL pthread_cond_destroy(pthread_cond_t *cond);
QUARK_DLLPORT int QUARK_CDECL pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex);
QUARK_DLLPORT int QUARK_CDECL pthread_cond_broadcast(pthread_cond_t *cond);
QUARK_DLLPORT int QUARK_CDECL pthread_join(pthread_t thread, void **value_ptr);
QUARK_DLLPORT int QUARK_CDECL pthread_equal(pthread_t thread1, pthread_t thread2);

QUARK_DLLPORT int QUARK_CDECL pthread_setconcurrency (int);

QUARK_DLLPORT unsigned int QUARK_CDECL pthread_self_id(void);

#endif
