/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/

#ifndef MAGMA_THREAD_HPP
#define MAGMA_THREAD_HPP

#include <queue>

#include "magma_internal.h"


/******************************************************************************/
extern "C"
void* magma_thread_main( void* arg );


/***************************************************************************//**
    Super class for tasks used with \ref magma_thread_queue.
    Each task should sub-class this and implement the run() method.
    @ingroup magma_thread
*******************************************************************************/
class magma_task
{
public:
    magma_task() {}
    virtual ~magma_task() {}
    
    virtual void run() = 0;  // pure virtual function to execute task
};


/******************************************************************************/
class magma_thread_queue
{
public:
    magma_thread_queue();
    ~magma_thread_queue();
    
    void launch( magma_int_t in_nthread );
    void push_task( magma_task* task );
    void sync();
    void quit();
    
protected:
    friend void* magma_thread_main( void* arg );
    magma_task* pop_task();
    void task_done();
    
    magma_int_t get_thread_index( pthread_t thread ) const;
    
private:
    std::queue< magma_task* > q;  ///<  queue of tasks
    bool            quit_flag;    ///<  quit() sets this to true; after this, pop returns NULL
    magma_int_t     ntask;        ///<  number of unfinished tasks (in queue or currently executing)
    pthread_mutex_t mutex;        ///<  mutex lock for queue, quit, ntask
    pthread_cond_t  cond;         ///<  condition variable for changes to queue and quit (see push, pop, quit)
    pthread_cond_t  cond_ntask;   ///<  condition variable for changes to ntask (see sync, task_done)
    pthread_t*      threads;      ///<  array of threads
    magma_int_t     nthread;      ///<  number of threads
};

#endif        //  #ifndef MAGMA_THREAD_HPP
