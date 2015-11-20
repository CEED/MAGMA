Conversion from MAGMA v1 to v2.

In src files:

0)  Run testers to verify that the code is working before changes.

        cd testing
        ./run_tests.py -s -m testing_*gehrd

    It's possible that something failed trying to make things backwards
    compatibile. Of course, this should be resolved before changing the code.

    The testers now run on a non-NULL stream, by default. This breaks any code
    that assumed extra synchronization from the NULL stream. To achieve the old behavior, use --null-stream:

        ./testing_dgeqrf_gpu --range 100:300:100 -c --version 3
        
        %   M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)    |b - A*x|
        %===============================================================
          100   100     ---   (  ---  )      1.44 (   0.00)     5.38e-18   ok
          200   200     ---   (  ---  )      3.54 (   0.00)     7.33e-05   failed
          300   300     ---   (  ---  )      5.33 (   0.01)     3.35e-05   failed
        
        
        ./testing_dgeqrf_gpu --range 100:300:100 -c --version 3 --null-stream
        
        %   M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)    |b - A*x|
        %===============================================================
          100   100     ---   (  ---  )      1.45 (   0.00)     5.38e-18   ok
          200   200     ---   (  ---  )      3.29 (   0.00)     3.40e-18   ok
          300   300     ---   (  ---  )      5.01 (   0.01)     2.58e-18   ok


1)  Automated partial conversion with scripts/convert_v2_src.pl.

    cd src
    ../scripts/convert_v2_src.pl zgehrd.cpp
    ../scripts/convert_v2_src.pl zlahr2.cpp
    ../scripts/convert_v2_src.pl zlahru.cpp

    This should do much of the tedious search and replace, but that's all it
    does -- it doesn't parse the code, so it can easily make mistakes.

    - Renames "stream[...]" to "queues[...]" (plural).

    - Replaces the #include header
      from common_magma.h   (which includes magma.h)
      to   magma_internal.h (which includes magma_v2.h)

    - Adds device to magma_queue_create.
      This may add redundant getdevice calls that need to be removed.
      If you already know the device (e.g., in a multi-GPU code that loops
      over the devices), you can remove the getdevice call entirely.

    - Comments out Set/GetKernelStream.
      You should then delete these, but you may need them to verify that it
      picked the correct queue for subsequent magmablas calls.

    - Add queue argument to magmablas calls.
      If the code called magmablasSetKernelStream( queue ), the script *assumes*
      that is the queue to use. Otherwise, it will put in UNKNOWN.

2)  Manually cleanup and verify code.

    - Remove any redundant or commented out code from step (1).

    - Create & destroy any queues that may be needed.
      For auxiliary routines such as larfb, lahr2, lahru, latrd, add queue(s)
      as an argument so it isn't created & destroyed for each panel.

      For Hessenberg, I create a queue in zgehrd, and pass that into zlahr2 and
      zlahru.

    - Run testers to verify that it still works. Here is the small (-s) and
      medium (-m) tests for all 4 precisions.

        cd testing
        ./run_tests.py -s -m testing_*gehrd
        
        [ lots of output ]
        
        ****************************************************************************************************
        summary
        ****************************************************************************************************
        all 472 tests in 8 commands passed!

3) Commit new code.
