
set terminal pdf
set title ""
set autoscale
#set yrange [1:10000]
#set nokey
#set xrange [0:10]
set logscale y
set ylabel "residual" 
set xlabel "iters"
set output "conv.pdf"
set datafile missing "?"
set style line 11 lt 1 lw 2 pt 1
set style line 12 lt 1 lw 2 pt 6
set style line 21 lt 2 lw 2 pt 1
set style line 22 lt 2 lw 2 pt 6
set style line 31 lt 3 lw 2 pt 1
set style line 32 lt 3 lw 2 pt 6
set style line 41 lt 4 lw 2 pt 1
set style line 42 lt 4 lw 2 pt 6
set style line 51 lt 5 lw 2 pt 1
set style line 52 lt 5 lw 2 pt 6
set style line 61 lt 6 lw 2 pt 1
set style line 62 lt 6 lw 2 pt 6
set style line 71 lt 7 lw 2 pt 1
set style line 72 lt 7 lw 2 pt 6
set style line 81 lt 8 lw 2 pt 1
set style line 82 lt 8 lw 2 pt 6
set style line 91 lt 9 lw 2 pt 1
set style line 92 lt 9 lw 2 pt 6
plot  \
      "data1" using 1:3  title "data1" with l ls 11 lw 2,\
	  "data2" using 1:3  title "data2" with l ls 21 lw 2,\
 	  "data3" using 1:3  title "data3" with l ls 31 lw 2,\
 	  "data4" using 1:3  title "data4" with l ls 41 lw 2,\
	  "data5" using 1:3  title "data5" with l ls 51 lw 2	


