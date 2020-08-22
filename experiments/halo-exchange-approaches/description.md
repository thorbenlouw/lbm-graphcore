We run each of our 5 halo exchange approaches 

-d  -b 100 -n 10 -h <strategy> -m

So: block size 100x100 and we run 10x

And then we take the emulator output for cycles etc. Every vertex has the compute cycles as 100, so 
we know our compute isn't it - the rest is down to implmentation


e.g.
Execution:

  Total cycles:                                         178,569,036 (approx 111,605.6 microseconds)
  Tile average compute cycles (including idle threads): 207,965.0 (0.1% of total)
  Tile average compute cycles (excluding idle threads): 98,504.3 (0.1% of total)
  Tile average IPU exchange cycles:                     19,309.1 (0.0% of total)
  Tile average global exchange cycles:                  0.0 (0.0% of total)
  Tile average host exchange cycles:                    0.0 (0.0% of total)
  Tile average sync cycles:                             178,341,761.9 (99.9% of total)

  Cycles by vertex type:
    popops::Zero2d<float>              (14592 instances):   17,612,544
    poplar_rt::DstLongStridedCopy       (7298 instances):   11,521,852
    poplar_rt::DstStridedCopyDA32       (9726 instances):   11,268,126
    poplar_rt::ShortMemcpy             (12744 instances):      973,414
    Fill<float>                         (2432 instances):      243,200
    IncludedHalosApproach<float>        (2432 instances):      243,200
    poplar_rt::Memcpy64Bit              (1220 instances):      101,132

