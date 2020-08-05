To evaluate our grid partitioning algorithm, we want to see 
(1) how many tiles and workers are wasted
(2) the load balance achieved
(3) the max speedup possible

when we run the partitioning for grids of various sizes


./tile_mapping_stats --min-width 128 --max-width 4000 --min-height 128 --max-height 4000 --num-ipus 1 -n 20000 > /tmp/mapping-efficiency.csv
./tile_mapping_stats --min-width 128 --max-width 20000 --min-height 128 --max-height 20000 --num-ipus 16  > /tmp/mapping-efficiency16.csv

