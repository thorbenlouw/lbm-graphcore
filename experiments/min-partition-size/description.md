We've been using 6x6 as a good size for minRowsPerTile x minColsPerTile.
But what happens if we go smaller (allow 1x1) or 

So we try with min col sizes 1 2 6 16 32 64 128 
and min row sizes 1 2 3 6 12 24 36 48 72
and ipus 1 2 4 8 16
for stencil with problem size 2200x1122 * 1000 iterations

```
rm output.txt
for ipu in 1 2 4 8 16; do
  for minRow in 72 48 36 24 12 6 3 2 1; do
    for minCol in 128 64 32 16 8 4 2 1; do
        echo "$minRow,$minCol,$ipu" >> output.txt
        ./low_level_stencil -n 1000 -i ../../images/cheese.png -o /tmp/cheese-blurred.png --num-ipus $ipu --min-rows-per-tile $minRow  --min-cols-per-tile $minCol >> output.txt 2>&1
    done
  done 
done
```

We got to
3,4,4


for ipu in  8 16; do
  for minRow in 72 48 36 24  ; do
    for minCol in 128 64 32 ; do
        echo "$minRow,$minCol,$ipu" >> output.txt
        ./low_level_stencil -n 1000 -i ../../images/cheese.png -o /tmp/cheese-blurred.png --num-ipus $ipu --min-rows-per-tile $minRow  --min-cols-per-tile $minCol >> output.txt 2>&1
    done
  done 
done


for ipu in  8 16; do
  for minRow in 72 48 36 24  ; do
    for minCol in 16 8 4 2 1 ; do
        echo "$minRow,$minCol,$ipu" >> output.txt
        ./low_level_stencil -n 1000 -i ../../images/cheese.png -o /tmp/cheese-blurred.png --num-ipus $ipu --min-rows-per-tile $minRow  --min-cols-per-tile $minCol >> output.txt 2>&1
    done
  done 
done


for ipu in  8 16; do
  for minRow in 12 6 3 2 1  ; do
    for minCol in 128 64 32 ; do
        echo "$minRow,$minCol,$ipu" >> output.txt
        ./low_level_stencil -n 1000 -i ../../images/cheese.png -o /tmp/cheese-blurred.png --num-ipus $ipu --min-rows-per-tile $minRow  --min-cols-per-tile $minCol >> output.txt 2>&1
    done
  done 
done

for ipu in  8 16; do
  for minRow in 12 6 3 2 1  ; do
    for minCol in 16 8 4 2 1 ; do
        echo "$minRow,$minCol,$ipu" >> output.txt
        ./low_level_stencil -n 1000 -i ../../images/cheese.png -o /tmp/cheese-blurred.png --num-ipus $ipu --min-rows-per-tile $minRow  --min-cols-per-tile $minCol >> output.txt 2>&1
    done
  done 
done