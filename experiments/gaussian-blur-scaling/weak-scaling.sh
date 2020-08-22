for dataType in half4 float2 float; do
  echo $dataType
  ./low_level_stencil -n 100 -i /tmp/500x1000.png -o /tmp/out.png --num-ipus 1 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 2>&1
  ./low_level_stencil -n 100 -i /tmp/1000x1000.png -o /tmp/out.png --num-ipus 2 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 2>&1
  ./low_level_stencil -n 100 -i /tmp/2000x1000.png -o /tmp/out.png --num-ipus 4 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 2>&1
  ./low_level_stencil -n 100 -i /tmp/4000x1000.png -o /tmp/out.png --num-ipus 8 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 2>&1
  ./low_level_stencil -n 100 -i /tmp/8000x1000.png -o /tmp/out.png --num-ipus 16 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 2>&1
done;



