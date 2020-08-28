rm "tmp.txt"
for dataType in half4 float2 float; do
  echo $dataType,1, >> tmp.txt
  ./low_level_stencil -n 100 -i /tmp/500x1000.png -o /tmp/out.png --num-ipus 1 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 >> tmp.txt 2>&1
  echo $dataType,2, >> tmp.txt
  ./low_level_stencil -n 100 -i /tmp/1000x1000.png -o /tmp/out.png --num-ipus 2 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 >> tmp.txt 2>&1
  echo $dataType,4, >> tmp.txt
  ./low_level_stencil -n 100 -i /tmp/2000x1000.png -o /tmp/out.png --num-ipus 4 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 >> tmp.txt 2>&1
  echo $dataType,8, >> tmp.txt
  ./low_level_stencil -n 100 -i /tmp/4000x1000.png -o /tmp/out.png --num-ipus 8 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 >> tmp.txt 2>&1
  echo $dataType,16, >> tmp.txt
  ./low_level_stencil -n 100 -i /tmp/8000x1000.png -o /tmp/out.png --num-ipus 16 --data-type ${dataType} --min-rows-per-tile 6 --min-cols-per-tile 6 >> tmp.txt 2>&1
done;


# And then afterwards, you can
echo "datatype,numIpus,time" > weak-scaling-results.csv
cat tmp.txt | grep ",\|IPU timing" | sed 's/Average IPU timing for program is://' | sed '$!N;s/\n/ /' | tr -s ' ' | sed 's/ //g' >> weak-scaling-results.csv




