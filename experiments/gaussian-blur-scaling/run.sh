rm -f gaussian-blur-scaling.csv
for image in bricks leaf cheese; do
  for datatype in half4 float2 float; do
    for ipu in 1 2 4 8 16; do
      echo "$image, $datatype, $ipu"
      echo "$image, $datatype, $ipu" >> gaussian-blur-scaling.csv
      ./low_level_stencil -n 100 -i ../../images/${image}.png -o /tmp/${image}-out.png --num-ipus ${ipu} --data-type ${datatype} --min-rows-per-tile 1 --min-cols-per-tile 1 >> gaussian-blur-scaling.csv 2>&1
    done
  done
done

# And then afterwards, you can
echo "image,datatype,numIpus,time" > tmp.csv
cat gaussian-blur-scaling.csv | grep ",\|IPU timing" | sed 's/Average IPU timing for program is:/,/' | sed '$!N;s/\n/ /' | tr -s ' ' | sed 's/ //g' >> tmp.csv
