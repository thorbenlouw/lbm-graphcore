rm -f effect-of-mem-on-conv.csv
for image in bricks leaf cheese; do
  for memchunks in "0.1 0.2" "0.3 0.4" "0.5 0.6" "0.7 0.8" "0.9"; do
    pids=""
    i=0
    for mem in $memchunks; do
      for datatype in half float; do
        for ipu in 1 2; do
          echo "$image, $datatype, $mem, $ipu"
          i=$((i+1))
          rm -f /tmp/${i}.csv
          echo "$image, $datatype, $mem, $ipu" >> /tmp/${i}.csv
          ./poplibs_stencil -n 100  -i ../../images/${image}.png -o /tmp/${image}-out${i}.png --num-ipus ${ipu} --data-type ${datatype} --conv-mem ${mem} >> /tmp/${i}.csv 2>&1  &
          pids[${i}]=$!
        done
      done
    done
    # wait for all pids
    for pid in ${pids[*]}; do
        wait $pid
    done
    for i in 1 2 3 4 5 6 7 8; do
      if [ -e /tmp/${i}.csv ]; then
        cat /tmp/${i}.csv >> effect-of-mem-on-conv.csv && rm /tmp/${i}.csv
      fi
    done
  done
done


# And then afterwards, you can
cat effect-of-mem-on-conv.csv | grep ",\|iter" | sed 's/                                  Running stencil iterations took/,/' | sed 's/s$//' | sed '$!N;s/\n/ /' | tr -s ' ' | sed 's/ //' > tmp.csv
