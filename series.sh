#!/bin/sh


for n in {1..64}
do
    for m in {1..64}
    do
        make perf N=$n M=$m PREFIX=/elxfs/ihpc/ihpc05 &
        while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
    done
done

wait
echo "all done"

for n in {1..64}
do
    for m in {1..64}
    do
        /elxfs/ihpc/ihpc05/perf$m-$n | tee --append diagonal.txt
    done
done
