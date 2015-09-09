#!/bin/sh


NSTART=64
NEND=100
MSTART=64
MEND=100

for m in {$MSTART .. $MEND}
do
    for n in {$NSTART .. $NEND}
    do
        make perf N=$n M=$m PREFIX=/elxfs/ihpc/ihpc05 &
        while test $(jobs -p | wc -w) -ge 200; do sleep 1; done
    done
done

wait
echo "all done"

for m in {$MSTART .. $MEND}
do
    for n in {$NSTART .. $NEND}
    do
        /elxfs/ihpc/ihpc05/perf$m-$n | tee --append diagonal.txt
    done
done
