#!/bin/bash


rm registerCounts.txt

for m in {1..100}
do
    for n in {1..100}
    do
        echo -n "$m $n " >> registerCounts.txt

        cuobjdump --dump-elf -fun blockProduct matmul$m-$n.o | grep reg | tail -n 1 | sed -r 's:.*reg = ([0-9]+).*:\1:' >> registerCounts.txt
    done
done
