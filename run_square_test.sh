#!/bin/sh

xrange=16
yrange=16

for (( x=1 ; x<$xrange; x++ ))
do
    for (( y=1 ; y<$yrange; y++ ))
    do
        make test  M=$x N=$y GENVER=$3 PREFIX=./build 2>&1 > /dev/null &
        while test $(jobs -p | wc -w) -ge 300; do sleep 1; done
    done
done



while test $(jobs -p | wc -w) -ge 2; do
    echo -en '\r'
    echo -n $(jobs -p | wc -w)
    sleep 1
done

#    sleep 1
#    completed=$xrange*$yrange-$(jobs -p | wc -w)

#    for (( k=1 ; k<$completed; k++ ))
#    do
 #       echo -n "."
  #  done
   # for (( k=$completed+1; k<($xrange*$yrange); k++ ))
    #do
 #       echo -n " "
 #   done
 #   echo -n "|"
#done

echo

wait
echo "all built"

for (( x=1 ; x<$xrange; x++ ))
do
    for (( y=1 ; y<$yrange; y++ ))
    do
        ./build/test$x-$y-$3
    done
done
