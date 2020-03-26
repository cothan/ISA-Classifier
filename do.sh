#!/bin/bash

export PATH=$PATH:/home/dnguye69/x-tools/$1/bin 
echo $PATH
echo "==============================1"
make clean 
echo "==============================2"
./configure CC=$1-gcc --host=$1
echo "==============================3"
sed '4415d' Makefile > Makefile1
mv Makefile1 Makefile
echo "==============================4"
make -j 8
echo "==============================5"
cd src/ 
echo "==============================6"
mkdir -p /tmp/binary-samples/$2
echo "==============================7"
find . -maxdepth 1 -type f -exec test -x {} \; -exec cp {} /tmp/binary-samples/$2 \;
echo "==============================8"
echo "- $1" >> /tmp/binary-samples/README.md
