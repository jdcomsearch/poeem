#!/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

g++  -std=c++11 -mavx2 -shared cpp/$1.cc -o bin/$1.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O3 -fopenmp -DNDEBUG
