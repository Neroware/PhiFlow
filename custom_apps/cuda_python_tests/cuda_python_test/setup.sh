nvcc -c testlib.cu -o testlib.o -Xcompiler -fPIC -I /usr/include/python3.6m -l python3.6m
g++ -shared -fPIC -o testlib.so testlib.o -L /usr/local/cuda/lib64/libcudart.so
