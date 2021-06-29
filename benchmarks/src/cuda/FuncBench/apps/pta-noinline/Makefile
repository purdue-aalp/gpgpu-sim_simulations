TOPLEVEL := ../..
BIN	:= $(TOPLEVEL)/bin
APP=pta-noinline

CFLAGS= -DINLINE=__noinline__ -DINLINE2=__noinline__

pta: main.cu andersen.h andersen.cu
	nvcc $(CFLAGS) -gencode arch=compute_70,code=sm_70 main.cu -o pta-noinline

#pta: util.o main.o
#	nvcc -gencode arch=compute_70,code=sm_70 -dlto util.o main.o -o pta
#
#util.o: util.cu util.h andersen.h andersen.cu
#	nvcc -gencode arch=compute_70,code=lto_70 -dc util.cu -o util.o
#
#main.o: main.cu
#	nvcc -gencode arch=compute_70,code=lto_70 -dc main.cu -o main.o

# separate compilation
#pta: util.o main.o
#	nvcc $(CFLAGS) -gencode arch=compute_70,code=sm_70 util.o main.o -o pta
#
#util.o: util.cu util.h andersen.h andersen.cu
#	nvcc $(CFLAGS) -gencode arch=compute_70,code=sm_70 -dc util.cu -o util.o
#
#main.o: main.cu
#	nvcc $(CFLAGS) -gencode arch=compute_70,code=sm_70 -dc main.cu -o main.o

#pta: util.o main.o
#	nvcc --maxrregcount=62 -gencode arch=compute_70,code=sm_70 util.o main.o -o pta
#
#util.o: util.cu util.h andersen.h andersen.cu
#	nvcc --maxrregcount=62 -gencode arch=compute_70,code=sm_70 -dc util.cu -o util.o
#
#main.o: main.cu
#	nvcc --maxrregcount=62 -gencode arch=compute_70,code=sm_70 -dc main.cu -o main.o
variants: $(VARIANTS)

clean:
	rm -f *.o
	rm -f pta-noinline $(BIN)/$(APP)
