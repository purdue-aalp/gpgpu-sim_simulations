TOPLEVEL := ../..
BIN	:= $(TOPLEVEL)/bin
APP=pta

#pta: main.cu
#	nvcc main.cu -o pta

pta: util.o main.o
	nvcc util.o main.o -o pta

util.o: util.cu util.h andersen.h andersen.cu
	nvcc -dc util.cu -o util.o

main.o: main.cu
	nvcc -dc main.cu -o main.o

variants: $(VARIANTS)

clean:
	rm -f *.o
	rm -f pta $(BIN)/$(APP)
