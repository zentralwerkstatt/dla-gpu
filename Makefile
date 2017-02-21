CC         = /Developer/NVIDIA/CUDA-7.5/bin/nvcc
FLAGS     = -arch=sm_30 -std=c++11
EXECUTABLE = dla-gpu

SOURCES    = dla-gpu.cu
OBJECTS    = $(SOURCES:.cpp=.o)

all: $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) $(FLAGS) -o $@

clean:
	rm -f *.o dla-gpu