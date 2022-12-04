CXX = g++
CXXFLAG = -O3
CXXLIBS = -lpthread
CXXINCS = -I ../common

EXE = pthread
CPP = pthread.cpp

OBJ = ${CPP:.cpp=.o}

all: ${EXE}

${EXE}: ${OBJ}
		${CXX} $^ ${CXXFLAG} ${CXXLIBS} -o $@

%.o: %.cpp
	${CXX} -c $< ${CXXFLAG} ${CXXINCS} ${CXXLIBS} -o $@

.PHONY: clean

clean:                             
	rm -rf ${EXE} ${OBJ}