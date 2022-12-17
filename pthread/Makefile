CXX = g++
CXXFLAG = -O3
CXXLIBS = -lpthread

EXE = conv.out
CPP = conv.cpp

OBJ = ${CPP:.cpp=.o}

all: ${EXE}

${EXE}: ${OBJ}
		${CXX} $^ ${CXXFLAG} ${CXXLIBS} -o $@

%.o: %.cpp
	${CXX} -c $< ${CXXFLAG} ${CXXLIBS} -o $@

.PHONY: clean

clean:                             
	rm -rf ${EXE} ${OBJ}