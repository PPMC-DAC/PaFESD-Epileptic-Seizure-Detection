.PHONY: all clean

CXX = g++-12
CXXFLAGS = -O3 -std=c++20 
LDFLAGS = -ltbb
LIB_DIR = EEGLib
INCLUDES = -I ../common/ -I $(LIB_DIR)
SOURCES = main.cpp $(LIB_DIR)/EEG_lib.cpp 
OBJ_FILES = $(SOURCES:.cpp=.o)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CXXFLAGS += -mavx512f
endif

all: main

main: $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp $(LIB_DIR)/EEG_lib.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@ $(INCLUDES)

$(LIB_DIR)/EEG_lib.o: $(LIB_DIR)/EEG_lib.cpp $(LIB_DIR)/EEG_lib.hpp
	$(CXX) $(CXXFLAGS) -c $(LIB_DIR)/EEG_lib.cpp -o $@ $(INCLUDES)

clean:
	rm -f main $(OBJ_FILES)
