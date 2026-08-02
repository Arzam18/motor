CXX ?= clang++
CXXFLAGS += -std=c++20 -O3 -Wunused -Wall -Wextra -DNDEBUG -march=native

ifeq ($(OS), Windows_NT)
    EXE ?= Motor.exe
    ifeq ($(findstring g++,$(CXX)),)
        CXXFLAGS += -Wl,/STACK:16777216
    else
        CXXFLAGS += -Wl,--stack,16777216
    endif
else
    EXE ?= motor
    CXXFLAGS += -lm
endif

CLANG_PLUS_PLUS_18 := $(shell command -v clang++-18 2>/dev/null)
ifneq ($(strip $(CLANG_PLUS_PLUS_18)),)
    CXX = clang++-18
endif

all:
	$(CXX) $(CXXFLAGS) main.cpp -o $(EXE)

clean:
	rm -f $(EXE) Motor.exe
