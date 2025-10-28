CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O3 -march=native
TARGET = example
OBJECTS = example.o Numc++.o

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

example.o: example.cpp Numc++.h
	$(CXX) $(CXXFLAGS) -c example.cpp

Numc++.o: Numc++.cpp Numc++.h
	$(CXX) $(CXXFLAGS) -c Numc++.cpp

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: clean

