CXX = g++
BASE_CXXFLAGS = -O3 -march=armv8.2-a+fp16fml+dotprod -fopenmp -std=c++17
SECURITY_FLAGS = -fPIC
COVERAGE_FLAGS = -fprofile-arcs -ftest-coverage
CXXFLAGS = -I../include $(BASE_CXXFLAGS) $(SECURITY_FLAGS) $(COVERAGE_FLAGS)
LINK_OPTS = -Wl,-z,relro,-z,now --coverage
OPENMP_LIB = -fopenmp

KRL_SOURCES = $(wildcard ../src/*.c)
KRL_OBJS = $(notdir $(KRL_SOURCES:.c=.o))

sve_search_codes.o: ../src/sve_search_codes.c
	$(CXX) $(CXXFLAGS) -march=armv8.2-a+sve -c $< -o $@
%.o: ../src/%.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@
.PHONY: all
all: lut4b.x lut8b.x sparse_distance.x dense_distance.x handle_IO.x reorder.x negative_distance.x

lut4b.x: 1-lookup_4bit_table.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -o $@

lut8b.x: 2-lookup_8bit_table.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -o $@
sparse_distance.x: 3-sparse_distance.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -o $@

dense_distance.x: 4-dense_distance.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -L../bin -lopenblas -o $@
handle_IO.x: 5-handle_read_write.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -o $@

reorder.x: 6-reorder_test.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -o $@
negative_distance.x: 7-negative_distance.o $(KRL_OBJS)
	$(CXX) $^ $(LINK_OPTS) $(OPENMP_LIB) -o $@

.PHONY: clean
clean:
	rm -f *.x *.o *.gcno *.gcda