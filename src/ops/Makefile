objects = bin/clustering_op.so bin/clustering_raw_op.so bin/search_op.so bin/tokenizer_op.so bin/knn_dataset_op.so bin/rec_dataset_op.so bin/combinatorial_op.so
tests = search_op.test tokenizer_op.test
perfs = clustering_op.perf # search_op.perf

all: $(objects)

$(objects): bin/%_op.so: cpp/%_op.cc cpp/vector_util.h
	./compile.sh $*_op

$(tests): %.test: test/test_%.py bin/%.so
	python $<

indexing.test: test/test_indexing.py bin/clustering_raw_op.so
	python $<

vector_util.test: test/test_vector_util.cc cpp/vector_util.h
	g++ -std=c++11 -mavx2 test/test_vector_util.cc -o bin/test_vector_util.so && bin/test_vector_util.so

unit_tests: vector_util.test

test: unit_tests $(tests) indexing.test

$(perfs): %.perf: perf/perf_%.py bin/%.so
	python $<

perf: $(perfs)

clean:
	rm -rf bin/*.so
