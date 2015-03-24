build:
	g++ -O3 -g -Wall -o binary_decision_tree binary_decision_tree.cpp --std=c++11

clean:
	rm -f binary_decision_tree *~
