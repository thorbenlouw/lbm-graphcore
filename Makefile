all: play lbm

play: play.cpp
	g++ -O3 -std=c++17 -Wall $^ -lpoplar -lpopops -lpoputil  -o $@


great-sadness: UnusedReduceFunctionCausesErr.cpp
	g++ -O3 -std=c++17 -Wall $^ -lpoplar -lpopops -lpoputil  -o $@

lbm: lbm.cpp
	g++ -O3 -std=c++17 -Wall $^ -lpoplar -lpopops -lpoputil  -o $@

lbm-debug: lbm.cpp
	g++ -O3 -std=c++17 -Wall $^ -g -lpoplar -lpopops -lpoputil  -o $@
