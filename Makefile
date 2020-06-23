all: play lbm

play: play.cpp
	g++ -O3 -std=c++17 -Wall $^ -lpoplar -lpopops -lpoputil  -o $@


lbm: main/lbm.cpp
	g++ -O3 -std=c++17 -Wall $^ -lpoplar -lpopops -lpoputil  -o $@

lbm-debug: main/lbm.cpp
	g++ -O3 -std=c++17 -Wall $^ -g -lpoplar -lpopops -lpoputil  -o $@