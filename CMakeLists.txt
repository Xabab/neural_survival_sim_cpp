cmake_minimum_required(VERSION 3.14)
project(neural_survival_sim_cpp)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_CXX_FLAGS "-std=c++0x -O0") # -ffast-math")

add_executable(neural_survival_sim_cpp src/main.cpp src/hackmath.h src/Stopwatch.h src/FeedForwardNeuralNetwork.h)