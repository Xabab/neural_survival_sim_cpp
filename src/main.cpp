#include <iostream>
#include <eigen3/Eigen/Dense>
#include "FeedForwardNeuralNetwork.h"

using namespace std;
using namespace Eigen;

float inc(float test){
	return test + 1;
}

int main() {
	vector < VectorXf, Eigen::aligned_allocator<VectorXf> > _neuronLayers;
	vector < VectorXf, Eigen::aligned_allocator<VectorXf> > _biases;
	vector < MatrixXf, Eigen::aligned_allocator<MatrixXf> > _weights;
	
	
	
	_neuronLayers.emplace_back(VectorXf(2));
	_neuronLayers[0] << 1, 0;
	_neuronLayers.emplace_back(VectorXf(2));
	_neuronLayers.emplace_back(VectorXf(1));
	
	_biases.emplace_back(VectorXf::Zero(2));
	_biases.emplace_back(VectorXf::Zero(1));
	
	_weights.emplace_back(MatrixXf(2, 2));
	_weights[0] << 0.45, -0.12,
			       0.78,  0.13;
	
	_weights.emplace_back(MatrixXf(1, 2));
	_weights[1] << 1.5, -2.3;
	
	FeedForwardNeuralNetwork f = FeedForwardNeuralNetwork(_neuronLayers, _biases, _weights, ActivationFunctions::sigm);
	
	
	cout << f.getOutputDebug()[0] << endl;
	
	
	
	
}

