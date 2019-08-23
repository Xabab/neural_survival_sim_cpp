//
// Created by xabab on 14.08.19.
//

#ifndef NEURAL_SURVIVAL_SIM_CPP_FEEDFORWARDNEURALNETWORK_H
#define NEURAL_SURVIVAL_SIM_CPP_FEEDFORWARDNEURALNETWORK_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/StdVector>
#include <random>
#include <chrono>

using namespace Eigen;
using namespace std;

class ActivationFunctions{
public:
	static float tanh         (float a){ return std::tanh(a)         ; }
	static float sigm         (float a){ return 1/(1 + exp(-a))      ; }
	static float ReLU         (float a){ return a > 0 ? a : 0        ; }
	static float smoothReLU   (float a){ return log(1 + exp(a))      ; }
	static float leakyReLU    (float a){ return a > 0 ? a : (0.01*a) ; }
};

class WeightInitialisationValues{

private:
	inline static auto _distribution = normal_distribution <float> (-1 , 1);
	inline static default_random_engine _engine;
	
	static float _getRandom(){
		static bool seedGiven = false;
		if (!seedGiven) {
			_engine.seed(chrono::system_clock::now().time_since_epoch().count());
			seedGiven = true;
		}
		
		return _distribution(_engine);
	}
public:
	// just normal normal distribution [-1, 1]
	static float normDistrMinusOneToOne    (int _redundand       ) { return _getRandom()                            ; }
	// weights fitted to activation dunction which output [0, 1]
	static float actFuncRangeZeroToOne     (int inputNeuronsCount) { return _getRandom()*sqrt(1/inputNeuronsCount)  ; }
	// weights fitted to activation dunction which output [-1, 1]
	static float actFuncRangeMinusOneToOne (int inputNeuronsCount) { return _getRandom()*sqrt(2/inputNeuronsCount)  ; }
};



class FeedForwardNeuralNetwork {
private:
	vector < VectorXf, Eigen::aligned_allocator<VectorXf> > _neuronLayers;  // vector of vectors  containing neuron layers
	vector < VectorXf, Eigen::aligned_allocator<VectorXf> > _biases;        // vector of vectors  containing biases for each neuron layer
	vector < MatrixXf, Eigen::aligned_allocator<MatrixXf> > _weights;       // vector of matrices containing weights
	
	float (*_activationFunction)(float);
	// todo think of bias mutation
	
	void _calculate(){
		for(int i = 0; i < _neuronLayers.size() - 1; i++) {
			_neuronLayers[i + 1] = _weights[i] * _neuronLayers[i] + _biases[i];        // calculating;
			_neuronLayers[i + 1] =_neuronLayers[i + 1].unaryExpr(_activationFunction); // applying activation function to each
		}	                                                                           //    neuron in calculated neuron layer
		
	}
	
public:
	FeedForwardNeuralNetwork(const vector < VectorXf, Eigen::aligned_allocator<VectorXf> > & neuronLayers, // for debugging
			                 const vector < VectorXf, Eigen::aligned_allocator<VectorXf> > & biases,
			                 const vector < MatrixXf, Eigen::aligned_allocator<MatrixXf> > & weights,
			                 float (*activationFunction)(float)){
		
		_neuronLayers = neuronLayers;
		_biases       = biases;
		_weights      = weights;
		_activationFunction = activationFunction;
	}
	
	
	FeedForwardNeuralNetwork(const vector<int> &layersNeuronCount,
			           float (*activationFunction)(float), float (*weightInitialisationFunction)(int layerNeuronCount)){
		
		for(int i = 0; i < layersNeuronCount.size(); i++){
			// initializing neuron layers
			_neuronLayers.emplace_back(VectorXf::Zero(layersNeuronCount[i]));
			
			if (i == layersNeuronCount.size() - 1) break; //biases and weights vectors has 1 element less each than neuron layers
			
			// initializing biases
			_biases.emplace_back(VectorXf::Zero(layersNeuronCount[i + 1]));
			
			// initializing weights
			_weights.emplace_back(MatrixXf(_neuronLayers[i + 1].size(), _neuronLayers[i].size()));
			_weights.back() = _weights.back().unaryExpr(
					[&] (int redundant) { return weightInitialisationFunction(_neuronLayers[i].size()); });
		}
		
		_neuronLayers. shrink_to_fit();     // releasing reserved excess memory for ram saving purposes
		_biases.       shrink_to_fit();     //                  because those vectors not going to grow
		_weights.      shrink_to_fit();
		
		
		_activationFunction = activationFunction;
	}
	
	VectorXf getOutputDebug(){
		_calculate();
		return _neuronLayers.back();
	}
};


#endif //NEURAL_SURVIVAL_SIM_CPP_FEEDFORWARDNEURALNETWORK_H
