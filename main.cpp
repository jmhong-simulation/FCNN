#include <iostream>

#include "NeuralNetwork.h"

void main()
{
	VectorND<D> output;

	NeuralNetwork nn_;

	nn_.initialize(2, 2, 0);

	nn_.feedForward();

	nn_.copyOutputVector(output);

	std::cout << output << std::endl;
}