#include <iostream>

#include "NeuralNetwork.h"

void main()
{
	VectorND<D> x(2);
	x[0] = 0.0; x[1] = 0.0;

	VectorND<D> y_target(2);
	y_target[0] = 1.0f;

	VectorND<D> y_temp(2);

	NeuralNetwork nn_;
	nn_.initialize(2, 1, 0);
	nn_.alpha_ = 0.01;

	for (int i = 0; i < 10000; i++)
	{
		nn_.setInputVector(x);
		nn_.propForward();

		nn_.copyOutputVector(y_temp);
		std::cout << y_temp << std::endl;

		nn_.propBackward(y_target);
	}
}