#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

// Activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
private:
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;

    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size, double learning_rate)
        : input_size(input_size), hidden_size(hidden_size), output_size(output_size), learning_rate(learning_rate) {
        
        // Initialize weights randomly
        srand(static_cast<unsigned>(time(nullptr)));

        // Weights from input to hidden layer
        weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                weights_input_hidden[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }

        // Weights from hidden to output layer
        weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                weights_hidden_output[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }
    }

    // Forward pass
    std::vector<double> predict(const std::vector<double>& input) {
        // Calculate values for the hidden layer
        std::vector<double> hidden(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < input_size; ++j) {
                sum += input[j] * weights_input_hidden[j][i];
            }
            hidden[i] = sigmoid(sum);
        }

        // Calculate values for the output layer
        std::vector<double> output(output_size);
        for (int i = 0; i < output_size; ++i) {
            double sum = 0.0;
            for (int j = 0; j < hidden_size; ++j) {
                sum += hidden[j] * weights_hidden_output[j][i];
            }
            output[i] = sigmoid(sum);
        }

        return output;
    }

    // Backpropagation
    void train(const std::vector<std::vector<double>>& training_inputs, const std::vector<std::vector<double>>& training_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < training_inputs.size(); ++i) {
                // Forward pass
                const std::vector<double>& input = training_inputs[i];
                const std::vector<double>& target = training_outputs[i];

                // Calculate values for the hidden layer
                std::vector<double> hidden(hidden_size);
                for (int j = 0; j < hidden_size; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < input_size; ++k) {
                        sum += input[k] * weights_input_hidden[k][j];
                    }
                    hidden[j] = sigmoid(sum);
                }

                // Calculate values for the output layer
                std::vector<double> output(output_size);
                for (int j = 0; j < output_size; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < hidden_size; ++k) {
                        sum += hidden[k] * weights_hidden_output[k][j];
                    }
                    output[j] = sigmoid(sum);
                }

                // Backpropagation

                // Calculate error for the output layer
                std::vector<double> output_errors(output_size);
                for (int j = 0; j < output_size; ++j) {
                    output_errors[j] = target[j] - output[j];
                }

                // Calculate gradient for the output layer
                std::vector<double> output_gradients(output_size);
                for (int j = 0; j < output_size; ++j) {
                    output_gradients[j] = output_errors[j] * sigmoid_derivative(output[j]);
                }

                // Update weights for the hidden to output layer
                for (int j = 0; j < hidden_size; ++j) {
                    for (int k = 0; k < output_size; ++k) {
                        weights_hidden_output[j][k] += learning_rate * output_gradients[k] * hidden[j];
                    }
                }

                // Calculate error for the hidden layer
                std::vector<double> hidden_errors(hidden_size);
                for (int j = 0; j < hidden_size; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < output_size; ++k) {
                        sum += output_errors[k] * weights_hidden_output[j][k];
                    }
                    hidden_errors[j] = sum;
                }

                // Calculate gradient for the hidden layer
                std::vector<double> hidden_gradients(hidden_size);
                for (int j = 0; j < hidden_size; ++j) {
                    hidden_gradients[j] = hidden_errors[j] * sigmoid_derivative(hidden[j]);
                }

                // Update weights for the input to hidden layer
                for (int j = 0; j < input_size; ++j) {
                    for (int k = 0; k < hidden_size; ++k) {
                        weights_input_hidden[j][k] += learning_rate * hidden_gradients[k] * input[j];
                    }
                }
            }
        }
    }
};

int main() {
    // Example usage
    NeuralNetwork neuralNetwork(2, 2, 1, 0.1);

    // Training data (XOR function)
    std::vector<std::vector<double>> training_inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> training_outputs = {{0}, {1}, {1}, {0}};

    // Train the neural network
    neuralNetwork.train(training_inputs, training_outputs, 10000);

    // Test the trained network
    for (const auto& input : training_inputs) {
        std::vector<double> predicted_output = neuralNetwork.predict(input);
        std::cout << "Input: {" << input[0] << ", " << input[1] << "} => Output: " << predicted_output[0] << std::endl;
    }

    return 0;
}
