#include <iostream>
#include <vector>
#include <cmath>
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

// Softmax Activation Layer
class SoftmaxLayer {
public:
    // Softmax activation
    std::vector<double> softmax(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        double sum = 0.0;

        // Calculate exponentials and their sum
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = exp(input[i]);
            sum += output[i];
        }

        // Normalize using the sum
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= sum;
        }

        return output;
    }
};

// Neural Network Class
class NeuralNetwork {
private:
    int input_size;
    int filter_size;
    int num_filters;
    int max_pool_size;
    int fully_connected_output_size;

    std::vector<std::vector<std::vector<double>>> filters;
    std::vector<double> biases;
    std::vector<std::vector<double>> fully_connected_weights;
    std::vector<double> fully_connected_biases;

    double learning_rate;

public:
    NeuralNetwork(int input_size, int filter_size, int num_filters, int max_pool_size, int fully_connected_output_size, double learning_rate)
        : input_size(input_size), filter_size(filter_size), num_filters(num_filters),
          max_pool_size(max_pool_size), fully_connected_output_size(fully_connected_output_size),
          learning_rate(learning_rate) {
        
        // Initialize filters and biases randomly
        srand(static_cast<unsigned>(time(nullptr)));

        // Initialize filters
        filters.resize(num_filters, std::vector<std::vector<double>>(filter_size, std::vector<double>(filter_size)));
        for (int i = 0; i < num_filters; ++i) {
            for (int j = 0; j < filter_size; ++j) {
                for (int k = 0; k < filter_size; ++k) {
                    filters[i][j][k] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
                }
            }
        }

        // Initialize biases
        biases.resize(num_filters);
        for (int i = 0; i < num_filters; ++i) {
            biases[i] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
        }

        // Initialize fully connected layer weights and biases
        fully_connected_weights.resize(fully_connected_output_size, std::vector<double>(input_size / max_pool_size * input_size / max_pool_size));
        for (int i = 0; i < fully_connected_output_size; ++i) {
            for (int j = 0; j < input_size / max_pool_size * input_size / max_pool_size; ++j) {
                fully_connected_weights[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }

        fully_connected_biases.resize(fully_connected_output_size);
        for (int i = 0; i < fully_connected_output_size; ++i) {
            fully_connected_biases[i] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
        }
    }

    // Convolutional operation
    std::vector<std::vector<double>> convolution(const std::vector<std::vector<double>>& input) {
        int output_size = (input_size - filter_size) / 1 + 1;
        std::vector<std::vector<double>> output(output_size, std::vector<double>(output_size));

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                for (int f = 0; f < num_filters; ++f) {
                    double sum = 0.0;
                    for (int k = 0; k < filter_size; ++k) {
                        for (int l = 0; l < filter_size; ++l) {
                            sum += input[i + k][j + l] * filters[f][k][l];
                        }
                    }
                    output[i][j] += sum + biases[f];
                }
            }
        }

        return output;
    }

    // Max pooling operation
    std::vector<std::vector<double>> maxPooling(const std::vector<std::vector<double>>& input) {
        int output_size = input_size / max_pool_size;
        std::vector<std::vector<double>> output(output_size, std::vector<double>(output_size));

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                double max_val = input[i * max_pool_size][j * max_pool_size];
                for (int k = 0; k < max_pool_size; ++k) {
                    for (int l = 0; l < max_pool_size; ++l) {
                        max_val = std::max(max_val, input[i * max_pool_size + k][j * max_pool_size + l]);
                    }
                }
                output[i][j] = max_val;
            }
        }

        return output;
    }

    // Flatten operation
    std::vector<double> flatten(const std::vector<std::vector<double>>& input) {
        std::vector<double> flattened_output;
        for (const auto& row : input) {
            flattened_output.insert(flattened_output.end(), row.begin(), row.end());
        }
        return flattened_output;
    }

    // Fully connected operation
    std::vector<double> fullyConnected(const std::vector<double>& input) {
        std::vector<double> output(fully_connected_output_size);

        for (int i = 0; i < fully_connected_output_size; ++i) {
            output[i] = 0.0;
            for (int j = 0; j < input.size(); ++j) {
                output[i] += input[j] * fully_connected_weights[i][j];
            }
            output[i] += fully_connected_biases[i];
        }

        return output;
    }

    // Softmax activation
    std::vector<double> softmax(const std::vector<double>& input) {
        std::vector<double> output(input.size());
        double sum = 0.0;

        // Calculate exponentials and their sum
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = exp(input[i]);
            sum += output[i];
        }

        // Normalize using the sum
        for (size_t i = 0; i < output.size(); ++i) {
            output[i] /= sum;
        }

        return output;
    }

    // Train function
    void train(const std::vector<std::vector<double>>& input, int target_class) {
        // Forward pass
        std::vector<std::vector<double>> conv_output = convolution(input);
        std::vector<std::vector<double>> pool_output = maxPooling(conv_output);
        std::vector<double> flattened_output = flatten(pool_output);
        std::vector<double> fully_connected_output = fullyConnected(flattened_output);
        std::vector<double> final_output = softmax(fully_connected_output);

        // Compute loss (cross-entropy loss)
        double loss = -log(final_output[target_class]);

        // Backward pass

        // Gradient for softmax layer
        std::vector<double> softmax_grad(final_output.size(), 0.0);
        softmax_grad[target_class] = 1.0;

        // Backpropagation through the fully connected layer
        std::vector<double> fully_connected_grad(fully_connected_output.size(), 0.0);
        for (size_t i = 0; i < fully_connected_output.size(); ++i) {
            fully_connected_grad[i] = (final_output[i] - softmax_grad[i]) * sigmoid_derivative(final_output[i]);
        }

        // Update fully connected layer weights and biases
        for (int i = 0; i < fully_connected_output_size; ++i) {
            for (size_t j = 0; j < flattened_output.size(); ++j) {
                fully_connected_weights[i][j] -= learning_rate * fully_connected_grad[i] * flattened_output[j];
            }
            fully_connected_biases[i] -= learning_rate * fully_connected_grad[i];
        }

        // Backpropagation through the flattening operation
        std::vector<double> flattened_grad(flattened_output.size(), 0.0);
        for (size_t i = 0; i < flattened_output.size(); ++i) {
            for (int j = 0; j < fully_connected_output_size; ++j) {
                flattened_grad[i] += fully_connected_grad[j] * fully_connected_weights[j][i];
            }
        }

        // Backpropagation through the max pooling layer
        std::vector<std::vector<double>> pool_grad(input_size, std::vector<double>(input_size, 0.0));
        for (int i = 0; i < pool_grad.size(); ++i) {
            for (int j = 0; j < pool_grad[0].size(); ++j) {
                pool_grad[i][j] = flattened_grad[i * max_pool_size + j];
            }
        }

        // Backpropagation through the convolutional layer
        std::vector<std::vector<std::vector<double>>> conv_grad(num_filters, std::vector<std::vector<double>>(filter_size, std::vector<double>(filter_size, 0.0)));
        for (int f = 0; f < num_filters; ++f) {
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    for (int k = 0; k < pool_grad.size(); ++k) {
                        for (int l = 0; l < pool_grad[0].size(); ++l) {
                            conv_grad[f][i][j] += input[k + i][l + j] * pool_grad[k][l];
                        }
                    }
                }
            }
        }

        // Update convolutional layer filters and biases
        for (int f = 0; f < num_filters; ++f) {
            for (int i = 0; i < filter_size; ++i) {
                for (int j = 0; j < filter_size; ++j) {
                    filters[f][i][j] -= learning_rate * conv_grad[f][i][j];
                }
            }
            biases[f] -= learning_rate * biases[f];
        }
    }
};

int main() {
    // Example usage
    const int input_size = 32;
    const int filter_size = 5;
    const int num_filters = 3;
    const int max_pool_size = 2;
    const int fully_connected_output_size = 2;
    const double learning_rate = 0.01;

    NeuralNetwork neural_network(input_size, filter_size, num_filters, max_pool_size, fully_connected_output_size, learning_rate);

    // Sample input data
    std::vector<std::vector<double>> input_data(input_size, std::vector<double>(input_size));
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            input_data[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
        }
    }

    // Target class for training
    int target_class = rand() % fully_connected_output_size;

    // Train the neural network
    neural_network.train(input_data, target_class);

    std::cout << "Training complete!" << std::endl;

    return 0;
}
