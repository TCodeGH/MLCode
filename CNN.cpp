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

// 2D Convolutional Layer
class Convolutional2DLayer {
private:
    int input_size;
    int filter_size;
    int output_size;
    int num_filters;
    int stride;

    std::vector<std::vector<std::vector<double>>> filters;
    std::vector<double> biases;

public:
    Convolutional2DLayer(int input_size, int filter_size, int num_filters, int stride)
        : input_size(input_size), filter_size(filter_size), num_filters(num_filters), stride(stride) {
        
        // Calculate output size
        output_size = (input_size - filter_size) / stride + 1;

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
    }

    // Convolution operation
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) {
        std::vector<std::vector<double>> output(output_size, std::vector<double>(output_size));

        for (int i = 0; i < output_size; i += stride) {
            for (int j = 0; j < output_size; j += stride) {
                for (int f = 0; f < num_filters; ++f) {
                    double sum = 0.0;
                    for (int k = 0; k < filter_size; ++k) {
                        for (int l = 0; l < filter_size; ++l) {
                            sum += input[i + k][j + l] * filters[f][k][l];
                        }
                    }
                    output[i / stride][j / stride] += sum + biases[f];
                }
            }
        }

        return output;
    }
};

// Max Pooling Layer
class MaxPoolingLayer {
private:
    int input_size;
    int pool_size;
    int output_size;

public:
    MaxPoolingLayer(int input_size, int pool_size)
        : input_size(input_size), pool_size(pool_size) {
        
        // Calculate output size
        output_size = input_size / pool_size;
    }

    // Max pooling operation
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input) {
        std::vector<std::vector<double>> output(output_size, std::vector<double>(output_size));

        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                double max_val = input[i * pool_size][j * pool_size];
                for (int k = 0; k < pool_size; ++k) {
                    for (int l = 0; l < pool_size; ++l) {
                        max_val = std::max(max_val, input[i * pool_size + k][j * pool_size + l]);
                    }
                }
                output[i][j] = max_val;
            }
        }

        return output;
    }
};

// Fully Connected Layer
class FullyConnectedLayer {
private:
    int input_size;
    int output_size;

    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

public:
    FullyConnectedLayer(int input_size, int output_size)
        : input_size(input_size), output_size(output_size) {
        
        // Initialize weights and biases randomly
        srand(static_cast<unsigned>(time(nullptr)));

        // Initialize weights
        weights.resize(output_size, std::vector<double>(input_size));
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                weights[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }

        // Initialize biases
        biases.resize(output_size);
        for (int i = 0; i < output_size; ++i) {
            biases[i] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
        }
    }

    // Fully connected operation
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> output(output_size);

        for (int i = 0; i < output_size; ++i) {
            output[i] = 0.0;
            for (int j = 0; j < input_size; ++j) {
                output[i] += input[j] * weights[i][j];
            }
            output[i] += biases[i];
        }

        return output;
    }
};

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

int main() {
    // Example usage
    const int input_size = 32;
    const int filter_size = 5;
    const int num_filters = 3;
    const int max_pool_size = 2;
    const int fully_connected_output_size = 2;

    Convolutional2DLayer conv_layer(input_size, filter_size, num_filters, 1);
    MaxPoolingLayer max_pool_layer(input_size, max_pool_size);
    FullyConnectedLayer fully_connected_layer(input_size / max_pool_size, fully_connected_output_size);
    SoftmaxLayer softmax_layer;

    // Sample input data
    std::vector<std::vector<double>> input_data(input_size, std::vector<double>(input_size));
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            input_data[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
        }
    }

    // Forward pass through the convolutional layer
    std::vector<std::vector<double>> conv_output = conv_layer.forward(input_data);

    // Forward pass through the max pooling layer
    std::vector<std::vector<double>> max_pool_output = max_pool_layer.forward(conv_output);

    // Flatten the max pooling output
    std::vector<double> flattened_output;
    for (const auto& row : max_pool_output) {
        flattened_output.insert(flattened_output.end(), row.begin(), row.end());
    }

    // Forward pass through the fully connected layer
    std::vector<double> fully_connected_output = fully_connected_layer.forward(flattened_output);

    // Softmax activation for the final output
    std::vector<double> final_output = softmax_layer.softmax(fully_connected_output);

    // Display results
    std::cout << "Input Data:\n";
    for (const auto& row : input_data) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << '\n';
    }
    std::cout << "\nConvolutional Layer Output:\n";
    for (const auto& row : conv_output) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << '\n';
    }
    std::cout << "\nMax Pooling Layer Output:\n";
    for (const auto& row : max_pool_output) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << '\n';
    }
    std::cout << "\nFully Connected Layer Output:\n";
    for (double val : fully_connected_output) {
        std::cout << val << " ";
    }
    std::cout << "\nSoftmax Activation for Final Output:\n";
    for (double val : final_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
