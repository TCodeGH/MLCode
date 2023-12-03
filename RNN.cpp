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

// Mean Squared Error loss
double meanSquaredError(const std::vector<double>& predicted, const std::vector<double>& target) {
    double error = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        error += 0.5 * pow(predicted[i] - target[i], 2);
    }
    return error;
}

// RNN Layer
class RNNLayer {
private:
    int input_size;
    int hidden_size;

    std::vector<std::vector<double>> input_weights;
    std::vector<std::vector<double>> recurrent_weights;
    std::vector<double> biases;

    std::vector<double> hidden_state;

public:
    RNNLayer(int input_size, int hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {
        
        // Initialize weights and biases randomly
        srand(static_cast<unsigned>(time(nullptr)));

        // Initialize input weights
        input_weights.resize(hidden_size, std::vector<double>(input_size));
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                input_weights[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }

        // Initialize recurrent weights
        recurrent_weights.resize(hidden_size, std::vector<double>(hidden_size));
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                recurrent_weights[i][j] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }

        // Initialize biases
        biases.resize(hidden_size);
        for (int i = 0; i < hidden_size; ++i) {
            biases[i] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
        }

        // Initialize hidden state
        hidden_state.resize(hidden_size, 0.0);
    }

    // Forward pass
    std::vector<double> forward(const std::vector<double>& input) {
        // Update hidden state using input and recurrent connections
        for (int i = 0; i < hidden_size; ++i) {
            hidden_state[i] = 0.0;
            for (int j = 0; j < input_size; ++j) {
                hidden_state[i] += input_weights[i][j] * input[j];
            }
            for (int j = 0; j < hidden_size; ++j) {
                hidden_state[i] += recurrent_weights[i][j] * hidden_state[j];
            }
            hidden_state[i] += biases[i];
            hidden_state[i] = sigmoid(hidden_state[i]);
        }

        return hidden_state;
    }

    // Backward pass
    void backward(const std::vector<double>& input, const std::vector<double>& target, double learning_rate) {
        // Compute gradients and update weights and biases

        // Compute error gradient with respect to output
        std::vector<double> output_grad(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            output_grad[i] = (hidden_state[i] - target[i]) * sigmoid_derivative(hidden_state[i]);
        }

        // Update input weights
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                input_weights[i][j] -= learning_rate * output_grad[i] * input[j];
            }
        }

        // Update recurrent weights
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                recurrent_weights[i][j] -= learning_rate * output_grad[i] * hidden_state[j];
            }
        }

        // Update biases
        for (int i = 0; i < hidden_size; ++i) {
            biases[i] -= learning_rate * output_grad[i];
        }
    }
};

int main() {
    // Example usage
    const int input_size = 3;
    const int hidden_size = 4;
    const double learning_rate = 0.01;

    RNNLayer rnn_layer(input_size, hidden_size);

    // Sample input sequence and target
    std::vector<std::vector<double>> input_sequence = {
        {0.5, 0.2, 0.1},
        {0.3, 0.1, 0.4},
        {0.7, 0.6, 0.2}
    };

    std::vector<std::vector<double>> targets = {
        {0.4, 0.5, 0.6, 0.2},
        {0.1, 0.7, 0.9, 0.5},
        {0.6, 0.8, 0.3, 0.1}
    };

    // Training loop
    for (size_t t = 0; t < input_sequence.size(); ++t) {
        // Forward pass
        std::vector<double> output = rnn_layer.forward(input_sequence[t]);

        // Compute and print loss
        double loss = meanSquaredError(output, targets[t]);
        std::cout << "Epoch " << t + 1 << ", Loss: " << loss << std::endl;

        // Backward pass
        rnn_layer.backward(input_sequence[t], targets[t], learning_rate);
    }

    return 0;
}
