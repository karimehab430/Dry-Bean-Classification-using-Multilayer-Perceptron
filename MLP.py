import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh_derivative(x):
    return 1 - (x ** 2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def encode_y(y):
    return [np.where(y == cls, 1, 0) for cls in np.unique(y)]


class Neuron:
    def __init__(self, parent, num_of_weights, activation):
        self.activation = sigmoid if activation == 'Sigmoid' else np.tanh
        rng = np.random.RandomState()
        self.weights = rng.random(num_of_weights + int(parent.bias))
        self.parent = parent

    def activate(self, x):
        if self.parent.bias:
            x = np.insert(x, 0, 1)
        y_pred = np.dot(x, self.weights)
        activation = self.activation(y_pred)
        return activation

    def update_weights(self, gradient, x):
        if self.parent.bias:
            x = np.insert(x, 0, 1)

        self.weights += gradient * self.parent.learning_rate * x


class MultilayerBackpropagation:

    def __init__(self, bias, learning_rate, epochs, layers, activation):
        self.bias = bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.layers = layers
        self.f_net_derivative = sigmoid_derivative if activation == 'Sigmoid' else tanh_derivative

        self.layers = [
            [Neuron(self, layers[i - 1], activation) for _ in range(layers[i])]
            for i in range(1, len(layers))
        ]

    def forward_step(self, x):

        f_net = [x] + [np.zeros(len(layer)) for layer in self.layers]

        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                f_net[i + 1][j] = self.layers[i][j].activate(f_net[i])
        return f_net

    def backward_step(self, layers_outputs, y_encoded, x_index):

        layers_gradients = []

        for layer in self.layers:
            layers_gradients.append([0] * len(layer))

        last_layer_index = len(self.layers) - 1
        last_layer = self.layers[last_layer_index]

        # Calculate gradients for neurons in the last layer
        for i in range(len(last_layer)):
            f_net = layers_outputs[last_layer_index + 1][i]  # f_net_output
            weight_update_sum = y_encoded[i][x_index] - f_net

            # sigma_output = f'(net) * (d_k - f_net_k)
            gradient = self.f_net_derivative(f_net) * weight_update_sum
            layers_gradients[last_layer_index][i] = gradient

            # delta w_output = learning rate * sigma_output * f_net_output
            last_layer[i].update_weights(gradient, layers_outputs[last_layer_index])

        # Calculate gradients for neurons in the hidden layers
        for layer_index in range(last_layer_index - 1, -1, -1):
            layer = self.layers[layer_index]

            for i in range(len(layer)):
                weight_update_sum = 0
                next_layer = self.layers[layer_index + 1]

                for next_neuron_index in range(len(next_layer)):
                    weight_index = i + int(self.bias)
                    weight = next_layer[next_neuron_index].weights[weight_index]

                    # summation(sigma * weight)
                    weight_update_sum += layers_gradients[layer_index + 1][next_neuron_index] * weight

                # sigma_hidden = f'(net) * summation(sigma * weight)
                gradient = self.f_net_derivative(layers_outputs[layer_index + 1][i]) * weight_update_sum
                layers_gradients[layer_index][i] = gradient

                # delta w_hidden = learning rate * sigma_hidden * output from previous layer
                layer[i].update_weights(gradient, layers_outputs[layer_index])

        # Check if the predicted class matches the target class
        error = int(y_encoded[np.argmax(layers_outputs[-1])][x_index] != 1)
        return error

    def train(self, input_data, target_data):
        x = np.array(input_data)
        y = np.array(target_data)

        y_encoded = encode_y(y)
        for _ in range(self.epochs):

            error = 0

            for i in range(x.shape[0]):
                error += self.backward_step(self.forward_step(x[i]), y_encoded, i)

            avg_error = error / x.shape[0]
            print(avg_error)
            if avg_error < 0.0001:
                break

    def predict(self, input_data):
        X = np.array(input_data)
        predictions = []
        for i in range(X.shape[0]):
            predictions.append(np.argmax(self.forward_step(X[i])[-1]))
        return predictions

    def evaluate(self, input_data, target_data):
        y_pred = self.predict(input_data)
        y_pred = np.array(y_pred)
        y_test = np.array(target_data)
        correct = sum(y_test == y_pred)
        accuracy = correct / len(y_test)
        print('Accuracy:', accuracy, '\n')


def calculate_confusion_matrix(actual_labels, predicted_labels):
    # Get Unique Values
    unique_labels = list(set(actual_labels) | set(predicted_labels))
    class_1 = unique_labels[0]
    class_2 = unique_labels[1]
    class_3 = unique_labels[2]
    true_class_1 = 0
    false_class_1_2 = 0
    false_class_1_3 = 0
    true_class_2 = 0
    false_class_2_1 = 0
    false_class_2_3 = 0
    true_class_3 = 0
    false_class_3_1 = 0
    false_class_3_2 = 0
    # predict is first and actual is second
    for actual, predicted in zip(actual_labels, predicted_labels):
        if predicted == class_1 and actual == class_1:
            true_class_1 += 1
        elif predicted == class_1:
            if actual == class_2:
                false_class_1_2 += 1
            else:
                false_class_1_3 += 1
        if predicted == class_2 and actual == class_2:
            true_class_2 += 1
        elif predicted == class_2:
            if actual == class_1:
                false_class_2_1 += 1
            else:
                false_class_2_3 += 1
        if predicted == class_3 and actual == class_3:
            true_class_3 += 1
        elif predicted == class_3:
            if actual == class_1:
                false_class_3_1 += 1
            else:
                false_class_3_2 += 1

    print("True BOMBAY : ", true_class_1, "  ", "False BOMBAY But Was CALI : ", false_class_1_2, "  ",
          "False BOMBAY But Was "
          "SIRA : ",
          false_class_1_3, "\n")
    print("True CALI : ", true_class_2, "  ", "False CALI But Was BOMBAY : ", false_class_2_1, "  ",
          "False CALI But Was "
          "SIRA : ",
          false_class_2_3, "\n")
    print("True SIRA : ", true_class_3, "  ", "False SIRA But Was BOMBAY : ", false_class_3_1, "  ",
          "False SIRA But Was "
          "CALI : ",
          false_class_3_2, "\n")

    confusion_matrix = np.array([[true_class_1, false_class_1_2, false_class_1_3],
                                 [false_class_2_1, true_class_2, false_class_2_3],
                                 [false_class_3_1, false_class_3_2, true_class_3]])

    group_names = ['True BOMBAY', 'False CALI', 'False SIRA', 'False BOMBAY', 'True CALI', 'False SIRA', 'False BOMBAY',
                   'False CALI',
                   'True SIRA']
    group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape((3, 3))

    Class_Label = ['BOMBAY', 'CALI', 'SIRA']
    figure, axis = plt.subplots()
    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap="binary")
    axis.set_xlabel('Predicted')
    axis.set_ylabel('Actual')
    axis.set_xticklabels(Class_Label)
    axis.set_xticks(np.arange(len(Class_Label)) + 0.5)
    axis.set_yticklabels(Class_Label)
    axis.set_yticks(np.arange(len(Class_Label)) + 0.5)
    axis.set_title('Confusion Matrix')
    plt.show()


def calculate_confusion_matrix_dynamic(actual_labels, predicted_labels):
    unique_labels = list(set(actual_labels) | set(predicted_labels))
    num_of_classes = len(unique_labels)

    confusion_matrix = np.zeros((num_of_classes, num_of_classes), dtype=int)

    for actual, predicted in zip(actual_labels, predicted_labels):
        actual_index = np.where(unique_labels == actual)[0][0]
        predicted_index = np.where(unique_labels == predicted)[0][0]
        confusion_matrix[actual_index, predicted_index] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", xticklabels=["BOMBAY", "CALI", "SIRA"],
                yticklabels=["BOMBAY", "CALI", "SIRA"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
