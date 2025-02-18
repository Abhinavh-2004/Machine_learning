import numpy as np
import matplotlib.pyplot as plt  

class Backpropagation:
    def __init__(self):
        self.w1 = np.random.rand()  
        self.w2 = np.random.rand()  
        self.w3 = np.random.rand()  
        self.w4 = np.random.rand()  
        self.b1 = 0
        self.b2 = 0
        self.b3 = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def train(self, input, output):
        errors = []  
        iteration_errors = []  
        count = 0
        while count < 100:
            iteration_error = 0  
            for inp, observed in zip(input, output):  
                x_1_i = (inp * self.w1) + self.b1
                x_2_i = (inp * self.w2) + self.b2
                y_1_i = self.sigmoid(x_1_i)
                y_2_i = self.sigmoid(x_2_i)
                x_1_i = y_1_i * self.w3
                x_2_i = y_2_i * self.w4
                sum_user = y_1_i * y_2_i
                y_hat = sum_user + self.b3
                computed_error = (observed - y_hat) ** 2  
                det_of_b3 = -2 * (observed - y_hat)

                errors.append(computed_error)
                iteration_error += computed_error  

                det_of_w3 = -2 * (observed - y_hat) * y_1_i
                det_of_w4 = -2 * (observed - y_hat) * y_2_i
                det_of_b1 = -2 * (observed - y_hat) * self.w3 * self.sigmoid_derivative(x_1_i)
                det_of_b2 = -2 * (observed - y_hat) * self.w4 * self.sigmoid_derivative(x_2_i)
                det_of_w1 = -2 * (observed - y_hat) * self.w3 * self.sigmoid_derivative(x_1_i) * inp
                det_of_w2 = -2 * (observed - y_hat) * self.w4 * self.sigmoid_derivative(x_2_i) * inp

                self.w1 -= 0.01 * det_of_w1
                self.w2 -= 0.01 * det_of_w2
                self.w3 -= 0.01 * det_of_w3
                self.w4 -= 0.01 * det_of_w4
                self.b1 -= 0.01 * det_of_b1
                self.b2 -= 0.01 * det_of_b2
                self.b3 -= 0.01 * det_of_b3

            iteration_errors.append(iteration_error / len(input))  
            count += 1

        
        plt.plot(range(1, count + 1), iteration_errors, marker='o')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Error')
        plt.title('Error vs Number of Iterations')
        plt.grid(True)
        plt.show()

        return sum(errors)  

    def predict(self, input):
        x_1_i = (input * self.w1) + self.b1
        x_2_i = (input * self.w2) + self.b2
        y_1_i = self.sigmoid(x_1_i)
        y_2_i = self.sigmoid(x_2_i)
        x_1_i = y_1_i * self.w3
        x_2_i = y_2_i * self.w4
        sum_user = y_1_i * y_2_i
        y_hat = sum_user + self.b3
        return y_hat



inputs = np.array([0.5, 1.0, 1.5, 2.0])  
outputs = np.array([0.0, 1.0, 1.0, 0.0])  


bp = Backpropagation()


error = bp.train(inputs, outputs)
print(f"Training Error: {error}")


new_input = 1.2  
prediction = bp.predict(new_input)
print(f"Prediction for input {new_input}: {prediction}")
