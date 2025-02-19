import numpy as np
import matplotlib.pyplot as plt

class linear_reg:
    def __init__(self):
        self.slope = 0
        self.c = 0

    def train(self, X, Y):
        x_average = np.average(X)
        y_average = np.average(Y)
        
        numerator = 0
        denominator = 0
        
        for x, y in zip(X, Y):
            numerator += (x - x_average) * (y - y_average)
            denominator += (x - x_average) ** 2
        
        if denominator == 0:
            raise ValueError("Division by zero error: All x values are the same.")
        
        self.slope = numerator / denominator
        self.c = y_average - self.slope * x_average
        
        print(f"The predicted function is: y = {self.slope}x + {self.c}")

    def test(self, X, Y):
        error = 0
        for x, y in zip(X, Y):
            error += (self.slope * x + self.c - y) ** 2
        print(f"The error found is {error}")

    def plot(self, X, Y):
        
        plt.scatter(X, Y, color='blue', label='Data Points')
        
        
        regression_line = [self.slope * x + self.c for x in X]
        plt.plot(X, regression_line, color='red', label='Regression Line')
        
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

    def predict_and_update(self, x, X, Y):
        
            y_pred = self.slope * x + self.c
            print(f"For the given input {x}, the predicted output is: {y_pred}")
            
            
            X = np.append(X, [x])
            Y = np.append(Y, [y_pred])
            
            
            self.train(X, Y)
            
            
            self.test(X, Y)
            
            
            return X, Y
# Static dataset
X = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
    91, 92, 93, 94, 95, 96, 97, 98, 99, 100
])

Y = np.array([
    2, 4, 5, 7, 8, 10, 11, 13, 14, 16,
    17, 19, 20, 22, 23, 25, 26, 28, 29, 31,
    32, 34, 35, 37, 38, 40, 41, 43, 44, 46,
    47, 49, 50, 52, 53, 55, 56, 58, 59, 61,
    62, 64, 65, 67, 68, 70, 71, 73, 74, 76,
    77, 79, 80, 82, 83, 85, 86, 88, 89, 91,
    92, 94, 95, 97, 98, 100, 101, 103, 104, 106,
    107, 109, 110, 112, 113, 115, 116, 118, 119, 121,
    122, 124, 125, 127, 128, 130, 131, 133, 134, 136,
    137, 139, 140, 142, 143, 145, 146, 148, 149, 151
])

# Create an instance of the linear_reg class
linear = linear_reg()

# Train the model
linear.train(X, Y)

# Test the model
linear.test(X, Y)

# Predict for a new input
X,Y = linear.predict_and_update(104,X,Y)

print(X)
print(Y)
