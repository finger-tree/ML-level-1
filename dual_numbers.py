class dual_numbers:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        return dual_numbers(self.real + other.real, self.dual + other.dual)

    def __sub__(self, other):
        return dual_numbers(self.real - other.real, self.dual - other.dual)

    def __mul__(self, other):
        return dual_numbers(self.real * other.real,
                            self.real * other.dual + self.dual * other.real)

    def __truediv__(self, other):
        denom = other.real ** 2
        return dual_numbers(self.real / other.real,
                            (self.dual * other.real - self.real * other.dual) / denom)

    def __repr__(self):
        return f"{self.real} + {self.dual}ε"
    
# Example usage:  
if __name__ == "__main__":
    a = dual_numbers(3, 1)  # Represents 3 + 1ε
    b = dual_numbers(2, 4)  # Represents 2 + 4ε

    print("a:", a)
    print("b:", b)

    print("Addition:", a + b)
    print("Subtraction:", a - b)
    print("Multiplication:", a * b)
    print("Division:", a / b)

    def babylonian_sqrt(x, iterations=10):
        guess = dual_numbers(x.real / 2, 0)
        for _ in range(iterations):
            guess = (guess + x / guess) * dual_numbers(0.5, 0)
            print("Current guess:", guess)
        return guess

    print("Square root of b:", babylonian_sqrt(babylonian_sqrt(dual_numbers(2401, 1))))


