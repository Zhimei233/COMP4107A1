import torch
import math
from assignment1 import gradient_descent

def a_solution():
    # Constants
    x1 = 3
    x2 = -2
    y1 = 0.5
    y2 = -0.75

    # f(a, b)
    def f_ab(params):
        a, b = params
        term1 = a * x1 + b - y1
        term2 = a * x2 + b - y2
        return 0.5 * (term1**2 + term2**2)

    # gradient of f(a, b)
    def df_ab(params):
        a, b = params

        df_da = 3 * (3*a + b - 0.5) - 2 * (-2*a + b + 0.75)
        df_db = (3*a + b - 0.5) + (-2*a + b + 0.75)

        return [df_da, df_db]

    # Initial guess and learning rate
    x0 = [0.0, 0.0]
    alpha = 0.01

    # Run gradient descent
    argmin, min_val = gradient_descent(f_ab, df_ab, x0, alpha)

    print("Minimum achieved at:")
    print("a =", argmin[0])
    print("b =", argmin[1])
    print("Minimum value of f =", min_val)

a_solution()

def b_solution():
    # Constants
    x1 = 3
    x2 = -2
    y1 = 0.5
    y2 = -0.75

    # SiLU
    def silu(z):
        return z / (1.0 + math.exp(-z))

    # derivative of SiLU
    def dsilu(z):
        sig = 1.0 / (1.0 + math.exp(-z))     # sigmoid(z)
        return sig + z * sig * (1.0 - sig)   # s'(z)

    # f(a, b)
    def f_ab(params):
        a, b = params
        z1 = a * x1 + b
        z2 = a * x2 + b
        return 0.5 * ((silu(z1) - y1) ** 2 + (silu(z2) - y2) ** 2)

    # gradient of f(a, b)
    def df_ab(params):
        a, b = params
        z1 = a * x1 + b
        z2 = a * x2 + b

        t1 = (silu(z1) - y1) * dsilu(z1)
        t2 = (silu(z2) - y2) * dsilu(z2)

        df_da = t1 * x1 + t2 * x2
        df_db = t1 + t2
        return [df_da, df_db]

    # Initial guess and learning rate
    x0 = [0.0, 0.0]
    alpha = 0.1

    argmin, min_val = gradient_descent(f_ab, df_ab, x0, alpha)

    print("Minimum achieved at:")
    print("a =", argmin[0])
    print("b =", argmin[1])
    print("Minimum value of f =", min_val)

b_solution()

