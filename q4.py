import torch
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