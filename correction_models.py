def linear_correction(x, p):
    return (p[0] + 1) * x + p[1]


def quadratic_correction(x, p):
    return p[0] * x ** 2 + (p[1] + 1) * x + p[2]
