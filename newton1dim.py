def newton1dim(f, df, x0, eps=1e-10, max_iter=1000):
    x = x0
    it = 0
    while True:
        x_new = x - f(x) / df(x)
        if abs(x - x_new) < eps:
            break
        x = x_new
        it += 1
        if it == max_iter:
            break

    return x_new


def f(x):
    return x ** 3 - 5 * x + 1


def df(x):
    return 3 * x ** 2 - 5


print(newton1dim(f, df, 2))
print(newton1dim(f, df, 0))
print(newton1dim(f, df, -3))
