# Day_03_01_cost_gradient_bias.py


def bias_basic():
    def cost(x, y, w, b):
        c = 0
        for i in range(len(x)):
            hx = w * x[i] + b
            c += (hx - y[i]) ** 2
        return c / len(x)


    def gradient_descent(x, y, w, b):
        c1, c2 = 0, 0
        for i in range(len(x)):
            hx = w * x[i] + b
            c1 += (hx - y[i]) * x[i]
            c2 += (hx - y[i])
        return c1 / len(x), c2 / len(x)


    def show_gradient():
        x = [1, 2, 3]
        y = [1, 2, 3]

        w, b = 10, 10
        for i in range(1000):
            c = cost(x, y, w, b)
            g1, g2 = gradient_descent(x, y, w, b)
            w -= 0.1 * g1
            b -= 0.1 * g2

            print(i, c)

        print(w)
        print(b)

    show_gradient()


def bias_advanced():
    def cost(x, y, w):
        c = 0
        for i in range(len(x)):
            hx = w[0] * x[i][0] + w[1] * x[i][1] + w[2] * x[i][2]
            c += (hx - y[i]) ** 2
        return c / len(x)

    def gradient_descent(x, y, w):
        c1, c2, c3 = 0, 0, 0
        for i in range(len(x)):
            hx = w[0] * x[i][0] + w[1] * x[i][1] + w[2] * x[i][2]
            c1 += (hx - y[i]) * x[i][0]
            c2 += (hx - y[i]) * x[i][1]
            c3 += (hx - y[i]) * x[i][2]
        return c1 / len(x), c2 / len(x), c3 / len(x)

    # 문제
    # 새로운 데이터에 맞게 w와 b를 처리해보세요.
    def show_gradient():
        x = [[1., 1., 0.],
             [1., 0., 2.],
             [1., 3., 0.],
             [1., 0., 4.],
             [1., 5., 0.]]
        y = [1, 2, 3, 4, 5]

        w = [10, 10, 10]
        for i in range(1000):
            c = cost(x, y, w)
            g = gradient_descent(x, y, w)
            w[0] -= 0.1 * g[0]
            w[1] -= 0.1 * g[1]
            w[2] -= 0.1 * g[2]

            print(i, c)

        print(w)
        # [8.716748764374427e-07, 0.9999997708612389, 0.9999997281286177]

    show_gradient()


# bias_basic()
bias_advanced()
