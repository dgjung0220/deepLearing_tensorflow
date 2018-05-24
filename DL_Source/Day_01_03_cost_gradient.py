# Day_01_03_cost_gradient.py
import matplotlib.pyplot as plt


def cost(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) ** 2
    return c / len(x)


def gradient_descent(x, y, w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        c += (hx - y[i]) * x[i]
    return c / len(x)


def cost_graph():
    # y = ax + b
    # hx = wx + b
    # y = x
    x = [1, 2, 3]
    y = [1, 2, 3]

    print(cost(x, y, 0))
    print(cost(x, y, 1))
    print(cost(x, y, 2))

    # 문제
    # -3에서부터 5까지의 cost를 계산해보세요.
    # 이때 증가는 0.1씩 진행합니다.
    weights, costs = [], []
    for i in range(-30, 50):
        w = i / 10
        c = cost(x, y, w)
        print(w, c)

        # plt.plot(w, c, 'ro')
        weights.append(w)
        costs.append(c)

    plt.plot(weights, costs, 'ro')
    # plt.plot(weights, costs)
    plt.show()


def show_gradient():
    x = [1, 2, 3]
    y = [1, 2, 3]

    old_c = 100000000
    w = 10
    for i in range(1000):
        c = cost(x, y, w)
        g = gradient_descent(x, y, w)
        w -= 0.1 * g

        # early stopping
        # if g < 1e-15:
        # if c < 1e-15:
        if old_c - c < 1e-15:
            break

        print(i, c)
        old_c = c

    print('-' * 50)
    print('5 :', w * 5)
    print('7 :', w * 7)

    # 문제
    # w가 1.0이 되게 만드는 두 가지 코드를 찾아보세요.
    # x가 5와 7일 때의 y값을 예측해보세요.


cost_graph()
# show_gradient()

# 미분 : 기울기. 순간변화량.
#        x축으로 1만큼 움직일 때 y축으로 움직인 거리

# y = 3                 3=1, 3=2, 3=3
# y = x                 1=1, 2=2, 3=3
# y = 2x                2=1, 4=2, 6=3
# y = (x + 1)           2=1, 3=2, 4=3
# y = xz
# y = x ^ 2             1=1, 4=2, 9=3 : 2 * x
# y = (x + 1) ^ 2                     : 2 * (x + 1)
