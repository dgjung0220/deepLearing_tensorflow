# Day_03_04_sigmoid.py
import math
import matplotlib.pyplot as plt


def show_sigmoid():
    def sigmoid(z):
        return 1 / (1 + math.e ** -z)


    print(math.e)

    print(sigmoid(-10))
    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))
    print(sigmoid(10))

    for i in range(-10, 10):
        s = sigmoid(i)

        plt.plot(i, s, 'ro')

    plt.show()


def select_ab(y):
    def A():
        return 'A'

    def B():
        return 'B'

    print(y*A() + (1-y)*B())

    if y == 1:
        print(A())
    else:
        print(B())


# show_sigmoid()
select_ab(0)
select_ab(1)
