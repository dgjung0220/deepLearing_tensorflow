# Day_02_01_numpy.py
import numpy as np

a1 = np.arange(5)
print(a1)
print(type(a1))     # <class 'numpy.ndarray'>
print(a1.dtype)

print(a1[0], a1[-1])
print(a1[:3])

a1[0] = 99
a1[:3] = 55
a1[::2] = 33
print(a1)

a2 = np.arange(6)
a3 = np.arange(6).reshape(2, 3)
a4 = np.arange(12).reshape(2, 2, 3)

print(a2)
print(a3)
print(a4)

print(a2.shape, a2.size)
print(a3.shape, a3.size)
print(a4.shape, a4.size, a4.ndim)

a5 = np.arange(6).reshape(-1, 3)
a6 = np.arange(6).reshape(2, -1)
print(a5.shape, a6.shape)
print('-' * 50)

# 문제
# 2차원 배열을 1차원으로 만들어보세요.
# 2가지 방법.
print(a6)
print(a6.reshape(6))
print(a6.reshape(a6.size))
print(a6.reshape(-1))
print('-' * 50)

# b = list(range(5))
b = np.array(range(5))
print(b)

b += 1              # broadcasting
print(b)
print(b ** 2)
print(b > 3)
print(b[b > 3])
print(np.sin(b))    # universal function

c = np.arange(6).reshape(2, 3)
print(c)
print(c ** 2)
print(c > 3)
print(c[c > 3])
print(np.sin(c))
print('-' * 50)

#  2 1 0
# [][][]
print(sum(c))
print(np.sum(c))
print(np.sum(c, axis=0))    # 수직(열)
print(np.sum(c, axis=1))    # 수평(행)
print('-' * 50)

d1 = np.arange(3)
d2 = np.arange(3, 6)
print(d1, d2)
print(d1 + d2)              # vector operation

e = np.arange(12).reshape(-1, 4)
print(e)

# 문제
# 2차원 배열을 거꾸로 출력해보세요.
print(e[::-1])
print(e[::-1][::-1])        # fail.
print(e[::-1, ::-1])

print(e[0])
print(e[0][0], e[0, 0])     # fancy indexing
print(e[-1][-1], e[-1, -1])
print('-' * 50)

g1 = np.arange(3)
g2 = np.arange(6)
g3 = np.arange(3).reshape(1, 3)
g4 = np.arange(3).reshape(3, 1)
g5 = np.arange(6).reshape(2, 3)

# print(g1 + g2)    # error.
print(g1 + g3)      # vector
print(g1 + g4)      # broadcasting + broadcasting
print(g1 + g5)      # vector + broadcasting

# print(g2 + g3)    # error.
print(g2 + g4)
# print(g2 + g5)    # error.

print(g3 + g4)
print(g3 + g5)

print(g4 + g5)





print('\n\n\n\n\n\n\n')