# Day_03_08_comprehension.py
import random

# 컴프리헨션 : 집계 함수에 사용할 리스트를 만드는 한 줄짜리 반복문

for i in range(5):
    i

for i in range(5):
    if i % 2:
        i

for i in range(5):
    for j in range(5):
        j

[i for i in range(5)]
(i for i in range(5))
{i for i in range(5)}

a = [i for i in range(5)]
print(a)
print([i for i in a])
print([0 for i in a])
print([[0] for i in a])

random.seed(1)
a1 = [random.randrange(100) for _ in range(10)]
a2 = [random.randrange(100) for _ in range(10)]
a3 = [random.randrange(100) for _ in range(10)]
print(a1)
print([i for i in a1 if i % 2])
print(sum([i for i in a1 if i % 2]))

# 문제
# 2차원 리스트의 합계를 구하세요.
b = [a1, a2, a3]
print([0 for i in b])
print([sum(i) for i in b])
print(sum([sum(i) for i in b]))

# 문제
# 2차원 리스트를 1차원으로 변경해보세요.
print([j for i in b for j in i])
print([j for i in b for j in i if j % 2])
