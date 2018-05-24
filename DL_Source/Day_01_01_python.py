# Day_01_01_python.py

# ctrl + /
# ctrl + shift + f10
# alt + 1
# alt + 4

a, b = 3, 5
print(a, b)

a, b = b, a
print(a, b)

print(12, 3.14, True, 'hello')
print(type(12), type(3.14), type(True), type('hello'))
# <class 'int'> <class 'float'> <class 'bool'> <class 'str'>

# 연산 : 산술, 관계, 논리
# 산술 : +  -  *  /  **  //  %
print(a + b)
print(a - b)
print(a * b)
print(a / b)        # 나눗셈(실수)
print(a ** b)       # 제곱
print(a // b)       # 나눗셈(정수, 몫)
print(a % b)        # 나머지

print('hello' + 'python')
# print('hello' * 'python')     # error
print('hello' * 3)
print('-' * 50)

# 관계 : >  >=  <  <=  ==  !=
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

print(10 <= a <= 19)
print('-' * 50)

# 논리 : and  or  not
print(True and True)
print(True and False)
print(False and True)
print(False and False)
print('-' * 50)

a = 3

if a % 2:
    print('홀수')
else:
    print('짝수')

if a < 0:
    print('음수')
elif a > 0:
    print('양수')
else:
    print('제로')

print('-' * 50)

# 0 1 2 3 4     0, 4, 1
# 0 2 4 6 8     0, 8, 2
i = 0
while i <= 4:
    print(i, end=' ')
    i += 1
print()

i = 0
while i < 5:
    print(i, end=' ')
    i += 1
print()

for i in range(0, 5, 1):
    print(i, end=' ')
print()

for i in range(0, 5):       # 시작, 종료
    print(i, end=' ')
print()

for i in range(5):          # 종료
    print(i, end=' ')
print()

# 문제
# 0~24의 사이의 정수를 거꾸로 출력해보세요.
for i in range(5, -1, -1):
    print(i, end=' ')
print()

for i in reversed(range(5)):
    print(i, end=' ')
print()
print('-' * 50)

# 매개변수 없고, 반환값 없고.
def f_1():
    print('f_1')

f_1()

# 매개변수 있고, 반환값 없고.
def f_2(c, d):
    print('f_2', c, d)

f_2(12, 3.14)

# 매개변수 없고, 반환값 있고.
def f_3():
    pass

a = f_3()
print(a)

def f_4():
    return 45

print(f_4())

# 매개변수 있고, 반환값 있고.
# 문제
# 2개의 정수에서 큰 숫자를 찾는 함수를 만드세요.
def f_5(g, h):
    # if g > h:
    #     return g
    # else:
    #     return h

    # if g > h:
    #     return g
    # return h

    if g > h:
        h = g
    return h

print(f_5(3, 5))
print(f_5(5, 3))

# 문제
# 4개의 정수에서 가장 큰 숫자를 찾는 함수를 만드세요.
def f_6(a1, a2, a3, a4):
    # if a1 < a2: a1 = a2
    # if a1 < a3: a1 = a3
    # if a1 < a4: a1 = a4
    # return a1

    # print('12', print('34'))

    # return f_5(f_5(a1, a2), f_5(a3, a4))
    return f_5(f_5(f_5(a1, a2), a3), a4)

print(f_6(1, 2, 3, 4))
print(f_6(4, 1, 2, 3))
print(f_6(3, 4, 1, 2))
print(f_6(2, 3, 4, 1))




