# Day_01_02_python.py

# collection : list, tuple, set, dictionary
#               []    ()          {}

a = [1, 3, 5, 'yes']
print(a)
print(a[0], a[1])
print(len(a))
print(a[3], a[len(a)-1], a[-1])

a.append(7)
# a.extend(9)   # error
a.extend([9])
a.append([9])

for i in range(len(a)):
    print(i, a[i])

for i in a:
    print(i)

a[0] = 123
a[-1] = 456

# 문제
# 리스트를 거꾸로 출력해보세요.
for i in reversed(range(len(a))):
    print(i, a[i])

for i in reversed(a):
    print(i)
print('-' * 50)

# tuple : 상수 버전의 리스트
t = (1, 3, 5)

for i in reversed(range(len(t))):
    print(i, t[i])

# t.append(7)       # error.
# t[-1] = 99        # error.

t = (1, 5)
t = 1, 5            # pack
print(t)

t1, t2 = 1, 5
print(t1, t2)

t3, t4 = t          # unpack
print(t3, t4)

# t3, t4, t5 = t    # error

def f_7(m, n):
    return m + n, m * n

t5, t6 = f_7(3, 8)
print(t5, t6)

t7 = f_7(3, 8)
print(t7)

t8, _ = f_7(3, 8)       # place holder
print(t8)
print('-' * 50)

# 영한 사전 : 영어 단어를 찾으면 한글 설명 나옴
# 영어 단어 : key
# 한글 설명 : value

#     key     value
d = {'name': 'hoon', 'age': 20}
d = dict(name='hoon', age=20)
print(d)
print(d['name'], d['age'])

k = 'korea'
d1 = {'name': 'hoon', 'age': 20, 3: 4, k: 'great'}
# d2 = dict(name='hoon', age=20, 3=4)
d2 = dict(name='hoon', age=20, k='great')

for key in d1:
    print(key, d1[key])

print(d1[k])
# print(d2[k])
print(d2['k'])
print('-' * 50)

def f_8(a, b, c):
    print(a, b, c)

f_8(1, 2, 3)            # positional argument
f_8(a=1, b=2, c=3)      # keyword argument
f_8(c=3, a=1, b=2)
f_8(1, 2, c=3)
# f_8(a=1, 2, c=3)      # positional은 keyword 앞에.

def f_9(*args):         # 가변인자, pack.
    print(args, *args)  # force unpack.

f_9()
f_9(12)
f_9(12, 'hello')

a = [1, 3, 5]
print(a, *a)

print(a, *a, end='===', sep='**')
print()
print('-' * 50)

b = list(range(10))
print(b)

print(b[3:7])       # slicing, range()

# 문제
# 앞쪽 절반을 출력해보세요.
# 뒤쪽 절반을 출력해보세요.
print(b[0:len(b)//2])
print(b[0:5])
print(b[:5])

print(b[len(b)//2:len(b)])
print(b[5:10])
print(b[5:])

# 문제
# 짝수 번째만 출력해보세요.
# 홀수 번째만 출력해보세요.
print(b[::2])
print(b[1::2])

# 문제
# 거꾸로 출력해보세요.
print(b[3:4])
print(b[3:3])
print(b[9:0:-1])
print(b[9:-1:-1])
print(b[-1:-1:-1])
print(b[::-1])      # 증감(양수 : 처음->마지막, 음수 : 마지막->처음)
