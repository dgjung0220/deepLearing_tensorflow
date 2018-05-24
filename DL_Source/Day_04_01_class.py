# Day_04_01_class.py


class Info:
    # name = 'hoon'

    def __init__(self, name):
        print('Info')
        self.name = name

    def show(self):
        print('show', self.name)

    @property
    def my_name(self):
        return self.name


i1 = Info('hoon')
i2 = Info('min')
print(i1)

# i1.name = 'hoon'
# i2.name = 'min'

# 문제
# show 함수를 호출해보세요.
# show(12)
# Info.show(12)
Info.show(i1)       # unbound method
i1.show()           # bound method
i2.show()

# print(i1.my_name())
print(i1.my_name)
print(i1.name)

# i1.my_name = 'sun'
i1.name = 'sun'



