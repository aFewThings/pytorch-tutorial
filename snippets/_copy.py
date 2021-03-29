import copy

a = [1, [1, 2, 3]]
b = copy.copy(a)     # shallow copy 리스트는 별도로 생성하지만 내부의 객체는 원래 객체와 동일함. (value는 그대로 복사, 내부 객체는 포인터를 참조)
c = copy.deepcopy(a) # deep copy 리스트를 별도로 생성, 내부 객체들도 새로 생성
d = a                # 단순 객체 복제 (리스트를 참조)

a[0] = 100
a[1].append(100)
print(a) # [100, [1, 2, 3, 100]]
print(b) # [1, [1, 2, 3, 100]] 
print(c) # [1, [1, 2, 3]]
print(d) # [100, [1, 2, 3, 100]]