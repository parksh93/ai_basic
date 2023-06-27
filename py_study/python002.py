import numpy as np

# 문제 1
print("makit \"code\" lab")
print('makit "code" lab')
print("she's gone")

# 문제 2
a = 10
b = 20
print("a의 값은 ", a)
print("b의 값은 ", b)
print("a와 b의 합은 ", a+b)

# 문제 3
a = 10
b = 'makit '
print(a * 3)
print(b * 3)

# 문제 4
a = ['메이킷', '우진', '시은']
print(a)
for i in a:
    print(i)

# 문제 5
a = ['메이킷', '우진','제임스', '시은']
print(a[:2])
print(a[1:])
print(a[2:])
print(a[0:4])

# 문제 6 - 7
a = ['우진', '시은']
b = ['메이킷', '소피아', '하워드']
print(a + b)
print(b)
print(b + a)
# 두개의 리스트를 붙임
print(b.extend(a))

# 문제 8 - 9
a = np.array([[1,2,3],[4,5,6]])
print('Original: \n',a)

a_transpose = np.transpose(a)
print('Transpose:\n',a_transpose)

a_reshape = np.reshape(a,(3,2))
print("Reshape:\n",a_reshape)
