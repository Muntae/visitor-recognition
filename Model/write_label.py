
"""
숫자 데이터를 바이너리화 하여 bin 파일에 저장한다.
라벨데이터를 만들기 위해 사용.
"""

f=open('C:\\Users\\SOL\\PycharmProjects\\untitled1\\image_sol\\test_data\\test_label.bin','wb')
a=bytes([0])
b=bytes([1])
c=bytes([2])
d=bytes([3])

for i in range(1000):
    f.write(a)
for i in range(1000):
    f.write(b)
for i in range(1000):
    f.write(c)
for i in range(1000):
    f.write(d)



print (a)
print (b)
print (c)
print (d)
f.close()