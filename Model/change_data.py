import cv2
import numpy as np

"""
cifar-10 예제에서 bird, cat, dog 만 추출하는 코드
저장된 bin 경로에 맞게끔 f=open('경로',rb) 로 설정해주면 된다. 
"""

f = open("C:\\Users\\SOL\\PycharmProjects\\untitled1\\cifar10_data\\cifar-10-batches-bin\\test_batch.bin", 'rb')
cifar10classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
step2=0
while True: # while 1회당 이미지 1개 씩 read
    label = []
    test_image=np.arange(3072).reshape(32,32,3)

    #label read
    s = f.read(1)
    label.append(s)
    num=int(ord(s))

    #image read r1024 g 1024 b 1024 한 후 bgr 로 저장
    for i in range(32):
        for j in range(32):
            s = f.read(1)
            r=int(ord(s))
            test_image[i][j][2]=r

            j += 1
        i += 1

    for i in range(32):
        for j in range(32):

            s = f.read(1)
            g=int(ord(s))
            test_image[i][j][1]=g

            j += 1
        i += 1


    for i in range(32):
        for j in range(32):
            s = f.read(1)
            b=int(ord(s))
            test_image[i][j][0]=b
            j += 1
        i += 1


            # print ('test',test_image)
    if (num == 2):  # bird
        cv2.imwrite('C:\\Users\\SOL\\PycharmProjects\\untitled1\\image_sol\\%s\\image6_%d.jpg' % (cifar10classes[num], step2), test_image)
    if (num == 3):  # cat
        cv2.imwrite('C:\\Users\\SOL\\PycharmProjects\\untitled1\\image_sol\\%s\\image6_%d.jpg' % (cifar10classes[num], step2), test_image)
    if (num == 5):  # dog
        cv2.imwrite('C:\\Users\\SOL\\PycharmProjects\\untitled1\\image_sol\\%s\\image6_%d.jpg' % (cifar10classes[num], step2), test_image)

    step2+=1
    if s == '': break
f.close()