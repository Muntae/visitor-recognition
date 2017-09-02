# cifar10_sol
cifar10 예제를 텐서보드를 활용하여 직관적인 모델의 정확도를 보여주기 위해 수정함.

기존코드와 다른점

cifar10 에서 제공하는 데이터 뿐 아니라 내가 직접 만든 데이터도 추가로 넣고 모델을 만들기 위해서다.(추가데이터 : 사람)

1. cifar10 data size 와 동일하게 32*32 크기의 원하는 종류의 jpg 파일들을 준비한다.

2. 기존의 bin 파일로 되어있는 cifar10 dataset  을  받은 후 jpg  파일로 변환하여 저장한다. (change_data.py)

3. cifar10 data + 내 데이터(사람) 에 대한 label 파일을 새로 작성해준다.  (write_label.py)

4. 이미지파일, 라벨파일 에 대해서 각각 filename queue 를 생성한다. ( 이미지와 라벨 매칭을 위해 shuffle=false 옵션 추가)

5. 모델을 학습한다.

6. 학습된 모델을 cifar10_eval.py 를 통해 확인하는데 직관적인 결과를 보기 위해 매번 step마다(batch 단위) 의 이미지들을 텐서보드에 저장한다. 해당 이미지에 대한 추론 결과는 print 를 통해 출력된다. 

7. 추론 결과와 실제 이미지(텐서보드에서 확인) 을 통해 어떤 이미지가 어떻게 분류되는지 확인할 수 있다.

사용한 이미지 : https://drive.google.com/file/d/0B2cKT79gKYZrRXlJTmlSZXluVFE/view?usp=sharing
