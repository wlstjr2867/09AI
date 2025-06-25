import tensorflow as tf
import pandas as pd
import numpy as np

# BMI 데이터 로드한 후 데이터프레임으로 변환
csv = pd.read_csv("./resData/bmi.csv")

# 데이터 정규화 후 각 컬럼에 즉시 적용
csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100

# label 컬럼의 체형정보를 원-핫 인코딩 형태로 변환(딕셔너리로 정의)
bclass = {"thin": [1, 0 , 0], "normal": [0, 1, 0], "fat": [0, 0, 1]}
# 변환한 데이터로 새로운 컬럼 생성
csv["label_pat"] = csv["label"].apply(lambda x: np.array(bclass[x]))
# 데이터 상위 5개 확인하기
print(csv.head())

# 테스트 데이터 분리
test_csv = csv[15000:20000]
# 테스트용 입력 데이터(몸무게, 키)
test_pat = np.array(test_csv[["weight", "height"]])
# 테스트용 정답 레이블(원-핫 인코딩)
test_ans = np.array(list(test_csv["label_pat"]))

# 신경망 모델 정의
'''
입력레이어 : 키와 몸무게 2개의 입력값을 사용
출력레이어 : 3개의 클래스(thin, noraml, fat)사용
'''
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

#
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습 데이터 준비
train_pat = np.array(csv[["weight", "height"]])
train_ans = np.array(list(csv["label_pat"]))

# 학습 진행
'''
    epochs = 데이터셋 전체를 35번 반복 학습 
    batch_size= 학습 중 한번에 사용할 데이터의 크기 지정
    validation_data= 테스트  데이터를 사용해 매 epochs마다 모델의 성능을 검증
    verbose= 훈련 진행 상황을 시작정으로 표시
'''
history = model.fit(
    train_pat, train_ans,
    epochs = 35,
    batch_size=100,
    validation_data=(test_pat, test_ans), # 테스트 데이터 검증
    verbose=1
)

# 최종 정확도 출력 : 손실 값과 정확도를 반환
test_loss, test_acc = model.evaluate(test_pat, test_ans)
print("정답률 =", test_acc)