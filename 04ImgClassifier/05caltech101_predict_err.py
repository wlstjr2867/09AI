import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image
import numpy as np

categories = ["chair", "camera", "butterfly", "elephant", "flamingo"]
nb_classes = len(categories)

image_w = 64
image_h = 64

data = np.load("./saveFiles/caltech_5object.npz")
X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]

X_train = X_train.astype("float") / 256
X_test = X_test.astype("float") / 256
print('X_train shape:', X_train.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # 2D 특징맵을 1D 벡터로 변환
model.add(Dense(512))  # 완전연결층(512개의 뉴런)
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',  # 손실함수 지정
              optimizer='rmsprop',  # RMSprop 옵티마이저 사용
              metrics=['accuracy'])  # 평가 지표로 정확도 사용

# 모델 훈련 및 hdf5로 저장
hdf5_file = "./saveFiles/caltech_5object_model.hdf5"
if os.path.exists(hdf5_file):
    # 이미 존재하면 가중치 모델 로드
    model.load_weights(hdf5_file)
else:
    #없으면 학습 후 가중치 모델 저장
    model.fit(X_train, Y_train, batch_size=32, epochs=50)
    model.save_weights(hdf5_file)

# 모델 평가 : 테스트 데이터를 통해 모델 예측
pre = model.predict(X_test)
# 평가된 결과 데이터를 통해 반복
for i, v in enumerate(pre):
    # 모델이 예측한 클래스를 통해 가장 확률이 높은 인덱스
    pre_ans = v.argmax()
    # 실제 정답 클래스 중 가장 확률이 높은 인덱스
    ans = Y_test[i].argmax()
    # 원본 이미지 데이터
    dat = X_test[i]
    # 예측 정답이면 반복문의 처음으로 돌아간다.
    if ans == pre_ans: continue

    # 분류가 잘못된 경우 NG 메세지 출력
    print("[NG]", categories[pre_ans],  "!=", categories[ans])
    print(v)
    # 예측에 실패한 이미지 저장 경로 설정
    fname = "./predict_error/" + str(i) + "-" + categories[pre_ans] + \
        "-ne-" + categories[ans] + ".png"
    # 0~1 범위의 이미지 데이터를 0~255로 변환
    dat *= 256
    # 넘파이 배열을 이미지 객체로 변환 후..
    img = Image.fromarray(np.uint8(dat))
    # 저장한다.
    img.save(fname)

# 평가
score = model.evaluate(X_test, Y_test)
# 손실 출력
print('loss=', score[0])
# 정확도 출력
print('accuracy=', score[1])

