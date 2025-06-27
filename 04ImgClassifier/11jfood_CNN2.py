from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import  utils
import numpy as np

# 분류 대상 카테고리
# root_dir = "./download"
categories = ["Gyudon", "Ramen", "Sushi", "Okonomiyaki", "Karaage"]
# 카테고리 갯수
nb_classes = len(categories)
# 이미지 크기
image_size = 100

def main():
    data = np.load("./saveFiles/japanese_food_aug.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    Y_train = data["Y_train"]
    Y_test = data["Y_test"]

    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256

    Y_train = utils.to_categorical(Y_train, nb_classes)
    Y_test = utils.to_categorical(Y_test, nb_classes)

    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

def build_model(in_shape):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
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
    return model

def model_train(X, Y):
    model = build_model(X.shape[1:])
    model.fit(X, Y, batch_size=32, epochs=30)
    hdf5_file = "./saveFiles/japanese_food_aug_model.hdf5"
    model.save_weights(hdf5_file)
    return model

def model_eval(model, X, Y):
    # 평가
    score = model.evaluate(X, Y)
    # 손실 출력
    print('loss=', score[0])
    # 정확도 출력
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()