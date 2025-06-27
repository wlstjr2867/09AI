import cv2

# 입력 파일 지정하기
image_file = "./resData/photo1.jpg"
# 캐스케이드 파일의 경로 지정하기
cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
'''
이 XML파일은 사전에 학습된 데이터를 기반으로 얼굴을 감지한다.
해당 경로에 들어가보면 얼굴뿐 아니라 눈, 입, 코등을 가지는 파일도 있다.
'''
print('cascade_file', cascade_file)
# exit를 사용하면 그 뒤에는 실행이 되지 않는다.`
# exit()

# 이미지 읽기. Numpy 배열 형태의 이미지 데이터로 저장
image = cv2.imread(image_file)
# 그레이스케일로 변환. 검출기의 성능 향상을 위해 컬러 정보를 제거하고,
# 발기 정보만을 사용한다.
image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 얼굴 인식 특징 파일 읽어 들이기
cascade = cv2.CascadeClassifier(cascade_file)
#얼굴 인식 실행
'''
scaleFactor = 이미지 크기를 10%씩 축소하면서 여러 크기의 얼굴을 감지.
    1. 1보다 크면 더 적은 얼굴을 감지하고, 그보다 작으면 더 많은 얼굴을
    감지하지만 오탐 가능성이 증가된다.
minNeighbors = 얼굴 후보가 검출될때 주변의 이웃 후보 개수를 설정.
`   값이 클수록 더 엄격한 기준으로 얼굴을 판단한다.
minSize = 검출할 최소 얼굴 크기 지정. 가로/세로 150px 보다 작은 얼굴은 무시한다.
'''
face_list = cascade.detectMultiScale(image_gs,
    scaleFactor=1.1,
    minNeighbors=1,
    minSize=(150, 150))

if len(face_list) > 0:
    # 인식한 부분 표시하기
    print(face_list)
    color = (0, 0, 255) # 빨간색
    # 얼굴 영역에 테두리 표시
    '''
    x,y : 얼굴의 좌상단 좌표
    w,h : 가로(폭), 세로(높이)의 길이
    thinkness : 선의 두께
    '''
    for face in face_list:
        x, y, w, h = face
        # 얼굴 위치에 사각형 테두리를 표시한다.
        cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=8)

    cv2.imwrite("./saveFiles/photo1-facedatect01.png", image)
else:
    print("얼굴을 인식할 수 없습니다.")