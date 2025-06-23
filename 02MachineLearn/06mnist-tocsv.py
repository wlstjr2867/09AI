import struct

# 다운로드 한 바이너리 파일을 csv파일로 변환하기 위한 함수 정의
def to_csv(name, maxdata):
    #파일열기
    # 레이블 파일 : 숫자의 실제값을 저장한 파일
    lbl_f = open("./resMnist/"+name+"-labels-idx1-ubyte", "rb")
    # 이미지 파일 : 손글씨 이미지 데이터를 저장한 파일
    img_f = open("./resMnist/"+name+"-images-idx3-ubyte", "rb")
    # 변환된 데이터를 저장할 csv 파일 경로 (쓰기모드)
    csv_f = open("./resMnist/"+name+".csv", "w", encoding="utf-8")

    # 헤더 정보 읽기(파일의 메타데이터)
    '''
    struct 모듈을 사용하여 바이너리 데이터를 읽고 정수를 변환한다.
    >II 는 빅엔디안 방식으로 4바이트 정수를 2개를 읽겠다는 의미로 사용된다.
    '''
    # 레이블과 이미지 파일에서 매직넘버와 아이템 갯수 읽기
    mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
    mag, img_count = struct.unpack(">II", img_f.read(8))
    # 이미지의 행과 열의 크기 읽기
    rows, cols = struct.unpack(">II", img_f.read(8))
    # 이미지의 픽셀 갯수 계산 (28*28)
    pixels = rows * cols

    # 이미지 데이터를 읽고 CSV로 저장
    res = []
    for idx in range(lbl_count):
        # 최대 데이터 갯수를 초과하면 반복문 탈출
        if idx > maxdata: break
        # 레이블(0~9)을 1바이트씩 읽어서 정수로 변환
        label = struct.unpack("B", lbl_f.read(1))[0]
        # 이미지 데이터를 픽셀 단위로 읽기
        bdata = img_f.read(pixels)
        # 각 픽셀값을 문자열로 변환 후 리스트에 저장
        sdata = list(map(lambda n: str(n), bdata))
        # csv 파일에 데이터 저장
        csv_f.write(str(label)+",") # 첫번째 컬럼에 레이블 저장
        csv_f.write(",".join(sdata)+"\r\n") # 픽셀 데이터 저장후 줄바꿈
        # 잘 저장되었는지 눈으로 확인하기 위해 일부 이미지를 PGM 포맷으로 저장
        if idx < 10: # 처음 10개만 저장한다.
            s = "P2 28 28 255\n" # PGM 헤더(ASCII 형식의 그레이스케일 이미지)
            s += " ".join(sdata) # 픽셀 데이터를 공백으로 구분하여 추가
            # 저장될 파일의 이름과 경로 지정
            iname = "./resMnist/{0}-{1}-{2}.pgm".format(name,idx,label)
            with open(iname, "w", encoding="utf-8") as f:
                f.write(s)
    # 자원해제
    csv_f.close()
    lbl_f.close()
    img_f.close()

# 바이너리 파일을 읽어 csv파일로 변환 실행
to_csv("train", 1000)
to_csv("t10k", 500)

to_csv("train", 70000)
to_csv("t10k", 20000)