import random

# 리턴하는함수
def calc_bmi(h, w):
    bmi = w / (h/100) ** 2
    if bmi < 18.5: return "thin"
    if bmi < 25: return "normal"
    return "fat"

# 출력 파일 준비하기
fp = open("./resData/bmi.csv", "w", encoding="utf-8")
# 쓰기 모드로 오픈한 후 첫번쨎 줄에는 컬럼명 입력
fp.write("height,weight,label\r\n")

cnt = {"thin":0, "normal":0, "fat":0}
# 2만개의 데이터 반복
for i in range(20000):
    # 정해진 범위에서 키와 몸무게 랜덤 생성
    h = random.randint(120, 200)
    w = random.randint(35, 80)
    # BMI 계산
    label = calc_bmi(h, w)
    # 3가지의 라벨 중 1증가
    cnt[label] += 1
    # csv 파일에 기록
    fp.write("{0},{1},{2}\r\n".format(h, w, label))
# 모든 입력이 완료되면 파일 객체를 닫아준다.
fp.close()
print("ok,", cnt)