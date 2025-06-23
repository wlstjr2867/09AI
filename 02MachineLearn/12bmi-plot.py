import matplotlib.pyplot as plt
import pandas as pd

# pandas 로 csv 파일 읽어 들이기. 세번째 열을 행 인덱스로 지정함.
tbl = pd.read_csv("./resData/bmi.csv", index_col=2)

# 그래프 및 Axe 객체 생성
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# 서브 플롯 전용 - 지정한 레이블을 임의의 색으로 칠하기
def scatter(lbl, color):
    b = tbl.loc[lbl]
    ax.scatter(b["weight"], b["height"], c=color, label=lbl)

# 비만, 정상, 저체중 순으로 산점도 그리기
scatter("fat","red")
scatter("normal","yellow")
scatter("thin","purple")

# 범례표시, 그래프 저장
ax.legend()
plt.savefig("./saveFiles/bmi-scatter.png")
plt.show()