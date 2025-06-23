import urllib.request as req
import gzip, os, os.path

# 다운로드 시 진행률을 콘솔에 표시하는 함수
def progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    percent = (downloaded / total_size) * 100 if total_size > 0 else 0
    print(f"다운로드 진행률: {percent:.2f}%")

# 파일을 저장할 디렉토리 지정
savepath = "./resMnist"
# 다운로드 할 github URL
baseurl = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
# 다운로드 할 파일명을 List로 정리
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"]

# 디렉토리가 없으면 생성
if not os.path.exists(savepath): os.mkdir(savepath)

# 리스트로 선언한 각 파일을 다운로드
for f in files:
    # 다운로드 및 저장 경로 조립
    url = baseurl + "/" + f
    loc = savepath + "/" + f
    print("download:", url)
    # 경로에 파일이 있으면..
    if not os.path.exists(loc):
        #다운로드 진행. url로 부터 다운로드 후 loc에 저장함.
        req.urlretrieve(url, loc)
        # 진행률을 표시하려면 3번째 인수로 progress(함수명)를 추가

# GZip 파일 압축 해제
for f in files:
    gz_file = savepath + "/" + f
    raw_file = savepath + "/" + f.replace(".gz", "")
    print("gzip:", f)
    with gzip.open(gz_file, "rb") as fp:
        body = fp.read()
        with open(raw_file, "wb") as w:
            w.write(body)

print("ok")