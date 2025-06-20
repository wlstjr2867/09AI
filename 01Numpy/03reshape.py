import numpy as np

'''
ndarray를 편하게 생성하기
    : 테스트용으로 데이터를 만들거나 대규모의 데이터를 일괄적으로 초기화
    하는 경우에 사용된다.
    
    arrange() : 파이썬의 표준함수인 range()와 유사하게 배열 생성
    ones() : 배열 생성 후 1로 초기화, 단 인자로 dtype을 정해주지 않으면
        float64 타입으로 초기화됨
    zeros() : ones()와 동일한 기능. 단 0으로 초기화됨
'''

print('arange()로 생성')
# 0~9로 구성된 배열 생성
sequence_array = np.arange(10)
print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

print('zeros()로 생성')
# 3행 2열인 정수타입의 2차원 배열 생성
zeros_array = np.zeros((3, 2), dtype='int32')
print(zeros_array)
print(zeros_array.dtype, zeros_array.shape)

print('ones()로 생성')
# 3행 2열인 배열 생성. 타입 지정이 없으므로 float64로 초기화.
one_array = np.ones((3,2))
print(one_array)
print(one_array.dtype, one_array.shape)

'''
reshape() : ndarray의 차원과 크기를 변경하는 함수
    -1을 인수로 사용하면 원래 ndarray와 호환되는 새로운 shape로 변환해줌.
    reshape(5, -1) 이라면 5행 x열의 배열로 변환된다.
'''

#--------------------------------------------------------------

array1 = np.arange(10)
print('array1:\n', array1)

# 2행 5열로 변경
arrayA1 = array1.reshape(2, 5)
print('arrayA1:\n', arrayA1)

# 5행 2열로 변경
arrayA2 = array1.reshape(5, 2)
print('arrayA2:\n', arrayA2)

# 변경이 불가능하면 에러 발생 (ValueError)
# arrayA3 = array1.reshape(4, 3)
# ------------------------------------------------------------

# -1을 사용하면 자동으로 행 혹은 3열이 맟춰진다
arrayB1 = array1.reshape(-1, 5)
print('arrayB1 shape', arrayB1.shape)

arrayB2 = array1.reshape(5, -1)
print('arrayB2 shape', arrayB2.shape)

# 변경이 불가능하면 에러 발생
# arrayB3 = array1.reshape(-1,4)
#----------------------------------------------

'''
tolist () : Numpy 배열을 리스트로 변환하는 함수.
    N차원의 배열을 사용하면 N차원 형태의 리스트가 반환된다.
'''

# 3차원 ndarray
# 0~7까지의 정수로 1차원 배열 생성
array2 = np.arange(8)
# 2개의 블록(높이)으로 구성된 2행 2열의 3차원 배열로 변환
arrayC1 = array2.reshape((2, 2, 2))
# 3차원 배열로 출력
print('arrayC1:\n', arrayC1)
# 3차원 형태의 리스트를 반환하여 출력
print('arrayC1:\n', arrayC1.tolist())

# 3차원 ndarray를 2차원 배열로 변환한다.
# x행1열로 변환하므로 8행 1열인 2차원 배열로 변환된다.
arrayC2 = arrayC1.reshape(-1, 1)
print('arrayC2:\n', arrayC2.tolist())
print('arrayC2 shape:', arrayC2.shape)

# 1차원 ndarray를 2차원으로 변환
arrayC3 = array2.reshape(-1, 1)
print('array2:\n', array2)
print('array2:\n', array2.shape)
print('arrayC3:\n', arrayC3.tolist())
print('arrayC3 shape:', arrayC3.shape)

