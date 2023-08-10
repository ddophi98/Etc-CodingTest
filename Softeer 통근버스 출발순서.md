# 통근버스 출발 순서 검증하기 - 3단계

## 문제 
### https://softeer.ai/practice/info.do?idx=1&eid=654&sw_prbl_sbms_sn=248409
<img width="1143" alt="image" src="https://github.com/ddophi98/Etc-CodingTest/assets/72330884/3555e6f2-6fec-490e-b8c9-c02054335060">


## 해결 포인트
**시간복잡도 계산해보기**
- 시간 복잡도 먼저 계산해보고, 줄인다면 어떻게 동작해야하는지 생각해보기

**계산 순서 바꿔보기**
- 첫번째, 두번째 정해놓고 세번째 살펴보면 n*3
- 첫번째 정해놓고, 두번째 살펴보면서 세번째도 정하면 n*2
  

## 내 코드
```python
import sys

n = int(sys.stdin.readline().strip())
lst = list(map(int, sys.stdin.readline().rstrip().split()))

count = 0
for i in range(n):
    big = 0
    for j in range(i+1, n):
        # 큰 거는 중첩해서 개수 세어놓기
        if lst[i] < lst[j]:
            big += 1
        # 작은 거 나오는 순간 가능한 배열이 됨
        elif lst[i] > lst[j]:
            count += big

print(count)
```

## 참고자료
https://velog.io/@trillionaire/Softeer-%EC%9D%B8%EC%A6%9D%ED%8F%89%EA%B0%804%EC%B0%A8-%EA%B8%B0%EC%B6%9C-%ED%86%B5%EA%B7%BC%EB%B2%84%EC%8A%A4-%EC%B6%9C%EB%B0%9C-%EC%88%9C%EC%84%9C-%EA%B2%80%EC%A6%9D%ED%95%98%EA%B8%B0
