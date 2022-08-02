# 유용한 팁들

### 여러 숫자가 입력될 때 각각 나눠서 저장하기
```
A, B = map(int, input().split())
```

### 조합 공식
<img width="129" alt="image" src="https://user-images.githubusercontent.com/72330884/182129405-0fffae97-6219-4fe2-b826-42c8890f5a00.png">
<img width="194" alt="image" src="https://user-images.githubusercontent.com/72330884/182129186-f754f915-c2bd-4144-bafd-50aa6eaf1f95.png">
<img width="293" alt="image" src="https://user-images.githubusercontent.com/72330884/182129267-e8359f82-e125-4a29-bebe-4eee39064c11.png">

### 정렬
```
list.sort() // 오름차순 자체 정렬
list.sort(reverse=True) // 내림차순 자체 정렬
new_list = sorted(list) // 오름차순 정렬 후 반환
new_list = sorted(list, reverse=True) // 내림차순 정렬 후 반환
new_list = sorted(list, key=lambda x: x[2]) // 특정 키 기준으로 정렬
new_list = sorted(list, key=lambda x: (x[2], x[1])) // 다중 키 기준으로 정렬
```

### 숫자 다루기
```
import math
a = math.inf // 양의 무한대
b = -math.inf // 음의 무한대
c = math.sqrt(49) // 제곱근 -> 7
d = math.pow(7, 2) // 제곱 -> 49
e = round(1.12345, 2) // 소수점 버리기 -> 1.12
```


### 리스트 다루기
```
lst = ['a', 'b', 'c', 'd']

// 리스트 요소들을 하나의 문자열로 합치기
lst = '_'.join(lst) // a_b_c_d

// 리스트에서 특정 요소의 인덱스 찾기
idx = lst.find('b') // 1 (값이 없을 땐 -1)
idx = lst.index('b') // 1 (값이 없을 땐 에러)
```
