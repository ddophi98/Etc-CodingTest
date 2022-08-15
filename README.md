# 유용한 팁들

### 알고리즘 종류
- **브루트 포스 알고리즘**   
모든 경우의 수를 전부 탐색함
- **그리디 알고리즘**  
당장 눈앞에 보이는 최적의 상황을 골라서 최종 해답에 도달함
- **동적 계획법**  
복잡한 문제를 하위 문제로 나눠서 생각함   
겹치는 하위 문제들은 메모리제이션을 통헤 빠르게 해를 얻어낼 수 있음   
Top Down과 Bottom Up 방식이 있음
- **크루스칼 알고리즘(MST)**  
사이클이 생기지 않는 선에서 가장 작은 가중치의 간선들을 선택해 나감   
사이클 체크는 각 정점의 아이디를 부여한 후 연결되면 하나로 통일시킴
- **프림 알고리즘(MST)**  
현재 정점들과 가장 작은 가중치로 연결된 정점을 선택해나감
- **에라토스테네스의 체(소수 구하기)**  
2부터 시작하여 숫자의 배수들을 지워나감


### 순열 및 조합
```
import itertools
p_list = itertools.permutations(arr, 2) # 길이가 2인 순열
c_list = itertools.combinations(arr, 2) # 길이가 2인 조합
```
<img width="129" alt="image" src="https://user-images.githubusercontent.com/72330884/182129405-0fffae97-6219-4fe2-b826-42c8890f5a00.png">
<img width="194" alt="image" src="https://user-images.githubusercontent.com/72330884/182129186-f754f915-c2bd-4144-bafd-50aa6eaf1f95.png">
<img width="293" alt="image" src="https://user-images.githubusercontent.com/72330884/182129267-e8359f82-e125-4a29-bebe-4eee39064c11.png">

### 정렬
```
list.sort() # 오름차순 자체 정렬
list.sort(reverse=True) # 내림차순 자체 정렬
new_list = sorted(list) # 오름차순 정렬 후 반환
new_list = sorted(list, reverse=True) # 내림차순 정렬 후 반환
new_list = sorted(list, key=lambda x: x[2]) # 특정 키 기준으로 정렬
new_list = sorted(list, key=lambda x: (x[2], x[1])) # 다중 키 기준으로 정렬
```

### 숫자 다루기
```
import math
a = math.inf # 양의 무한대
b = -math.inf # 음의 무한대
c = math.sqrt(49) # 제곱근 -> 7
d = math.pow(7, 2) # 제곱 -> 49
e = round(1.12345, 2) # 소수점 반올림 -> 1.12
f = math.ceil(1.34) # 소수점 올림 -> 2
g = math.floor(1.34) # 소수점 내림 -> 1

import sys
h = sys.maxsize # 최대 정수값
```


### List 다루기
```
lst = ['a', 'b', 'c', 'd']
```
- 리스트에 요소 추가하기
```
lst.append('e') # ['a', 'b', 'c', 'd', 'e']
lst.insert(1, 'g') # ['a', 'g', 'b', 'c', 'd']
```
- 리스트 요소 삭제하기
```
del lst[1] # ['a', 'c', 'd']
removed = lst.pop(1) # ['a', 'c', 'd'], removed='b'
lst.remove('b') # ['a', 'c', 'd']
```
- 리스트 특정 요소 개수 세기
```
num = lst.count('a')
```
- 리스트 거꾸로 하기
```
lst = reversed(lst)
```
- 리스트 요소들을 하나의 문자열로 합치기
```
lst = '_'.join(lst) # a_b_c_d
```
- 리스트에서 특정 요소의 인덱스 찾기
```
idx = lst.find('b') # 1 (값이 없을 땐 -1)
idx = lst.index('b') # 1 (값이 없을 땐 에러)
```
- 리스트에서 교집합 찾기
```
lst = list(set(a_lst) & set(b)_lst)
```
- 얕은 복사
```
a = [1, 2, 3]
b = a
a[0] = 4
print(b) # [4, 2, 3]
```
- 깊은 복사
```
a = [1, 2, 3]
b = a[:]
a[0] = 4
print(b) # [1, 2, 3]

--- or ---

import copy
b = copy.deepcopy(a)
```
- 두 리스트 묶기
```
a = [1, 1, 1]
b = [2, 2, 2]
c = list(zip(a, b)) # [(1,2), (1,2), (1,2)]
```

### 문자열 다루기
- 특정 문자 바꾸기
```
sentence.replace('is', 'are') # is를 are로 바꾸기
```
- 특정 문자로 시작 또는 끝나는지 확인하기
```
if sentence.startswith('wo')
if sentence.endswith('rd')

sentence.startswith('wo', 2) # 두번째 인자는 찾기 시작할 지점
```

### Dictionary 다루기
- 선언
```
dict = {'a': 1, 'b': 2}
```
- 추가하기
```
dict['c'] = 3
```
- 삭제하기
```
del dict['c']
```
- 존재 여부 확인
```
if 'c' in dict:
    print("딕셔너리에 존재함")
```
- for문 이용하기
```
for key in dict.keys()
for value in dict.values()
for key, value in dict.items()
```

### Deque 다루기
- 선언
```
from collections import deque
queue = deque([4, 5, 6])
```
- 추가하기
```
queue.append(7) # [4, 5, 6, 7]
queue.appendleft(3) # [3, 4, 5, 6, 7]
```
- 삭제하기
```
removed = queue.popleft() # [4, 5, 6, 7], removed=3
removed = queue.pop() # [4, 5, 6], removed=7
```

### 기타
- for문을 전부 돌면서 조건에 안걸리는 것을 찾고싶을 때 -> 일단 더해놓고 조건이 걸리면 빼기
```
cnt += 1
for i in range(10):
    if i == 5:
        cnt -= 1
        break
```
- 여러 숫자가 입력될 때 각각 나눠서 저장하기
```
A, B = map(int, input().split())
```

