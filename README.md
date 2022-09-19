# 유용한 팁들

### 알고리즘 종류

**브루트 포스 알고리즘**   
> - 모든 경우의 수를 전부 탐색함  

**그리디 알고리즘**  
> - 당장 눈앞에 보이는 최적의 상황을 골라서 최종 해답에 도달함

**동적 계획법**  
> - 복잡한 문제를 하위 문제로 나눠서 생각함   
> - 겹치는 하위 문제들은 메모리제이션을 통헤 빠르게 해를 얻어낼 수 있음   
> - Top Down과 Bottom Up 방식이 있음

**크루스칼 알고리즘(MST)**  
> - 사이클이 생기지 않는 선에서 가장 작은 가중치의 간선들을 선택해 나감   
> - 사이클 체크는 유니온 파인드 알고리즘을 사용함
> - [백준 1647번](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%801647.md)
> - https://chanhuiseok.github.io/posts/algo-33/

**유니온 파인드**   
> - 각각 인덱스를 부여한뒤, 합쳐질때 부모를 통일시킴
> - 부모를 재귀로 찾아갈 때, 거쳐간 애들의 부모도 

**프림 알고리즘(MST)**  
> - 현재 정점들과 가장 작은 가중치로 연결된 정점을 선택해나감   
> - https://www.weeklyps.com/entry/%ED%94%84%EB%A6%BC-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-Prims-algorithm

**다익스트라(특정 노드로부터 최소 거리 구하기)**   
> - 출발 지점으로부터 인접한 노드 거리 구하기 -> 최솟값이 나온 노드로부터 인접한 노드 거리 구하기 -> 기존 노드거리와 새로 계산한 노드 거리중 최솟값 선택  
> - heap을 쓰는 방법과 안쓰는 방법이 있음   
> - [백준 1238번](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%801238.md)
> - https://wooono.tistory.com/397   
> - https://brownbears.tistory.com/554   

**벨만포드(특정 노드로부터 최소 거리 구하기)**   
> - 각각의 간선 봐보면서 기존 노드 거리 + 간선 < 목적지 노드 거리 이면 업데이트 하기 -> 이걸 v-1번 반복   
> - 음수 간선에 적용 가능   
> - v-1번째에도 업데이트가 된다면 음수 사이클 존재하는 것임   
> - [백준 11675번](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%8011675.md) 
> - https://8iggy.tistory.com/153   

**플로이드-워셜(모든 노드간에 최소 거리 구하기)**   
> - 중간 지점(b) 하나씩 선택해가면서 a->b + b->c < a->c 인지 확인하고 업데이트   
> - 음수인 경우도 적용 가능   
> - https://chanhuiseok.github.io/posts/algo-50/   

**위상정렬**   
> - 진입차수가 0인 노드를 큐에 넣음 -> 큐에서 노드를 꺼내면 그 노드와 연결된 노드들의 진입차수를 1 줄임   
> - 비순환 방향그래프만 적용 가능   
> - https://velog.io/@kimdukbae/%EC%9C%84%EC%83%81-%EC%A0%95%EB%A0%AC-Topological-Sorting

**에라토스테네스의 체(소수 구하기)**  
> - 2부터 시작하여 숫자의 배수들을 지워나감



### DFS 및 BFS
- DFS
```
def dfs():
    while len(stack) != 0:              # 스택이 빌때까지
        selected_v = stack.pop()        # 한개를 뽑아내서
        if selected_v in done:
            continue                    # 만약 이미 한거라면 continue 하고
        done.append(selected_v)         # 안한거라면 했다는 표시를 해놓고
        for v in graph[selected_v]:
            stack.append(v)             # 해당 점점으로부터 이어지는 정점들 스택에 추가하기
```
- BFS
```
def bfs():
    while len(queue) != 0:              # 큐가 빌때까지
        selected_v = queue.popleft()    # 한개를 뽑아내서
        if selected_v in done:        
            continue                    # 만약 이미 한거라면 continue 하고
        done.append(selected_v)         # 안한거라면 했다는 표시를 해놓고
        for v in graph[selected_v]:     
            queue.append(v)             # 해당 정점으로부터 이어지는 정점들 큐에 추가하기
```

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
- 리스트 합치기
```
list1 += list2
list1.extend(list2)
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
lst = list(set(a) & set(b))
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
- 대문자, 소문자
```
upperStr = sentence.upper()
lowerStr = sentence.lower()
if upperStr.isupper()
if lowerStr.islower()
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
- 값 가져오기
```
val = dict.get('a')
val = dict.get('c', 0) // 있으면 값 반환, 없으면 0 반환
```
- for문 이용하기
```
for key in dict.keys()
for value in dict.values()
for key, value in dict.items()
```

### Set 다루기
- 선언
```
my_set = set([0, 4])
my_set = {0, 4}
```
- 추가하기
```
# 한개 추가
my_set.add(6)
# 여러개 추가
my_set.update([6, 7, 8])
```
- 삭제하기
```
# 값이 없으면 에러 발생
my_set.remove(6)
# 값이 없어도 에러 발생 안함
my_set.discard(6)
```
- Frozenset
```
# frozenset은 immutable한 버전의 set이다 (딕셔너리의 키로 사용 가능)
# 사용 방법
my_frozenset = frozenset(my_list)
my_frozenset = frozenset(my_set)
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

### Heap 다루기
- 선언
```
# 기본적으로 최소 힙
import heapq as hq
heap = [7, 3, 4] # 평범한 리스트
```
- 힙으로 만들기
```
# 반환하지 않고 인자로 넘긴 리스트 자체를 
hq.heapify(heap)
```
- 추가하기
```
# 첫번째는 최솟값이지만 두번째가 그 다음 최솟값이라는 보장은 없음
hq.heappush(heap, 1) # [1, 7, 3, 4] 
```
- 삭제하기
```
removed = hq.heappop(heap) # [3, 7, 4], removed=1
```
- 최대 힙
```
# 값 자체를 음수로 주기 
hq.heappush(heap, -value)
max_val = -hq.heappop(heap) 

# 맨 앞 값을 기준으로 정렬된다는 점을 이용하기
hq.heappush(heap, (-value, value))
max_val = hq.heappop(heap)[1]
```

### 기타
- 입력받는 방법
```
 # 첫번째 방법
import sys
sys.stdin.readline().strip() # 이게 더 빠르다

# 두번째 방법
input() 
```
- for문을 전부 돌면서 조건에 안걸리는 것을 찾고싶을 때 -> 일단 더해놓고 조건이 걸리면 빼기
```
cnt += 1
for i in range(10):
    if i == 5:
        cnt -= 1
        break
```
- 리스트에서 요소 개수 딕셔너리로 받기
```
from collections import Counter
lst = [a, a, b]
count_dict = Counter(lst)
```
- 여러 숫자가 입력될 때 각각 나눠서 저장하기
```
A, B = map(int, input().split())
```
- 세 꼭짓점의 좌표로 삼각형 넓이 구하기
<img width="400" alt="image" src="https://user-images.githubusercontent.com/72330884/184631017-e8c8734b-2995-4157-89b9-5957e134dc6a.png">

### 주의할 점
- 전역 변수는 왠만하면 피하고 return으로 결과값을 반환하자
- 방문한 노드를 저장할 때는 ```checked.append((n, node))``` 보다는 ```checked[n].append(node)``` 형태로 저장하자
- 간단한 코드상에서는 Python3가 메모리, 속도 측에서 우세할 수 있는 것이고, 복잡한 코드(반복)을 사용하는 경우에서는 PyPy3가 우세한 편이다
- 숫자가 무조건 1자리라고만 생각하지는 말자

