# 코딩 테스트 준비
## 목차
- [알고리즘 종류](#알고리즘-종류)   
- [DFS 및 BFS](#dfs-및-bfs)   
- [순열 및 조합](#순열-및-조합)  
- [이진법](#이진법)
- [정렬](#정렬)   
- [숫자 다루기](#숫자-다루기)   
- [List 다루기](#list-다루기)   
- [문자열 다루기](#문자열-다루기)   
- [Dictionary 다루기](#dictionary-다루기)   
- [Set 다루기](#set-다루기)   
- [Deque 다루기](#deque-다루기)   
- [Heap 다루기](#heap-다루기)   
- [정규 표현식](#정규-표현식)   
- [기타](#기타)   
- [주의할 점](#주의할-점)   


## 알고리즘 종류

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
> - [백준 1719번](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%801719.md)
> - https://chanhuiseok.github.io/posts/algo-50/   

**위상정렬**   
> - 진입차수가 0인 노드를 큐에 넣음 -> 큐에서 노드를 꺼내면 그 노드와 연결된 노드들의 진입차수를 1 줄임   
> - 비순환 방향그래프만 적용 가능   
> - [백준 2252번](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%802252.md)
> - https://velog.io/@kimdukbae/%EC%9C%84%EC%83%81-%EC%A0%95%EB%A0%AC-Topological-Sorting

**에라토스테네스의 체(소수 구하기)**  
> - 2부터 시작하여 숫자의 배수들을 지워나감
> - [백준 1644번](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%801644.md)



## DFS 및 BFS
- DFS
``` python
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
``` python
def bfs():
    while len(queue) != 0:              # 큐가 빌때까지
        selected_v = queue.popleft()    # 한개를 뽑아내서
        if selected_v in done:        
            continue                    # 만약 이미 한거라면 continue 하고
        done.append(selected_v)         # 안한거라면 했다는 표시를 해놓고
        for v in graph[selected_v]:     
            queue.append(v)             # 해당 정점으로부터 이어지는 정점들 큐에 추가하기
```

## 순열 및 조합
- itertool 사용 O
``` python
import itertools
p_list = itertools.permutations(arr, 2) # 길이가 2인 순열
c_list = itertools.combinations(arr, 2) # 길이가 2인 조합
```
- itertool 사용 X 

<img width="400" alt="image" src="https://user-images.githubusercontent.com/72330884/192279911-79e8d22a-5cfc-45bd-8188-27908600415c.png">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/72330884/192279965-245d574b-c98a-412e-82f7-78ae3872a8a9.png">

``` python
# 조합
def combination(arr, remain):
    if remain == 0:
        return [[]]
    
    comb_lst = []
    for i in range(len(arr)):
        for rst in combination(arr[i+1:], remain-1):
            comb_lst.append([arr[i]] + rst)

    return comb_lst
``` 
``` python
# 순열
def permutation(arr, remain):
    if remain == 0:
        return [[]]

    perm_lst = []
    for i in range(len(arr)):
        for rst in permutation(arr[:i] + arr[i+1:], remain-1):
            perm_lst.append([arr[i]] + rst)
            
    return perm_lst
``` 

<img width="129" alt="image" src="https://user-images.githubusercontent.com/72330884/182129405-0fffae97-6219-4fe2-b826-42c8890f5a00.png">
<img width="194" alt="image" src="https://user-images.githubusercontent.com/72330884/182129186-f754f915-c2bd-4144-bafd-50aa6eaf1f95.png">
<img width="293" alt="image" src="https://user-images.githubusercontent.com/72330884/182129267-e8359f82-e125-4a29-bebe-4eee39064c11.png">

## 이진법
- 함수 사용 O
``` python
num = 6
bin_num = bin(num)[2:]
print(bin_num) # "110"
``` 

- 함수 사용 X
``` python
num = 6
bin_num = ''
while num != 0:
    bin_num += str(num % 2)
    num = num // 2
bin_num = bin_num[::-1]
print(bin_num) # "110"
```

## 정렬
``` python
list.sort() # 오름차순 자체 정렬
list.sort(reverse=True) # 내림차순 자체 정렬
new_list = sorted(list) # 오름차순 정렬 후 반환
new_list = sorted(list, reverse=True) # 내림차순 정렬 후 반환
new_list = sorted(list, key=lambda x: x[2]) # 특정 키 기준으로 정렬
new_list = sorted(list, key=lambda x: (x[2], x[1])) # 다중 키 기준으로 정렬
```

## 숫자 다루기
``` python
import math
a = math.inf # 양의 무한대
a = float('inf') # 양의 무한대
b = -math.inf # 음의 무한대
b = float('-inf') # 음의 무한대
c = math.sqrt(49) # 제곱근 -> 7
c = 49**0.5 # 제곱근 -> 7
d = math.pow(7, 2) # 제곱 -> 49
d = 7**2 # 제곱 -> 49
e = 9//2 # 몫 -> 4
f = 9%2 # 나머지 -> 1
e = round(1.12345, 2) # 소수점 반올림 -> 1.12
f = math.ceil(1.34) # 소수점 올림 -> 2
g = math.floor(1.34) # 소수점 내림 -> 1 (-1.34 -> -2)
h = math.trunc(1.34) # 소수점 버림 -> 1 (-1.34 -> -1)

import sys
i = sys.maxsize # 최대 정수값

print('%.1f' %(1.34)) # 소수점 첫째자리까지만 나타내기 -> 1.3
```


## List 다루기
``` python
lst = ['a', 'b', 'c', 'd']
```
- 리스트에 요소 추가하기
``` python
lst.append('e') # ['a', 'b', 'c', 'd', 'e']
lst.insert(1, 'g') # ['a', 'g', 'b', 'c', 'd']
```
- 리스트 요소 삭제하기
``` python
del lst[1] # ['a', 'c', 'd']
removed = lst.pop(1) # ['a', 'c', 'd'], removed='b'
lst.remove('b') # ['a', 'c', 'd']
```
- 리스트 합치기
``` python
list1 += list2
list1.extend(list2)
```
- 리스트 특정 요소 개수 세기
``` python
num = lst.count('a')
```
- 리스트 거꾸로 하기
``` python
lst = reversed(lst)
lst = lst[::-1]
```
- 리스트 요소들을 하나의 문자열로 합치기
``` python
lst = '_'.join(lst) # a_b_c_d
```
- 리스트에서 특정 요소의 인덱스 찾기
``` python
idx = lst.find('b') # 1 (값이 없을 땐 -1)
idx = lst.index('b') # 1 (값이 없을 땐 에러)
```
- 얕은 복사
``` python
a = [1, 2, 3]
b = a
a[0] = 4
print(b) # [4, 2, 3]
```
- 깊은 복사
``` python
a = [1, 2, 3]
b = a[:]
a[0] = 4
print(b) # [1, 2, 3]

--- or ---

import copy
b = copy.deepcopy(a)
```
- 두 리스트 묶기
``` python
a = [1, 1, 1]
b = [2, 2, 2]
c = list(zip(a, b)) # [(1,2), (1,2), (1,2)]
```
- 리스트에서 요소 개수 딕셔너리로 받기
``` python
from collections import Counter
lst = [a, a, b]
count_dict = Counter(lst) # {'a': 2, 'b': 1}
```

## 문자열 다루기
- 특정 문자 바꾸기
``` python
sentence.replace('is', 'are') # is를 are로 바꾸기
```
- 특정 문자로 시작 또는 끝나는지 확인하기
``` python
if sentence.startswith('wo')
if sentence.endswith('rd')

sentence.startswith('wo', 2) # 두번째 인자는 찾기 시작할 지점
```
- 대문자, 소문자
``` python
upperStr = sentence.upper()
lowerStr = sentence.lower()
if upperStr.isupper()
if lowerStr.islower()
```

### Dictionary 다루기
- 선언
``` python
dict = {'a': 1, 'b': 2}
```
- 추가하기
``` python
dict['c'] = 3
```
- 삭제하기
``` python
del dict['c']
```
- 존재 여부 확인
``` python
if 'c' in dict:
    print("딕셔너리에 존재함")
```
- 값 가져오기
``` python
val = dict['a']
val = dict.get('a')
val = dict.get('a', 0) // 있으면 값 반환, 없으면 0 반환
```
- for문 이용하기
``` python
for key in dict.keys()
for value in dict.values()
for key, value in dict.items()
```

## Set 다루기
- 선언
``` python
my_set = set([0, 4])
my_set = {0, 4}
```
- 추가하기
``` python
# 한개 추가
my_set.add(6)
# 여러개 추가
my_set.update([6, 7, 8])
```
- 삭제하기
``` python
# 값이 없으면 에러 발생
my_set.remove(6)
# 값이 없어도 에러 발생 안함
my_set.discard(6)
```
- 집합 연산
``` python
s1 = {1, 2, 3, 4}
s2 = {3, 4, 5}

# 합집합
s1 | s2  # {1, 2, 3, 4, 5}
s1.union(s2)  # {1, 2, 3, 4, 5}

# 교집합
s1 & s2  # {3, 4}
s1.intersection(s2)  # {3, 4}

# 차집합
s1 - s2  # {1, 2}
s1.difference(s2)  # {1, 2}
```
- Frozenset
``` python
# frozenset은 immutable한 버전의 set이다 (딕셔너리의 키로 사용 가능)
# 사용 방법
my_frozenset = frozenset(my_list)
my_frozenset = frozenset(my_set)
```


## Deque 다루기
- 선언
``` python
from collections import deque
queue = deque([4, 5, 6])
```
- 추가하기
``` python
queue.append(7) # [4, 5, 6, 7]
queue.appendleft(3) # [3, 4, 5, 6, 7]
```
- 삭제하기
``` python
removed = queue.popleft() # [4, 5, 6, 7], removed=3
removed = queue.pop() # [4, 5, 6], removed=7
```

## Heap 다루기
- 선언
``` python
# 기본적으로 최소 힙
import heapq as hq
heap = [7, 3, 4] # 평범한 리스트
```
- 힙으로 만들기
``` python
# 반환하지 않고 인자로 넘긴 리스트 자체를 
hq.heapify(heap)
```
- 추가하기
``` python
# 첫번째는 최솟값이지만 두번째가 그 다음 최솟값이라는 보장은 없음
hq.heappush(heap, 1) # [1, 7, 3, 4] 
```
- 삭제하기
``` python
removed = hq.heappop(heap) # [3, 7, 4], removed=1
```
- 최대 힙
``` python
# 값 자체를 음수로 주기 
hq.heappush(heap, -value)
max_val = -hq.heappop(heap) 

# 맨 앞 값을 기준으로 정렬된다는 점을 이용하기
hq.heappush(heap, (-value, value))
max_val = hq.heappop(heap)[1]
```

## 정규 표현식
- 특정 문자중에 매치
``` python
a|b|c # a,b,c 중 하나와 매치
[abc] # a,b,c 중 하나와 매치
[a-z] # a부터 z까지 중 하나와 매치
[a-zA-Z] # 알파벳과 매치
[^a-zA-Z] # 알파벳을 제외한 문자와 매치
[0-9] # 숫자와 매치
\s # 공백
\S # 공백을 제외한 문자

# 주의할 점
[(ab)c] # ab와 c중에 선택하는게 아니라 a,b,c,(,) 중에 선택하는 것이다
a(b # 괄호 문자 자체를 찾고 싶다면 a\(b로 써야한다
```
- 아무 문자 중에 매치
``` python
a.b # acb, a1b와 다 매치됨 (\n은 제외)
a.*b # a1010b, ab와 다 매치됨
a[.]b # a.b와 매치됨
```
- 반복되는 문자
``` python
ca*t # 0개 이상 반복
ca+t # 1개 이상 반복
ca?t # 0개거나 1개거나
ca{2}t # 2번 반복
ca{2,4}t # 2개 이상 4개 이하 반복
```
- 파이썬에서 쓰는법
``` python
import re
pattern = re.compile('[a-z]+')

# or

rst = re.match('[a-z]+', 'python is the best')
```
``` python
pattern = re.compile('\s') # \s가 공백으로 변해버림
pattern = re.compile(r'\s') # \s로 컴파일 됨
pattern = re.compile('\\s') # \s로 컴파일 됨
```
``` python
# 문자열의 처음부터 매칭되는지 확인
# match 객체 반환
# if rst / else 로 사용 가능
rst = pattern.match('python is the best')
```
``` python
# 문자열의 처음이 아니더라도 중간부터 매칭되는지 확인
# match 객체 반환
# if rst / else 로 사용 가능
rst = pattern.search('python is the best')
```
``` python
# 매치되는 문자열들을 match객체의 iterator로 반환
rst = pattern.finditer('python is the best')
```
``` python
# 매치되는 문자열들을 리스트로 반환
rst = pattern.findall('python is the best')
```
``` python
# 매치되는 문자열들을 다른 문자열로 변환한 뒤 전체 문자열 반환
rst = pattern.sub('after', 'before')
```
``` python
# 전체 문자열에서 매치되는 패턴 문자를 기준으로 양 옆 문자를 나눠버린 리스트 반환
rst = pattern.split('python is the best')
```

``` python
m = (match 객체)
m.group() # 문자열
m.group(1) # 첫번째 괄호 안 문자열
m.start() # 시작 지점
m.end() # 끝 지점+1
m.span() # (시작 지점, 끝 지점+1)
```

``` python
# 예제1
m = re.search('([0-9]{4})-([0-9]{2})-([0-9]{2})', '오늘은 2023-04-02 입니다.')
m.group() / m.group(0) # 2023-04-02
m.group(1) # 2023
m.group(2) # 04
m.group(3) # 02

# 예제2
new_sentence = re.sub('apple|orange', 'fruit', 'apple box orange tree') # fruit box fruit tree 

# 예제 3
split_list = re.split(r'[: ,]', 'apple orange:banana,tomato') # ['apple', 'orange', 'banana', 'tomato']
```

## 기타
- 입력받는 방법
``` python
 # 첫번째 방법
import sys
sys.stdin.readline().strip() # 이게 더 빠르다

# 두번째 방법
input() 
```
- for문을 전부 돌면서 조건에 안걸리는 것을 찾고싶을 때 -> 일단 더해놓고 조건이 걸리면 빼기
``` python
cnt += 1
for i in range(10):
    if i == 5:
        cnt -= 1
        break
```
- 소수인지 판별하기
``` python
def is_prime(num):
    if num < 2:
        return False

    for i in range(2, int(num**0.5)+1):
        if num % i == 0:
            return False
    return True
```
- 여러 숫자가 입력될 때 각각 나눠서 저장하기
``` python
A, B = map(int, input().split())
```
- 재귀한도 늘리기
``` python
import sys
sys.setrecursionlimit(10**6) # 기본 1000으로 되어있음
```
- 여러줄에 걸쳐서 코드 쓰기
``` python
# 첫번째 방법
if (a and
    b and
    c):
    return True

# 두번째 방법
if a and \
   b and \
   c:
   return True
```

- 이차원 배열에서 최댓값 구하기
``` python
max(map(max, dp))
```

- 세 꼭짓점의 좌표로 삼각형 넓이 구하기
<img width="400" alt="image" src="https://user-images.githubusercontent.com/72330884/184631017-e8c8734b-2995-4157-89b9-5957e134dc6a.png">

## 주의할 점
- 전역 변수는 왠만하면 피하고 return으로 결과값을 반환하자
- 방문한 노드를 저장할 때는 ```checked.append((n, node))``` 보다는 ```checked[n].append(node)``` 형태로 저장하자
- 간단한 코드상에서는 Python3가 메모리, 속도 측에서 우세할 수 있는 것이고, 복잡한 코드(반복)을 사용하는 경우에서는 PyPy3가 우세한 편이다
- 숫자가 무조건 한자리라고만 생각하지는 말자
- ```a in lst```로 확인하기보다는 ```dict[a] = False``` 같은걸로 확인하는게 빠르다
- 그리고 만약 가능하다면 ```visited = [[False for _ in range(w)] for _ in range(h)]``` 처럼 visited도 graph와 같은 형태로 정의하고 체크하는게 훨씬 빠르다
- 투포인터는 ```l, r = 0, 0``` 에서 시작할수도 있고, ```l, r = 0, n``` 에서 시작할 수도 있다
- if else 문에서 ```중첩 if```와 ```if 조건1 and 조건2``` 는 다르다는 것을 기억하자
- 시간효율이 별로 중요하지 않은 문제라면, 시간복잡도가 크더라도 코드를 짧게 써서 푸는 것에 집중해보는 것도 괜찮다 [(기둥과 보)](https://school.programmers.co.kr/learn/courses/30/lessons/60061)
- 흔하지 않은 경우긴 하지만 재귀한도 때문에 런타임 에러가 발생할 수도 있으니 알고는 있자 [(길찾기 게임)](https://school.programmers.co.kr/learn/courses/30/lessons/42892)
- 배열을 엄청 크게 잡아야만 풀리는 문제도 있다 [(광고 삽입)](https://school.programmers.co.kr/learn/courses/30/lessons/72414)
- 바로 문제의 최종 답을 생각해내는 것이 아니라 현재 필요한 기능을 하나씩 함수로 빼보자 [(드래곤 커브)](https://www.acmicpc.net/problem/15685)
- bfs를 사용할 때는 가장 먼저 찾아지는 값이 최솟값이다 [(아기 상어)](https://www.acmicpc.net/problem/16236)
- 유명한 알고리즘을 그대로 따라하는 것이 아니라 응용해서 써야할 때도 있다 [(등산 코스)](https://github.com/ddophi98/Etc-CodingTest/blob/main/%ED%94%84%EB%A1%9C%EA%B7%B8%EB%9E%98%EB%A8%B8%EC%8A%A4(%EB%93%B1%EC%82%B0%20%EC%BD%94%EC%8A%A4).md)
- in 연산을 할 때는 list보다는 set에 하는 것이 시간 효율이 좋다
- bfs 문제에서 큐에 append하는 데이터는 최대한 줄이자 (시간과 공간 복잡도를 둘 다 줄일 수 있다) [(알고스팟)](https://www.acmicpc.net/problem/1261)
- dfs 문제가 무조건 스택 자료구조로 풀리는 것은 아니니, 재귀도 한번쯤 생각해보자 [(내리막길)](https://github.com/ddophi98/Etc-CodingTest/blob/main/%EB%B0%B1%EC%A4%801520.md)
