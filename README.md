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
new_list = sorted(list, key=lambda item: item[2]) // 특정 키 기준으로 정렬
```
