def func(array):
    k = array[0]
    for x in range(1,len(array)):
        k = k | array[x]

    return k

#Program
t = int(input())
final = []
for _ in range(t):
    n, q = map(int,input().split(" "))
    a = list(map(int,input().split(" ")))
    query = []
    for i in range(q):
        qu = list(map(int,input().split(" ")))
        query.append(qu)

    answer = []

    answer.append(func(list(set(a))))
    #applying query
    for item in query:
        X, V = item
        a[X-1] = V
        answer.append(func(list(set(a))))

    final.append(answer)

for _ in range(t):
    for e in final[_]:
        print(e)













