t = int(input())
final = []
for _ in range(t):
    n ,r = map(int,input().split(" "))
    a = list(map(int,input().split(" ")))
    b = list(map(int,input().split(" ")))
    answer = []
    answer.append(b[0])
    for x in range(1,len(a)):
        element = answer[x-1] - (a[x]-a[x-1])*r
        if element < 0:
            element = 0

        answer.append(element + b[x])

    final.append(max(answer))

for _ in range(t):
    print(final[_])

