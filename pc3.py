import math
t = int(input())
final = []
for _ in range(t):

    n, k = map(int,input().split(" "))
    ans1 = math.ceil(n/k)
    loc_ans1 = 2 * (math.floor(n / ans1)) - k
    ans2 = math.floor(n/k)
    loc_ans2 = 2 * (math.floor(n / ans2)) - k
    

    loc_ans = 2*(math.floor(n/ans)) - k
    final.append([ans,loc_ans])
# print(final)

for _ in range(t):
    # print(final[_])
    print(str(final[_][0]) + ' ' + str(final[_][1]))


