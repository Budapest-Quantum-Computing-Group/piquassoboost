from math import comb


runtimesArray = [None] * 41
runtimesArray[0] = 0.0
runtimesArray[40] = 3600.0
minRuntime = 1.0e-5    
magicNumber = 70

runtimeLimit = 2 * 24 * 60 * 60


def runtimes():
    for i in range(39, 0, -1):
        runtimesArray[i] = max(runtimesArray[i+1] / 2, minRuntime) + minRuntime * magicNumber
        
runtimes()

runtimesArray[0] = 0
runtimesArray[1] = 3.17792e-06
runtimesArray[2] = 2.79692e-05
runtimesArray[3] = 4.58211e-05
runtimesArray[4] = 7.17591e-05
runtimesArray[5] = 9.79019e-05
runtimesArray[6] = 0.000132938
runtimesArray[7] = 0.000183283
runtimesArray[8] = 0.000235456
runtimesArray[9] = 0.000316914
runtimesArray[10] = 0.000446607
runtimesArray[11] = 0.000548008
runtimesArray[12] = 0.000668615
runtimesArray[13] = 0.00077919
runtimesArray[14] = 0.00107105


def numberOfPermanents(dim):
    numberOfPerms = 0
    runtime = 0
    for i in range(dim):
        numberOfPermsPerSize = comb(dim, i) * dim
        numberOfPerms += numberOfPermsPerSize
        runtimeForThisSize = numberOfPermsPerSize * runtimesArray[i+1]
        runtime += runtimeForThisSize
        print(i, "runtime:", runtimeForThisSize)
    print("dim:", dim)
    print("Number of perms:", numberOfPerms)
    print("Expected time:  ", runtime)
    return runtime


def printRuntimes():
    for i in range(41):
        print(i, runtimesArray[i])
    
printRuntimes()

numberOfPermanents(10)
numberOfPermanents(14)
numberOfPermanents(20)
numberOfPermanents(40)

index = 1;
while (numberOfPermanents(index) < runtimeLimit):
    index += 1

print(index, " runtime: ", numberOfPermanents(index))


