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
            
    #tested runtimes added explicitly from C++ testing
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
    runtimesArray[15] = 0.000974743
    runtimesArray[16] = 0.00120671
    runtimesArray[17] = 0.00165589
    runtimesArray[18] = 0.00258016
    runtimesArray[19] = 0.0042586
    runtimesArray[20] = 0.00518282 # worst case: 0.032312
    runtimesArray[21] = 0.0621917
    runtimesArray[22] = 0.0607615
    runtimesArray[23] = 0.0756915
    runtimesArray[24] = 0.106247
    runtimesArray[25] = 0.166702
    runtimesArray[26] = 0.287981
    runtimesArray[27] = 0.529678
    runtimesArray[28] = 1.01351
    runtimesArray[29] = 1.98159

runtimes()





def numberOfPermanents(dim):
    numberOfPerms = 0
    runtime = 0
    for i in range(dim):
        numberOfPermsPerSize = comb(dim, i) * dim
        numberOfPerms += numberOfPermsPerSize
        runtimeForThisSize = numberOfPermsPerSize * runtimesArray[i+1]
        runtime += runtimeForThisSize
        #print(i, "runtime:", runtimeForThisSize)
    #print("dim:", dim)
    #print("Number of perms:", numberOfPerms)
    #print("Expected time:  ", runtime)
    return (numberOfPerms, runtime)

# printing all average runtimes
def printRuntimes():
    for i in range(41):
        print(i, runtimesArray[i])
     
printRuntimes()

r10 = numberOfPermanents(10)
r14 = numberOfPermanents(14)
r20 = numberOfPermanents(20)
r40 = numberOfPermanents(40)

# Calculate the index of the highest possible mode number below the runtimeLimit
index = 0;
while (numberOfPermanents(index+1)[1] < runtimeLimit):
    index += 1
index -= 1

# printing the maximal runtime
runtimeForMaximal = numberOfPermanents(index)[1]
print(index, " runtime: ", runtimeForMaximal, "seconds", "=", runtimeForMaximal / 60 / 60, "hours")


