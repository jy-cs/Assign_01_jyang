import numpy

def min_edit_distance(src, tar):
    n = len(src)
    m = len(tar)
    distances = numpy.zeros((n + 1, m + 1))

    for i in range(n + 1):
        distances[i][0] = i

    for j in range(m + 1):
        distances[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if (src[i-1] == tar[j-1]):
                distances[i][j] = distances[i - 1][j - 1]
            else:
                distances[i][j] = 1 + min(distances[i - 1][j],     # delete
                                          distances[i - 1][j - 1], # substitute
                                          distances[i][j - 1])     # insert
    
    return distances[n][m]


