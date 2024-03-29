# Compute the cosine similarity between two vectors
def cosine(h1, h2):
    cos = 0
    s1 = 0
    s2 = 0
    assert len(h1) == len(h2)
    for i in range(len(h1)):
        cos += h1[i]*h2[i]
        s1 += h1[i]**2
        s2 += h2[i]**2
    s1 = s1**(1/2)
    s2 = s2**(1/2)
    
    return cos/(s1*s2)

# Compute the eucledian distance between 
def euclidian(h1, h2):
    assert len(h1) == len(h2)
    distance = 0
    for i in range(len(h1)):
        distance += (h1[i] - h2[i])**2
    distance = distance**(1/2)
    return distance