import math

def euclidian_distance(point1,point2):

    distances=[(p-q)**2 for p,q in zip(point1,point2)]

    return math.sqrt(sum(distances))



point1=(1,2)

point2=(2,1)

print(euclidian_distance(point1,point2))