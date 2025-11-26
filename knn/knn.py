import math
from collections import Counter
def euclidian_distance(point1,point2):

    distances=[(p-q)**2 for p,q in zip(point1,point2)]

    return math.sqrt(sum(distances))





def knn(data,query,k,distance_fn=euclidian_distance):
    distances=[]

    for id,example in enumerate(data):
        # print(example[0])
        distance= distance_fn(example[0],query)
        distances.append((distance,id))

    sorted_distances=sorted(distances)

    top_k_distances= sorted_distances[:k]

    top_k_labels=[]

    for j in top_k_distances:
        top_k_labels.append(data[j[1]][1])


    
    prediction= Counter(top_k_labels).most_common(1)[0][0]

    return prediction





data= [((1,2),0),
       ((2,2),0),
       ((3,4),0),
       ((3,8),0),
       ((4,5),1),
       ((9,9),1),
       ((10,3),1),
       ((3,7),1),
]

query=(0,1)

prediction=knn(data, query,k=3)

print(prediction)