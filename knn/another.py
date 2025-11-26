from collections import Counter

# Define the k_nearest_neighbors function as described in the lesson
def k_nearest_neighbors(data, query, k, distance_fn):
    neighbor_distances_and_indices = []
    
    # Compute distance from each training data point, using example[0] to get the feature data
    for idx, label in enumerate(data):
        distance = distance_fn(label[0], query)  
        neighbor_distances_and_indices.append((distance, idx))
    
    # Sort the list by distance and select the first k entries
    sorted_neighbors = sorted(neighbor_distances_and_indices)
    k_nearest_distances_and_indices = sorted_neighbors[:k]
    
    # TODO: Assign labels to k_nearest_labels based on the k_nearest_distances_and_indices
    
    k_nearest_labels = []
    for j in k_nearest_distances_and_indices:
        k_nearest_labels.append(data[j[1]][1])
    
    # TODO: Use the Counter class for a majority vote to determine the predicted label
    most_common = Counter(k_nearest_labels).most_common(1)[0][0]

    return most_common

# Define the euclidean_distance function as needed
def euclidean_distance(point1, point2):
    return sum((p - q) ** 2 for p, q in zip(point1, point2)) ** 0.5

# A cosmic dataset of objects with features 'size' and 'brightness'
cosmic_objects = [
    ((1, 5), 'Dwarf Star'),  # Size 1, Brightness 5
    ((3, 8), 'Giant Star'),  # Size 3, Brightness 8
    ((2, 6), 'Dwarf Star'),  # Size 2, Brightness 6
]

# New object to classify
new_object = (2, 7)

# Using the defined functions to classify the new object
predicted_class = k_nearest_neighbors(cosmic_objects, new_object, k=2, distance_fn=euclidean_distance)
print(predicted_class)  # Predicted class is expected to be 'Dwarf Star'