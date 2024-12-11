import numpy as np
import math

def compute_similarity(vector1: list, vector2: list): # Pearson correlation
    # for i in range(10):
    #     print(type(vector1[i]), vector1[i], math.isnan(vector1[i]))

    # calculating averages
    v1nonan = [x for x in vector1 if not math.isnan(x)]
    avg1 = sum(v1nonan) / len(v1nonan)
    v2nonan = [x for x in vector2 if not math.isnan(x)]
    avg2 = sum(v2nonan) / len(v2nonan)

    numerator = sum((vector1[i] - avg1) * (vector2[i] - avg2) for i in range(len(vector1)) if not math.isnan(vector1[i]) and not math.isnan(vector2[i]))

    count_nonans = len([1 for i in range(len(vector1)) if not math.isnan(vector1[i]) and not math.isnan(vector2[i])])
    mult = min(count_nonans/100, 1)

    denominator = sum((vector1[i] - avg1) ** 2 for i in range(len(vector1)) if not math.isnan(vector1[i]))
    denominator *=  sum((vector2[i] - avg2) ** 2 for i in range(len(vector2)) if not math.isnan(vector2[i]))
    denominator = denominator ** 0.5

    if denominator == 0:
        return 0
    return numerator * mult / denominator

if __name__ == "__main__":
    
    # vector_a, vector_b = [1, 2, 3, 4], [4, 3, 2, 1]
    # vector_a, vector_b = [3, 4], [3, 5]
    vector_a, vector_b = [1, 2, 3, 4, float('nan'), 3, 5], [1, 2, 3, 5, 2, 3, float('nan')]
    print(compute_similarity(vector_a, vector_b))
    