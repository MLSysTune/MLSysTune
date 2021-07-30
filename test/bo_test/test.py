from .recommender import recommend_next_x
import numpy as np

X = np.array([[1,2,3,4], [4,5,6,7], [7,8,9,10]], dtype=float)
l = np.array([1,3,5], dtype=float)
y = np.array([2,4,6], dtype=float)
recommend_next_x(X, l, y)
