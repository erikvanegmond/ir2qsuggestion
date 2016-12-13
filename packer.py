from sessionizer import Sessionizer
import numpy as np

snizer = Sessionizer()
train_sess = snizer.get_sessions_with_numbers()

dummy_set = train_sess[0]
dummy_sess = [dummy_set] * 10

x = []
y = []

for pairs in dummy_sess:
    x.append(np.array(pairs[-2]).reshape((2, 1)))
    y.append(np.array(pairs[-1]).reshape((1, 1)))
for query, target in zip(x, y):
    print(query)
    print(target)


