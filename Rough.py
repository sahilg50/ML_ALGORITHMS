from collections import Counter

import pandas as pd

c = Counter([1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1])
print(c)

# c = list(sorted(c.items(), key=lambda item: item[1]))
# print(c)
#
# print(c[-1][0])
#
# def sample(y: int, y1) -> float:
#     return float(y / y1)
# print(sample.__annotations__)

print( c.get(1,0))


