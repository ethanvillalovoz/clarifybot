from collections import Counter

with open("question_ratings.log") as f:
    lines = f.readlines()

counter = Counter()
for line in lines:
    q, rating = line.strip().split('\t')
    counter[(q, rating)] += 1

for (q, rating), count in counter.most_common():
    print(f"{rating.upper()} ({count}): {q}")