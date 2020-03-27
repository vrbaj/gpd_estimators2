# resample
data = []
[data.append(-2 * x) for x in range(23)]
print(data)

resampled_data = []
for i in range(0, len(data), 5):
    print(max(data[i:i + 5]))
