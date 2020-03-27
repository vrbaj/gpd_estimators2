import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import math


def resample_data(data, window_len):
    resampled_sequence = []
    for i in range(0, len(data), window_len):
        resampled_sequence.append(max(data[i:i + window_len]))
    return resampled_sequence


def create_dataset(data):
    data_set = {"positive": [],
                "negative": []}
    i=0
    for experiment in data:
        i = i + 1
        print(i)
        print(len(experiment))
        data_set["positive"].append(experiment[10])
        random_sample = random.choice(list(range(0, 10)) + list(range(11, 20)))
        data_set["negative"].append(experiment[random_sample])
    return data_set


def evaluate_roc(data_set):
    points = []
    P = len(data_set["positive"])
    N = len(data_set["negative"])
    bot_bound = math.floor(min([min(data_set["positive"]), min(data_set["negative"])])) - 1
    up_bound = math.ceil(max([max(data_set["positive"]), max(data_set["negative"])]))
    print("bot bound", bot_bound)
    print("up bound", up_bound)
    thresholds = np.linspace(bot_bound - 0.01 * bot_bound, up_bound + 0.01 * up_bound, num=500000)
    # thresholds = [threshold for threshold in range(bot_bound, up_bound)]
    FPR = []
    TPR = []
    for threshold in thresholds:

        TP = 0
        FP = 0
        for sample in data_set["positive"]:
            if sample > threshold:
                TP = TP +1
        for sample in data_set["negative"]:
            if sample > threshold:
                FP = FP + 1
        FPR.append(FP / N)
        TPR.append(TP / P)
    return FPR, TPR




data = pickle.load(open("results_roc_05.txt", "rb"))
resampled = {"ese": [],
             "le": [],
             "elbnd": []}

for sample in data["ese"]:
    resampled["ese"].append(resample_data(sample, 20))

for sample in data["le"]:
    resampled["le"].append(resample_data(sample, 20))

for sample in data["elbnd"]:
    resampled["elbnd"].append(resample_data(sample, 20))


dataset_elbnd = create_dataset(resampled["elbnd"])
dataset_ese = create_dataset(resampled["ese"])
dataset_le = create_dataset(resampled["le"])

print(dataset_ese["positive"])
print(dataset_ese["negative"])
print(max(dataset_ese["positive"]))
print(min(dataset_ese["positive"]))
print(max(dataset_ese["negative"]))
print(min(dataset_ese["negative"]))
fpr_ese, tpr_ese = evaluate_roc(dataset_ese)
fpr_le, tpr_le = evaluate_roc(dataset_le)
fpr_elbnd, tpr_elbnd = evaluate_roc(dataset_elbnd)

plt.plot(fpr_ese, tpr_ese)
plt.plot(fpr_elbnd, tpr_elbnd, "r")
plt.plot(fpr_le, tpr_le, "g")
plt.plot()
plt.autoscale(enable=True, tight=True)
plt.show()

