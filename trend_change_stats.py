import padasip as pa
import numpy as np
from pot import pot
from datetime import datetime
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import csv
import pickle

print(datetime.now())
experiments_number = 10000

experiment_len = 1600
inputs_number = 2
filter_len = 3
parameter_change_idx = [1400]
gev_window = 1200
noise_sigmas = [0.5]
gpd_params = [[],[],[]]
gpd_params_dict = {"1": dict.fromkeys(["gamma", "mu", "sigma"], []),
                   "2": dict.fromkeys(["gamma", "mu", "sigma"], []),
                   "3": dict.fromkeys(["gamma", "mu", "sigma"], [])}
results = {"ese": [],
           "elbnd": [],
           "le": []}
print(gpd_params_dict)
gpd_result = np.zeros([experiments_number, ])
elbnd_result = np.zeros([experiments_number, ])
le_result = np.zeros([experiments_number, ])
e_result = np.zeros([experiments_number, ])
snr = np.zeros([experiments_number, ])
fit = [0, 0, 0]
mu_check = []

for noise_sigma in noise_sigmas:
    gpd_result = np.zeros([experiments_number, ])
    elbnd_result = np.zeros([experiments_number, ])
    le_result = np.zeros([experiments_number, ])
    e_result = np.zeros([experiments_number, ])
    snr = np.zeros([experiments_number, ])
    for seed_counter in range(0, experiments_number):
        np.random.seed(seed_counter)
        x = np.random.uniform(low=-1, high=1, size=(experiment_len, inputs_number))
        desired_output = np.zeros([experiment_len, ])
        filter_data = np.zeros([experiment_len, 3])
        # random_weights = np.random.uniform(low=-1, high=1, size=3)
        random_weights = np.random.normal(0, 1, 3)
        noiseless_signal = np.zeros([experiment_len, ])
        for idx in range(experiment_len):
            if idx == 0 or idx in parameter_change_idx:
                # random_weights = np.random.normal(0, 1, 1)
                random_weights = np.random.uniform(low=0.1, high=-0.1, size=1)
                random_add = np.random.uniform(-0.02, 0.02)
            filter_data[idx, 0] = x[idx, 0]
            filter_data[idx, 1] = x[idx, 1]
            filter_data[idx, 2] = 1
            if idx < parameter_change_idx[0]:
                desired_output[idx] = (x[idx, 0] + x[idx, 1]) + 0.01 * idx + np.random.normal(0, noise_sigma, 1)
                noiseless_signal[idx] = (x[idx, 0] + x[idx, 1]) + 0.01 * idx
            else: # 0.4 1.6
                desired_output[idx] = (x[idx, 0] + x[idx, 1]) + (0.01 + random_add) * idx + np.random.normal(0, noise_sigma, 1)
                noiseless_signal[idx] = (x[idx, 0] + x[idx, 1]) + (0.01 + random_add) * idx

        honu_filter = pa.filters.FilterGNGD(filter_len, mu=0.5)
        y, e, w = honu_filter.run(desired_output, filter_data)
        elbnd = pa.detection.ELBND(w, e, function="sum")

        dw = np.copy(w)
        dw[1:] = np.abs(np.diff(dw, n=1, axis=0))
        dw_count = int(dw.shape[0])

        hpp = np.ones((dw_count - gev_window, filter_len))
        for i in range(gev_window, dw.shape[0]):
            if i % 100 == 0:
                pass  # print((str(datetime.now())), " processing: ", i)
            for j in range(filter_len):
                poted_values = pot(dw[i - gev_window:i, j], 1)

                if dw[i, j] > poted_values[-1]:
                    fit = genpareto.fit(poted_values, floc=[poted_values[-1]])
                    fit = genpareto.fit(poted_values, floc=fit[1], fscale=fit[2])
                    if j == 0:
                        #print(fit[2])
                        mu_check.append(poted_values[-1])
                    gamma = fit[0]
                    mu = fit[1]

                    sigma = fit[2]
                    #gpd_params_dict[str(j + 1)]["gamma"].append(gamma)
                    #gpd_params_dict[str(j + 1)]["mu"].append(mu[0])
                    #gpd_params_dict[str(j + 1)]["sigma"].insert(sigma)
                    if dw[i, j] >= fit[1]:
                        hpp[i - gev_window, j] = 1 - genpareto.cdf(dw[i, j], fit[0], fit[1], fit[2]) + 1e-50
                        #gpd_params[j].append(fit)


        totalhpp1 = -np.log10(np.prod(hpp, axis=1))
        min_index = np.argmax(totalhpp1)
        le = pa.detection.learning_entropy(w, m=1200, order=1)



        snr[seed_counter] = 10 * np.log10((np.std(desired_output[gev_window:]) ** 2) / (noise_sigma ** 2))
        # print(Fore.RED + "experiment number: " + str(seed_counter))
        # print(Fore.GREEN + "SNR: " + (str(snr[seed_counter])))
        # print(Fore.BLACK + "min_index GPD: " + str(min_index))
        if min_index > 199 and min_index < 211:
            gpd_result[seed_counter] = 1

        max_index_elbnd = np.argmax(elbnd[-400:])
        # print("max_index elbnd: ", max_index_elbnd)
        if max_index_elbnd > 199 and max_index_elbnd < 211:
            elbnd_result[seed_counter] = 1


        total_le = np.sum(le, axis=1)
        max_index_le = np.argmax(total_le[-400:])
        # print("max_index LE: ", max_index_le)
        if max_index_le > 199 and max_index_le < 211:
            le_result[seed_counter] = 1

        max_index_e = np.argmax(abs(e[-400:]))
        # print("max_index E: ", max_index_e)
        if max_index_e > 199 and max_index_e < 211:
            e_result[seed_counter] = 1

        results["ese"].append(totalhpp1)
        results["elbnd"].append(elbnd[-400:])
        results["le"].append(total_le[-400:])

        #print("GPD detections: ", sum(gpd_result) / (seed_counter + 1)) #experiments_number)
        #print("ELBND detections: ", sum(elbnd_result) / (seed_counter + 1)) #experiments_number)
        #print("LE detections: ", sum(le_result) / (seed_counter + 1))#experiments_number)
        #print("E detections: ", sum(e_result) / (seed_counter + 1)) #experiments_number)
        #print("AVG SNR: ", sum(snr) / (seed_counter + 1))
    gpd_detections = sum(gpd_result) /experiments_number
    elbnd_detections = sum(elbnd_result) / experiments_number
    le_detections = sum(le_result) / experiments_number
    e_detections = sum(e_result) / experiments_number
    avg_snr = np.mean(snr)
    print(datetime.now())
    print("sigma:", noise_sigma)
    print("GPD detections: ", gpd_detections)
    print("ELBND detections: ", elbnd_detections)
    print("LE detections: ", le_detections)
    print("E detections: ", e_detections)
    print("AVG SNR: ", avg_snr)

    with open("trend04.csv", mode="a", newline='') as results_file:
        # writer = csv.writer(results_file)
        wr = csv.writer(results_file, dialect='excel')
        wr.writerow([noise_sigma, avg_snr, gpd_detections, elbnd_detections, le_detections, e_detections])

# gpd_gamma = gpd_params_dict["1"]["gamma"]
# gpd_mu = gpd_params_dict["1"]["mu"]
# gpd_sigma = gpd_params_dict["1"]["sigma"]
# print(gpd_params_dict["1"]["sigma"])
#
# #print(results)
#
# plt.figure()
# plt.plot(gpd_gamma)
# plt.title("gamma")
# plt.figure()
# plt.plot(gpd_mu)
# plt.title("mu")
# plt.figure()
# plt.title("sigma")
# plt.plot(gpd_sigma)
# plt.figure()
# plt.title("ese")
# plt.plot(totalhpp1)
# plt.figure()
# plt.plot(mu_check)
# plt.title("mu_check")



with open('results_roc.txt', 'wb') as f:
    pickle.dump(results, f)
# plt.show()
print(datetime.now())

