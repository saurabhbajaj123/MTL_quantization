import matplotlib.pyplot as plt
import numpy as np
k_choices = [-7, -6, -5, -4, -3, -2]

# task_seg_acc = {
#     -7: [0.2537, 0.2536, 0.2545],
#     -6: [0.2537, 0.2554, 0.2873],
#     -5: [0.2554, 0.2556, 0.2554],
#     -4: [0.2555, 0.2555, 0.2555],
#     -3: [0.2553, 0.2553, 0.2553],
#     -2: [0.2557, 0.2555, 0.2612]
# }

# mIoU = {
#     -7: [0.2556, 0.2556, 0.2556],
#     -6: [0.2556, 0.2556, 0.2377],
#     -5: [0.2556, 0.2556, 0.2556],
#     -4: [0.2556, 0.2556, 0.2556],
#     -3: [0.2556, 0.2556, 0.2556],
#     -2: [0.2556, 0.2556, 0.17]
# }
# pp = []
# # plot the raw observations
# for k in k_choices:
#     accuracies = task_seg_acc[k]
#     plt.scatter([k] * len(accuracies), accuracies)

# # plot the trend line with error bars that correspond to standard deviation
# accuracies_mean = np.array([np.mean(v) for k,v in sorted(task_seg_acc.items())])
# accuracies_std = np.array([np.std(v) for k,v in sorted(task_seg_acc.items())])
# p = plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
# pp.append(p[0])

# for k in k_choices:
#     mIoU_accuracies = mIoU[k]
#     plt.scatter([k] * len(mIoU_accuracies), mIoU_accuracies)

# # plot the trend line with error bars that correspond to standard deviation
# mIoU_accuracies_mean = np.array([np.mean(v) for k,v in sorted(mIoU.items())])
# mIoU_accuracies_std = np.array([np.std(v) for k,v in sorted(mIoU.items())])
# p = plt.errorbar(k_choices, mIoU_accuracies_mean, yerr=mIoU_accuracies_std)
# pp.append(p[0])

# print(mIoU_accuracies_mean)
# print(mIoU_accuracies_std)
# plt.legend(pp, ["Pix acc", "mIoU"], numpoints=1)
# plt.title('Cross validation on tau')
# plt.xlabel('tau (10^)')
# plt.ylabel('Cross-validation accuracy')
# plt.show()
# plt.savefig("task_seg_acc.png")


# compression_rates = {
#     -7: [66.52199758134284, 74.83842659990854, 60.38403653409849],
#     -6: [62.32654776856671, 71.33349405987042, 57.13090166371504],
#     -5: [48.52257521608417, 64.15701007376971, 50.8672855264417],
#     -4: [65.17857839744902, 65.17857839744902, 65.17857839744902],
#     -3: [57.669635274859914, 57.669635274859914, 57.669635274859914],
#     -2: [138.872339098156, 182.68775677057877, 327.34305571217703]
# }
# b4train_compression_rates = {
#     -7: [50.08899823125288, 50.08899823125288, 50.08899823125288],
#     -6: [50.087817631360586, 50.087817631360586, 50.087817631360586],
#     -5: [50.096807359153814, 50.096807359153814, 50.096807359153814],
#     -4: [50.09046310897993, 50.09046310897993, 50.09046310897993],
#     -3: [50.08964549224141, 50.08964549224141, 50.08964549224141],
#     -2: [50.089124762160786, 50.089124762160786, 50.089124762160786]
# }
# pp = []
# # plot the raw observations
# for k in k_choices:
#     accuracies = compression_rates[k]
#     plt.scatter([k] * len(accuracies), accuracies)

# # plot the trend line with error bars that correspond to standard deviation
# accuracies_mean = np.array([np.mean(v) for k,v in sorted(compression_rates.items())])
# accuracies_std = np.array([np.std(v) for k,v in sorted(compression_rates.items())])
# p = plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
# print(accuracies_mean)
# print(accuracies_std)
# pp.append(p[0])
# for k in k_choices:
#     b4train_accuracies = b4train_compression_rates[k]
#     plt.scatter([k] * len(b4train_accuracies), b4train_accuracies)

# # plot the trend line with error bars that correspond to standard deviation
# b4train_accuracies_mean = np.array([np.mean(v) for k,v in sorted(b4train_compression_rates.items())])
# b4train_accuracies_std = np.array([np.std(v) for k,v in sorted(b4train_compression_rates.items())])
# p = plt.errorbar(k_choices, b4train_accuracies_mean, yerr=b4train_accuracies_std)
# pp.append(p[0])
# plt.title('Cross validation on tau')
# plt.xlabel('tau (10^)')
# plt.ylabel('Cross-validation accuracy')
# plt.legend(pp, ["after", "before"], numpoints=1)
# # plt.show()
# plt.savefig("compression_rates.png")


task_seg_acc = {
    -7: [0.2537, 0.2536, 0.2545],
    -6: [0.2537, 0.2554, 0.2873],
    -5: [0.2554, 0.2556, 0.2554],
    -4: [0.2555, 0.2555, 0.2555],
    -3: [0.2553, 0.2553, 0.2553],
    -2: [0.2557, 0.2555, 0.2612]
}

mIoU = {
    -7: [0.2556, 0.2556, 0.2556],
    -6: [0.2556, 0.2556, 0.2377],
    -5: [0.2556, 0.2556, 0.2556],
    -4: [0.2556, 0.2556, 0.2556],
    -3: [0.2556, 0.2556, 0.2556],
    -2: [0.2556, 0.2556, 0.17]
}
pp = []
# plot the raw observations
for k in k_choices:
    accuracies = task_seg_acc[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(task_seg_acc.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(task_seg_acc.items())])
p = plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
pp.append(p[0])

for k in k_choices:
    mIoU_accuracies = mIoU[k]
    plt.scatter([k] * len(mIoU_accuracies), mIoU_accuracies)

# plot the trend line with error bars that correspond to standard deviation
mIoU_accuracies_mean = np.array([np.mean(v) for k,v in sorted(mIoU.items())])
mIoU_accuracies_std = np.array([np.std(v) for k,v in sorted(mIoU.items())])
p = plt.errorbar(k_choices, mIoU_accuracies_mean, yerr=mIoU_accuracies_std)
pp.append(p[0])

print(mIoU_accuracies_mean)
print(mIoU_accuracies_std)
plt.legend(pp, ["Pix acc", "mIoU"], numpoints=1)
plt.title('Cross validation on tau')
plt.xlabel('tau (10^)')
plt.ylabel('Cross-validation accuracy')
plt.show()
plt.savefig("task_seg_acc.png")