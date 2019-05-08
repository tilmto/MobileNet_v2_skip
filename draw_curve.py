import matplotlib.pyplot as plt
import numpy as np

flops_sum = 1011197952

flops_skip = np.array([0.64,0.68,0.7,0.8,1])*flops_sum
acc_skip = [0.46,0.49,0.486,0.488,0.478]

flops_ee1 = np.array([0.887,0.889,0.892,0.902,0.913])*flops_sum
acc_ee1 = [0.445,0.433,0.441,0.444,0.453]

flops_ee2 = np.array([0.896,0.907,0.912,0.931])*flops_sum
acc_ee2 = [0.450,0.451,0.458,0.456]

plt.plot(flops_skip,acc_skip,'^-')
plt.plot(flops_ee1,acc_ee1,'^-')
plt.plot(flops_ee2,acc_ee2,'^-')
plt.xlabel('Flops')
plt.ylabel('Accuracy')
plt.title('Accuracy-Flops Curve of MobileNet_v2')
plt.legend(['skip_layer','early_exit conf=0.6','early_exit conf=0.5'])
plt.grid()
plt.savefig('Accuracy-Flops.jpg')
plt.show()
