import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

tol_init = 2
tol_final = 7
ns = 31
noeuds = True
if noeuds:
    mat_content = sio.loadmat('RMSE_ip_ns.mat')
else:
    mat_content = sio.loadmat('RMSE_ip_nQ.mat')
RMSE_ip = mat_content['RMSE_ip']
if noeuds:
    RMSE_ac = np.ndarray((tol_final-tol_init, 31))
else:
    RMSE_ac = np.ndarray((tol_final-tol_init, 10))
for i in range(tol_init, tol_final):
    if noeuds:
        mat_content = sio.loadmat('RMSE_ac_ns_' + str(i) + ".mat")
    else:
        mat_content = sio.loadmat('RMSE_ac_nX_' + str(i) + ".mat")
    RMSE_ac[i-tol_init, :] = mat_content['RMSE_ac']

#
x1 = range(4, tol_final)
width = 0.2
# x1 = [i + width for i in x1]
height = []
for i in range(2, tol_final - tol_init):
    # height.append(abs(np.log(1+(RMSE_ip[i, 30]))))
    height.append(RMSE_ip[i, 30])
plt.bar(x1, height, width, color='red')
plt.xlabel('1e-tol')
plt.ylabel('RMSE')

x2 = [i - width for i in x1]
height = []
for i in range(2, tol_final - tol_init):
    # height.append(abs(np.log(1+(RMSE_ac[i, 30]))))
    height.append(RMSE_ac[i, 30])
plt.bar(x2, height, width, color='blue')
plt.xlabel('1e-tol')
plt.ylabel('RMSE')
plt.title('RMSE')
plt.legend(["Ipopt", "Acados"], loc=2)
plt.show()