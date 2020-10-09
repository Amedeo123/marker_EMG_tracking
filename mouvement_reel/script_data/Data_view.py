import scipy.io as sio
import numpy as np
from pyomeca import Analogs

sujet = '2'
data_path = f"/home/amedeo/Documents/programmation/data_Article_Colombe/data/données_bruts/Donnees_brutes/" \
            f"Sujet_{sujet}/Essais_{sujet}/"
data = 'flexion'

mat_content = sio.loadmat(data_path + data + ".mat")
# print(mat_content)

muscles_names = ["Voltage.Voltage.", "Voltage.Voltage.","Voltage.Voltage.","Voltage.Voltage."]
a = Analogs.from_c3d(data_path+data+'.c3d')
# # a.plot(x="time", col="channel", col_wrap=3)
emg = (
    a.meca.band_pass(order=4, cutoff=[10, 425])
    .meca.center()
    .meca.abs()
    .meca.low_pass(order=4, cutoff=5, freq=a.rate)
    # .meca.normalize()
)
emg.plot(x="time", col="channel")