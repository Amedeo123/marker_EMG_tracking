
from updated.ConvertOsim2Biorbd import ConvertedFromOsim2Biorbd5
modelin = "colombe_real_data.osim"
modelout = "colombe_real_data_scaled.osim"
xmlin = "scaling_tools.xml"
xml_out = "scaled_file.xml"
marker = "marker_full_rot.trc"

# Scale(modelin, modelout, xmlin, xml_out, marker, mass=-1, height=-1)
Model = ConvertedFromOsim2Biorbd5( "./Belaise_scaled_test_updated.bioMod", "./colombe_real_scaled.osim")