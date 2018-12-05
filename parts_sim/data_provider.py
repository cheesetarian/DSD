import numpy as np

class DataProvider:

    def __init__(self, rwp, p0, p1, lwp, p0_cloud, p1_cloud):
        self.rwp = rwp
        self.p0  = p0
        self.p1  = p1
        self.lwp = lwp
        self.p0_cloud  = p0_cloud
        self.p1_cloud  = p1_cloud

    # copied values from ~/arts/controlfiles/testdata/tropical.*.xml, then cut
    # off some levels from the top
    def get_pressure(self):
        return np.array(
            [ 101300, 90400, 80500, 71500, 63300, 55900, 49200, 43200,
               37800, 32900, 28600, 24700, 21300, 18200, 15600, 13200,
               11100, 9370,  7890,  6660,  5650,  4800,  4090,  3500,
                3000, 2570,  1763,  1220,  852,   600,   426,   305,
                 220, 159,   116])

    def get_temperature(self):
        return np.array(
            [ 2.997000e+02,2.937000e+02,2.877000e+02,2.837000e+02,2.770000e+02,2.703000e+02,
              2.636000e+02,2.570000e+02,2.503000e+02,2.436000e+02,2.370000e+02,2.301000e+02,
              2.236000e+02,2.170000e+02,2.103000e+02,2.037000e+02,1.970000e+02,1.948000e+02,
              1.988000e+02,2.027000e+02,2.067000e+02,
              2.107000e+02,2.146000e+02,2.170000e+02,2.192000e+02,2.214000e+02,
              2.270000e+02,2.323000e+02,2.377000e+02,2.431000e+02,
              2.485000e+02,2.540000e+02,2.594000e+02,2.648000e+02,2.696000e+02])

    def get_altitude(self):  # 1km spacing in troposphere
        return np.array(
             [ 0.000000e+00,1.000000e+03,2.000000e+03,3.000000e+03,4.000000e+03,
            5.000000e+03,6.000000e+03,7.000000e+03,8.000000e+03,9.000000e+03,
            1.000000e+04,1.100000e+04,1.200000e+04,1.300000e+04,1.400000e+04,
            1.500000e+04,1.600000e+04,1.700000e+04,1.800000e+04,1.900000e+04,
            2.000000e+04,2.100000e+04,2.200000e+04,2.300000e+04,2.400000e+04,
            2.500000e+04,2.750000e+04,3.000000e+04,3.250000e+04,3.500000e+04,
            3.750000e+04,4.000000e+04,4.250000e+04,4.500000e+04,4.750000e+04]
        )

    def get_CO2(self):
        return np.ones(35)*.000390

    def get_O2(self):
        return np.ones(35)*.20915

    def get_O3(self):
        return np.ones(35)*1e-7

    def get_H2O(self):
        return np.array(
           [2.595108e-02,1.950403e-02,1.535123e-02,8.606544e-03,4.443232e-03,
            3.348814e-03,2.103083e-03,1.289580e-03,7.645933e-04,4.101490e-04,
            1.913013e-04,7.310937e-05,2.907518e-05,9.906947e-06,6.224153e-06,
            4.003531e-06,3.001779e-06,2.902016e-06,2.752485e-06,2.601595e-06,
            2.601854e-06,2.651555e-06,2.801431e-06,2.901955e-06,3.202031e-06,
            3.251939e-06,3.601740e-06,4.003950e-06,4.302710e-06,4.603415e-06,
            4.905742e-06,5.204125e-06,5.504169e-06,5.704886e-06,5.904445e-06])#*0.5

    def get_N2(self):
        return np.ones(35)*.7815

    def get_N2O(self):
        return np.ones(35)*2e-7

    def get_rain_mass_density(self):
        z = self.get_altitude()
        p = self.get_pressure()
        md = np.zeros(p.shape)
        inds = np.logical_and(p <= self.p0,
                              p >= self.p1)
        dz = z[np.where(inds)[0][-1]] - z[np.where(inds)[0][0]]
        md[inds] = self.rwp / dz

        return md

    def get_cloud_mass_density(self):
        z = self.get_altitude()
        p = self.get_pressure()
        md = np.zeros(p.shape)
        inds = np.logical_and(p <= self.p0_cloud,
                              p >= self.p1_cloud)
        dz = z[np.where(inds)[0][-1]] - z[np.where(inds)[0][0]]
        md[inds] = self.lwp / dz
        return md

    def get_cloud_mass_weighted_diameter(self):
        z = self.get_altitude()
        dm = 15e-6 * np.ones(z.size)
        return dm


    def get_surface_temperature(self):
        return np.array([[290.0]])

    def get_wind_speed(self):
        return np.array([[5.0]])

    def get_rain_mass_weighted_diameter(self):
        z = self.get_altitude()
        dm = 1e-4 * np.exp(-(z - 10.0e3) ** 2 / 1e3 ** 2)
        dm = np.maximum(dm, 1e-6)
        return dm
