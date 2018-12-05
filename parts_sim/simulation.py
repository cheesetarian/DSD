
# coding: utf-8

# In[1]:

# PARTS-based script to import GMM-derived rain DSDs, set up a model atmosphere with
#  a liquid cloud and a rain layer below it, and simulate GMI-type observations' spread
#  from using fixed RWP with different DSDs 
# [simon wrote most of the below]

get_ipython().magic('env ARTS_INCLUDE_PATH=/home/dudavid/arts/controlfiles')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')
from data_provider import DataProvider
import math, os, sys
import numpy as np
import matplotlib.pyplot as plt
from ddfunk import *
from dmap import *
plt.style.use('bmh')


# ## PSD Data

# In[2]:

ngmm = 6 # number of GMM classes from input data
data = np.load("/home/dudavid/projects/dsd/dj/OceanClustering/ND_rwcnorm_"+str(ngmm)+".npy") # mean GMM class 1 data (just as example), output from table.ipynb
print(data.shape)
ex = np.load('/home/dudavid/projects/dsd/oceanrain_binsizes.npy')[12:(12+data.shape[1])] # as defined by OR
x = ex/1e3 # from mm into m
ex = x # save for later use
print(x.shape, ex)


# In[3]:

def mgd(in_D,in_Dm,in_mu,in_RWC): #Nw):
    """
     Calcuate the Modified Gamma drop size distribution

     Input (note, all inputs should be vectors of same size!) : 
            in_D:  vector of diameters [mm]
            in_Dm: mass-weighted mean diameter [mm]
            in_mu: shape parameter [dim-less]
            in_RWC: RWC [kg m-3]
            #in_Nw: normalized intercept parameter [IN LOG10()!!] [mm-1 m-3]
     output: N(D) [len(in_D)] in m^{-3} mm^{-1}
    """
    if(len(in_D)==1):
        print("need to input vector for in_D")
    enot = (4**4)/(np.pi*1000.0) * in_RWC/((in_Dm*1e-3)**4) *1e-3 
    # calc intercept param from RWC, Dm -- get units into mm-1 m-3
    #enot = 10**in_Nw
    eff = 6/(4**4) * ((4+in_mu)**(4+in_mu) ) / math.gamma(np.float128(4+in_mu))
    ModGam = enot * eff * (in_D/in_Dm)**in_mu * np.exp(-(4+in_mu)*in_D/in_Dm)
    return ModGam


# ## Data provider
# 
# The data provider provides the data describing the atmospheric state. [set in external file]

# In[4]:

#get_ipython().magic('matplotlib inline')
#import matplotlib.pyplot as plt


rwp = 100e-3     # set RWP [kg/m2]
lwp = 200e-3  # set CLWP [kg/m2]
p0  = 101300 #e5   # lower bound of rain layer (pressure [Pa])
p1 = 9.04e4    # upper ""  --- CANNOT MAKE THINNER WITHOUT ADJUSTING DATAPROVIDER FILE
p2 = 7.8e4

data_provider = DataProvider(rwp, p0, p1, lwp, p1, p2)

# vars just defined here for plotting
p  = data_provider.get_pressure() # if no arg, P is defined as in external DP file
print(p)
z  = data_provider.get_altitude() # """"""
print(z)
md = data_provider.get_rain_mass_density() # in kg/m3
md_cloud = data_provider.get_cloud_mass_density() # in kg/m3
print(md[0:2],p[0:2],z[0:2])


plt.plot(md*1e3, z / 1e3) # alt from m to km
plt.plot(md_cloud*1e3, z / 1e3) # alt from m to km
plt.xlabel("RWC [g/m3]")
plt.ylabel("Altitude [km]")
plt.ylim([0,18])


# ## Scattering data
# 
# We need to interpolate the psd data to match the liquid sphere scattering data.

# In[5]:

# load scattering data from Dendrite, then interpolate to match database
ssdb_path = "/home/dudavid/Dendrite/Dendrite/SSDB/ArtsScatDbase/ArtsScatDbase_v1.0.0/StandardHabits/FullSet/"
#ssdb_path = "/home/dudavid/Dendrite/Dendrite/SSDB/SSDB_temp/StandardHabits/"
# 2nd SSDB path has much more spectral resolution ## and takes longer to run
scattering_data = ssdb_path + "LiquidSphere.xml"
scattering_meta = ssdb_path + "LiquidSphere.meta.xml"
from typhon.arts.xml import load
sd = load(scattering_data)
sd_meta = load(scattering_meta)

sd_grid = np.array([m.diameter_volume_equ for m in sd_meta])
sd_grid * 1e3 # so, into m
print('scat data pts in mm: ',sd_grid*1000)

psd_x = sd_grid
psd_shape = np.interp(sd_grid, x, data[0, :])


# In[6]:

# verify that interpolated DSD matches the input closely [check]
plt.figure(figsize=[10,7])
for p in range(ngmm):
    yessir = np.interp(sd_grid, x, data[p, :])
    yessir[psd_x > .006] = 0.0
    yessir[psd_x < .0001] = 0.0
    #print(yessir)
    plt.plot(psd_x*1000, yessir , label=str(p+1))
    #plt.plot(psd_x*1000, np.interp(sd_grid, x, data[p, :]) , label=str(p+1))
#plt.plot(psd_x*1000, np.interp(sd_grid, x, mgd(x*1000,Dmi,mui,0.0003)))#data_provider.get_rain_mass_density()[0])))
plt.yscale("log")
plt.xlim([.01,7])
plt.ylim([1e-1,2e4])
plt.legend()


# ## Parts simulation -- import parts

# In[7]:

import sys
sys.path += ["/home/dudavid/src/parts"] # to verify that PARTS is in the system path
#sys.path


# In[ ]:




# In[8]:

from parts import ArtsSimulation
from parts.scattering import ScatteringSpecies
from parts.atmosphere import Atmosphere1D
from parts.sensor import PassiveSensor, ICI
from parts.atmosphere.absorption import O2, N2, H2O
from parts.atmosphere.surface import Tessem
from parts.scattering.solvers import RT4, Disort


# ### Fixed-shape psd
# 
# The FixedShape class  represents PSD data that has a fixed shape. It takes the mass density profile provided by the data provider and multiplies it with the normalized PSD shape.

# In[9]:

from parts.scattering.psd.fixed_shape import FixedShape
#from parts.scattering.psd.modified_gamma import ModifiedGamma

psd = FixedShape(psd_x, psd_shape)

# instead of using ModifiedGamma function... calculate manually and set as new fixed shape
#psd_mgd = ModifiedGamma()
Dmi, mui = 1.5, 2

#mgd_shape = mgd(x*1000, 1.3, 3, data_provider.get_rain_mass_density() ) # D, Dm, mu, RWC
#psd_shape_mgd = np.interp(sd_grid, x, mgd_shape)
mgd_psd = FixedShape(psd_x, np.interp(sd_grid,x,mgd(x*1000, Dmi, mui, data_provider.get_rain_mass_density()[0])))
#plt.plot(x*1000, mgd(x*1000, Dmi, mui, data_provider.get_rain_mass_density()[0]))


# In[10]:

#psd.psd.get_mass_density()
#f_mgd.psd.get_mass_density()


# In[11]:

#plt.plot(psd_x * 1000.0, psd.shape.ravel())


# ### Atmosphere

# In[12]:

from parts.scattering.psd import D14  # Delanoe 2014 PSD (for now, since it's in parts)
# add cloud layer properties
cloud = ScatteringSpecies("cloud", D14(1.0, 2.0, rho = 1000.0),
                        scattering_data = scattering_data,
                        scattering_meta_data = scattering_meta)


# In[13]:

# add rain species, setup atmospheric abs and scat species, plus surface

rain = ScatteringSpecies("rain", psd,
                        scattering_data = scattering_data,
                        scattering_meta_data = scattering_meta)

atmosphere = Atmosphere1D(absorbers = [O2(), N2(), H2O()],
                          scatterers = [rain, cloud],
                          surface = Tessem())  # Tessem sfc model currently only option in PARTS


# ### Sensor

# In[14]:

#channels = np.array([10.65e9, 18.7e9, 23.8e9, 36.65e9, 89e9]) #, 166e9, 180.31e9, 190.31e9]) # made-up sensor for now

channels = np.array([19e9, 36.5e9, 89e9]) # made-up sensor for now
ch_str = ['19GHz','36GHz','89GHz']
nch = channels.size
gmi = PassiveSensor("gmi", channels, stokes_dimension = 1) 
gmi.sensor_line_of_sight = np.array([[135.0]]) # angle from SC, not EIA
gmi.sensor_position = np.array([[407e3]]) # alt in m
gmi.iy_unit = "PlanckBT"  # since using Disort, no V/H polarization information...


# ### Running the simulation

# In[15]:

from typhon.arts.workspace.api import arts_api


# In[ ]:




# In[16]:

simulation = ArtsSimulation(atmosphere = atmosphere,
                            data_provider = data_provider,
                            sensors = [gmi])
#simulation.scattering_solver = Disort() 
#RT4(nstreams = 8, auto_inc_nstreams = 8, robust = 1) # use this or else problems?


#simulation.scattering_solver = RT4(nstreams = 4, auto_inc_nstreams = 16, robust = 1) # use this or else problems?
simulation.scattering_solver = Disort(nstreams = 32)
simulation.setup()
simulation.run()
print("TEST SIMULATION: ")
print(gmi.y)


# In[17]:

#FixedShape.shape?
ws = simulation.workspace
ws.verbositySet(agenda = 0, screen = 0, file = 0)


# In[18]:

# run a case with RWP=0 to form base case
data_provider.rwp = 0
simulation.run()
base_tb = np.array(np.copy(gmi.y).ravel())
print('BASE TB with RWP=0: ',base_tb)


# In[19]:

# try running through RWP values and assess spread in Tb intensity
rwp_max = 0.6
rwp_step = .08
rwp_vals = np.arange(0.001,rwp_max,rwp_step) # since 0 will be same for all

sv_tb = np.zeros([nch, rwp_vals.size, ngmm+1])
print(sv_tb.shape)

#plt.figure(figsize=[12,8])
for r in range(len(rwp_vals)):
    print('RWP loop: ',rwp_vals[r])
    for s in range(ngmm):
        data_provider.rwp = rwp_vals[r]
        new_shape = np.interp(sd_grid, x, data[s,:])
        new_shape[sd_grid > .006] = 0.0 # limiting weird stuff from huge drops
        new_shape[sd_grid < .0001] = 0.0
        simulation.atmosphere.scatterers[0].psd.set_shape( new_shape )
        simulation.run()
        sv_tb[:,r,s] = np.ravel(np.copy(gmi.y)) #-base_tb
        #print(s,np.trapz(psd.shape.ravel() * psd_x ** 3, x = psd_x), data_provider.rwp) # verify mass-weighted drops
        #plt.plot(psd_x, psd.shape.ravel(),label='GMM'+str(s+1))
    # then with a token MGD curve -- defined above
    new_shape = np.interp(sd_grid, x, mgd(x*1000,Dmi,mui,1.0)) # fixedshape will normalize, so rwc doesn't matter
    new_shape[sd_grid > .006] = 0.0 # limiting weird stuff from huge drops
    new_shape[sd_grid < .0001] = 0.0
    simulation.atmosphere.scatterers[0].psd.set_shape( new_shape )
    simulation.run()
    sv_tb[:,r,s+1] = np.ravel(np.copy(gmi.y)) #-base_tb
    psd = simulation.atmosphere.scatterers[0].psd
    #plt.plot(psd_x, psd.shape.ravel(),label='MGD')
    
#plt.yscale('log')
#plt.xlim([0,.004])
#plt.ylim([1e4,1e11])
#plt.legend()


# In[20]:

psd.psd.get_mass_density()
from copy import copy
psd_2 = copy(psd)
psd_2.psd.data = psd.shape
psd_2.psd.get_mass_density()
psd_2.size_parameter.b


# In[21]:

psd = simulation.atmosphere.scatterers[0].psd
#plt.plot(psd_x, psd.shape.ravel())

# In[22]:
f19=plt.figure(figsize=[14,9])
fs=14
plt.plot(rwp_vals, sv_tb[0,:,s+1]-base_tb[0],'k',label='19GHz, MGD ($D_m$='+str(Dmi)+', $\mu$='+str(mui)+')')
for s in range(ngmm):
    plt.plot(rwp_vals, sv_tb[0,:,s]-base_tb[0],label='19GHz, GMM'+str(s+1))
plt.legend(fontsize=fs,loc='upper left')
plt.xlabel(r"RWP [$kg m^{-2}$]",fontsize=fs+4)
plt.ylabel('$\Delta T_B$ [K]',fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
f19.savefig('/home/dudavid/projects/dsd/img/gmm_passive_clwp200.19ghz.v1.png',dpi=300)

# In[22]:
f36=plt.figure(figsize=[14,9])
plt.plot(rwp_vals, sv_tb[1,:,s+1]-base_tb[1],'k',label='36GHz, MGD ($D_m$='+str(Dmi)+', $\mu$='+str(mui)+')')
for s in range(ngmm):
    plt.plot(rwp_vals, sv_tb[1,:,s]-base_tb[1],label='36GHz, GMM'+str(s+1))
plt.legend(fontsize=fs,loc='upper left')
plt.xlabel(r"RWP [$kg m^{-2}$]",fontsize=fs+4)
plt.ylabel('$\Delta T_B$ [K]',fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
f36.savefig('/home/dudavid/projects/dsd/img/gmm_passive_clwp200.36ghz.v1.png',dpi=300)

# In[22]:
f89=plt.figure(figsize=[14,9])
fs=14
plt.plot(rwp_vals, sv_tb[2,:,s+1]-base_tb[2],'k',label='89GHz, MGD ($D_m$='+str(Dmi)+', $\mu$='+str(mui)+')')
for s in range(ngmm):
    plt.plot(rwp_vals, sv_tb[2,:,s]-base_tb[2],label='89GHz, GMM'+str(s+1))
plt.legend(fontsize=fs,loc='upper left')
plt.xlabel(r"RWP [$kg m^{-2}$]",fontsize=fs+4)
plt.ylabel('$\Delta T_B$ [K]',fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
f89.savefig('/home/dudavid/projects/dsd/img/gmm_passive_clwp200.89ghz.v1.png',dpi=300)

farts # to kill processs
# In[22]:

print(info(sv_tb))
plt.figure(figsize=[14,9])
fs=14
plt.plot(rwp_vals, sv_tb[0,:,s+1]-base_tb[0],'k-',label='19GHz, MGD ($D_m$='+str(Dmi)+', $\mu$='+str(mui))
plt.plot(rwp_vals, sv_tb[1,:,s+1]-base_tb[1],'-.',label='36GHz, MGD')
plt.plot(rwp_vals, sv_tb[2,:,s+1]-base_tb[2],'k',label='89GHz, MGD ($D_m$='+str(Dmi)+', $\mu$='+str(mui))
for s in range(ngmm):
    plt.plot(rwp_vals, sv_tb[0,:,s]-base_tb[0],'--',label='19GHz, GMM'+str(s+1))
    plt.plot(rwp_vals, sv_tb[1,:,s]-base_tb[1],'-.',label='36GHz, GMM'+str(s+1))
    plt.plot(rwp_vals, sv_tb[2,:,s]-base_tb[2],label='89GHz, GMM'+str(s+1))
plt.legend(fontsize=fs,loc='upper left')
plt.xlabel(r"RWP [$kg m^{-2}$]",fontsize=fs+4)
plt.ylabel('$\Delta T_B$ [K]',fontsize=fs+4)

plt.figure(figsize=[14,9])
for ch in range(len(channels)):
    plt.plot(rwp_vals, np.std(sv_tb[ch,:,:ngmm], axis=1), label=ch_str[ch])
fs=14
#plt.title("Stddev(TB) with "+str(ngmm)+" classes",fontsize=fs)
plt.xlabel(r"RWP [$kg m^{-2}$]",fontsize=fs+4)
plt.ylabel("$\sigma (T_B)$ [K]",fontsize=fs+4)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.legend(fontsize=fs,loc='upper left')


# In[23]:

#ws = simulation.workspace
#y  = ws.pnd_field
#
#psd = simulation.atmosphere.scatterers[0].psd
#x = psd._wsvs["x"]
#x.ws = simulation.workspace
##plt.plot(x.value[:] * 1000.0, y.value[:, 0, 0, 0])
##plt.yscale("log")
##plt.xlim([.3,4])
#


# In[24]:

# Running the simulation setup for 'native' OceanRAIN data... read in output from collate_m first
#  then use the fixed shape parts call to run fwd model without modifying input data.
typ = 'MSs3'
dir_raw_data = "/home/dudavid/projects/dsd/dj/data/"
# take every _th point to save time:
per = 15 #00

dm = np.load(dir_raw_data+'alldmd'+typ+'.npy')[::per]
rr = np.load(dir_raw_data+'allrrd'+typ+'.npy')[::per]
ku = np.load(dir_raw_data+'allkud'+typ+'.npy')[::per]
ka = np.load(dir_raw_data+'allkad'+typ+'.npy')[::per]
mu = np.load(dir_raw_data+'allmud'+typ+'.npy')[::per]
nw = np.log10( np.load(dir_raw_data+'allnwd'+typ+'.npy')[::per] ) # given as log10()
print(np.mean(mu),np.mean(dm),np.mean(nw))
nor = dm.size
#vcts = np.load(dir_raw_data+'vallcts'+typ+'.80.npy').transpose()[::per,:] # vol-weighted, smoothed
#ncts = np.load(dir_raw_data+'normallcts'+typ+'.80.npy').transpose()[::per,:] # vol-weighted and normalized
rcts = np.load(dir_raw_data+'rallcts'+typ+'.80.npy').transpose()[::per,:] # raw (per mm) counts
#scts = np.load(dir_raw_data+'snallcts'+typ+'.80.npy').transpose()[::per,:] # smoothed, normalized, not weighted
#rwcts =np.load(dir_raw_data+'rwallcts'+typ+'.80.npy').transpose()[::per,:] # RWC-normalized raw
cts = rcts # choose which one to use

# read in 80 size bins above, trim to __ in rest of code
mb = 60
bye = 1 # how many consecutive bins to sum/avg over
cts_new = np.array([ np.mean(cts[:,x*bye:(x*bye+bye)],axis=1) for x in range(int(mb/bye)) ]).transpose()
#rcts_new= np.array([ np.mean(rcts[:,x*bye:(x*bye+bye)],axis=1) for x in range(int(mb/bye)) ]).transpose()
print(cts_new.shape)
#print(x)


# In[25]:

d_mid = np.array([ np.mean(ex[n:n+2]) for n in range(int(mb/bye)) ]) # diam midpoint in m
#print(d_mid)
dD = [1e3*(ex[n+1]-ex[n]) for n in range(len(ex)-1)]  #delta D in mm

# calculate RWC from the raw data  i.e. RWC = rho*sum(4/3 pi r^3 dr)
LWC = np.array([1000.0*np.pi/6 * np.sum( cts_new[i,:-1] * (d_mid[:-1])**3 *dD ) for i in range(nor)]) 
# should be in kg/m^3 (so multiply by maybe 500m to get an idea of rwp)
print(info(LWC))


# In[26]:

zees = data_provider.get_altitude()[data_provider.get_rain_mass_density() > 0] #.size
print(zees) # tops of atmospheric levels
deltaz = zees[-1] - 0 # should give change in altitude, RWP depth
# THIS ASSUMES RAIN LAYER IS UNIFORM, ETC. AS ASSUMED ABOVE
print(deltaz)


# In[ ]:

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y[1:], box, mode='same')
    y_smooth = np.append(y[0],y_smooth) # to leave first point unmolested\n",
    return y_smooth


# In[ ]:

tb_or = np.zeros([nor,nch]) # save TBs output from simulation

#pnds = np.zeros((26, psd_x.size))
#for i in range(26): #nor):
for i in range(nor):
    data_provider.rwp = LWC[i]* deltaz # kg/m3 * m
    #if data_provider.rwp < 0.1:
    #    data_provider.rwp = 0.1 # kg/m3 * m
    new_shape = np.interp(sd_grid, x, smooth(cts_new[i,:],5))  # smooth to limit interp issues
    new_shape[np.logical_or(sd_grid > .006, sd_grid < .0003)] = 0.0 # limiting weird stuff from huge/tiny drops
    simulation.atmosphere.scatterers[0].psd.set_shape( new_shape )
    simulation.run()
    
    if np.mod(i,19)==0:  # just for checking as it runs
        #print('refl:',ka[i],ku[i],rr[i])
        print('TB:',np.ravel(np.copy(gmi.y)), data_provider.rwp )
        #print(' ')
    #print(rr[i],'dm/nw:',dm[i],nw[i])#'RWP: ',data_provider.rwp,i,nor)
    #plt.scatter(x,smooth(cts_new[i,:],5),label=str(i))
    #plt.scatter(sd_grid,new_shape,label=str(i))
    #pnds[i, :] = simulation.workspace.pnd_field.value[:200, 0, 0, 0]
    tb_or[i,:] = np.ravel(np.copy(gmi.y))
#plt.yscale('log')
#plt.xlim([0.0001,.003])
#plt.legend()


# In[ ]:

#pnds.shape


# In[ ]:

#plt.plot(psd_x, pnds[22, :], label = "22")
#plt.plot(psd_x, pnds[23, :], label = "23")
#plt.plot(psd_x, pnds[24, :], label = "24")
#plt.xlim([0.0, 0.005])
#plt.yscale("log")
#plt.legend()


# ### Results

# In[ ]:

np.save('OR_native_tbs_1kmlayer.more',tb_or)
info(tb_or), info(LWC*deltaz)
rwp_h = np.array([0,.03,.06,.1,.15,.2,.25,.3,.4,.5,.6,.8,1])#,1.25,1.5])
mns  = np.zeros([rwp_h.size-1,nch])
stds = np.zeros([rwp_h.size-1,nch])
#dbz_std = np.zeros([rwp_h.size-1, 2]) # save ku and ka
for ar in range(rwp_h.size-1):
    dex = np.logical_and(LWC*deltaz <= rwp_h[ar+1], LWC*deltaz > rwp_h[ar])
    print(LWC[dex].size,np.mean(LWC[dex]*deltaz))
    mns[ar,:]  = np.mean( tb_or[dex,:]-base_tb, axis=0 )
    stds[ar,:] = np.std(  tb_or[dex,:], axis=0 )
    #dbz_std[ar,:] = [ np.std(ku[dex]), np.std(ka[dex]) ]  # LATER --- convert dBZ to Z and back for the statistics!!
    #hist_1 = np.histogram(LWC*deltaz, tb_or[:,0], bins=rwp_h)


# In[ ]:

print(info(LWC*deltaz))
print(LWC.shape,cts_new.shape)
print(base_tb)
print(tb_or.shape)
print(mns)
print(stds)


# In[ ]:

f_nat = plt.figure(figsize=[12,9])
plt.errorbar(rwp_h[:-1], mns[:,0], stds[:,0], label='19GHz', fmt='-o')
plt.errorbar(rwp_h[:-1], mns[:,1], stds[:,1], label='36GHz', fmt='-o')
plt.errorbar(rwp_h[:-1], mns[:,2], stds[:,2], label='89GHz', fmt='-o')
plt.legend(fontsize=fs,loc='upper left')
plt.xlim([0,0.5])
plt.ylim([0,50])
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)
plt.ylabel('$\Delta T_B [K]$',fontsize=fs+4)
plt.xlabel('$RWP [kg m^{-2}]$',fontsize=fs+4)

f_nat.savefig('/home/dudavid/projects/dsd/img/nat_passive_clwp200.3freq.v1.png',dpi=300)


# In[ ]:

# now do the same type of analysis, but with radar reflectivities (no need to fwd model!)


# In[ ]:

rwc_h = np.array([0,.03,.06,.1,.15,.2,.25,.3,.4,.5,.6,.8,1]) # in g/m3
dbz_h = np.arange(12,45,2)
#rr_h = 2**(np.arange(-3,5,.2))
mns_dbz  = np.zeros([rwc_h.size-1,2])
stds_dbz = np.zeros([rwc_h.size-1,2])
stds_rr =  np.zeros([dbz_h.size-1,2])
for ar in range(rwc_h.size-1):
    #dex = np.logical_and(rr <= rr_h[ar+1], rr > rr_h[ar])
    dex = np.logical_and(LWC <= rwc_h[ar+1], LWC> rwc_h[ar])
    #dex = np.logical_and(LWC*deltaz <= rwp_h[ar+1], LWC*deltaz > rwp_h[ar])
    #print(rr[dex].size,np.mean(rr[dex]))
    ku_z, ka_z = 10**(ku[dex]/10), 10**(ka[dex]/10)
    mns_dbz[ar,:]  = [ np.mean( ku_z ),np.mean( ka_z )]
    stds_dbz[ar,:] = [ np.std(  ku_z ),np.std(  ka_z )] #convert dBZ to Z and back for the statistics!!
mns_dbz = np.log10(10*mns_dbz)
stds_dbz = np.log10(10*stds_dbz)

for ar in range(dbz_h.size-1):
    dexu = np.logical_and(ku <= dbz_h[ar+1], ku > dbz_h[ar])  # based on Ku refl only!
    dexa = np.logical_and(ka <= dbz_h[ar+1], ka > dbz_h[ar])  # based on Ka refl only!
    stds_rr[ar,:] = [ np.std( rr[dexu] ),np.std(  rr[dexa] )] #convert dBZ to Z and back for the statistics!!

d2 = plt.figure(figsize=[12,9])
plt.plot(rwc_h[:-1], stds_dbz[:,0], label='Ku')
plt.plot(rwc_h[:-1], stds_dbz[:,1], label='Ka')
#plt.errorbar(rr_h[:-1], mns_dbz[:,0], stds_dbz[:,0], label='Ku')
#plt.errorbar(rr_h[:-1], mns_dbz[:,1], stds_dbz[:,1], label='Ka')
plt.legend()
plt.xscale('log')
plt.ylabel('$\sigma$(dBZ)')
plt.xlabel('Rain water content [$g m^{-3}$]')

d2.savefig('/home/dudavid/projects/dsd/img/dbz_active_rwc.v1.png',dpi=300)

#print(stds_dbz)


# In[ ]:

plt.plot(dbz_h[:-1], stds_rr[:,0], label='Ku')
plt.plot(dbz_h[:-1], stds_rr[:,1], label='Ka')
plt.legend()
plt.ylabel('std(RR)')
plt.xlabel('dBZ')

#print(stds_dbz)


# In[ ]:



