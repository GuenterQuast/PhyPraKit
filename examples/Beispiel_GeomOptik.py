'''Parameter transformation in Geometrical Optics:
      - determine f1, f2 and d of a two-lense system
        from system focal widhts f and postions h1 and h2 of princpal planes 


'''
# Imports  #
from kafe2 import IndexedContainer,Fit, Plot, ContoursProfiler
import numpy as np, matplotlib.pyplot as plt

# the input data
nm = 2
# - Systembrennweiten f: 
f =  np.array([10.20, 17.38])
# - Hauptebenenlagen:
hu = np.array([0.43, -21.31])
hg = np.array([-5.54, -7.12])
# - Kovarianzmatrizen aus fits von f, hu, hg 
cov = np.array([
        [[ 0.0898, -0.1247, -0.1668],
         [-0.1247,  0.1770,  0.2356],
         [-0.1668,  0.2356,  0.36410]],
        [[ 0.3255, -0.6464, -0.4029],
         [-0.6464,  1.3471,  0.8079],
         [-0.4029,  0.8079,  0.5033]] ])
# - distances of the lenses
d = np.array([10.35, 18.50])
#unc_d = 5. # very large error, d's effectively undefined
unc_d = 0.1 # measured d's as constraint 

print("*==* Eingabedaten:")
print("Systembrennweiten f: \n",f)
print("Hauptebenenlagen hu:\n", hu)
print("Hauptebenenlagen hg:\n", hg)
print("Kovarianzmatrizen:")
for i in range(nm):
      print(cov[i])
print("Linsenabstände:\n", d, " +/- ", unc_d)
      
# form vector of input parameters      
allp  = np.concatenate( (f, hu, hg, d) )
# construct over-all covariance matrix
allp_cov = np.zeros( (4*nm, 4*nm) )
# f, hu hg from Fits
for i in range(nm):
    for j in range(3):
        for k in range(3):
            allp_cov[j*nm+i, k*nm+i] = cov[i][j][k]        
    allp_cov[3*nm+i, 3*nm+i] = unc_d *unc_d         

# construct an IndexedFit Container for kafe2
iData = IndexedContainer(allp)
iData.add_matrix_error(allp_cov, matrix_type="cov")        
iData.axis_labels = [None, 'f, hu, hg, d']
    
# define the physics model
def all_from_f1f2d(f1=10, f2=20, d1=10., d2=10.):
   # calulate f, hu, hg (and d)
    data = iData.data
    nm = len(data)//4 # expect 4 concatenated arrays as input
    p_in = data.reshape((4, nm))
    fs = p_in[0]    # Brennweiten
    hus = p_in[1]   # Lagen der 1. Hauptebenen
    hgs = p_in[2]   # Lagen der 2. Hauptebenen
    ds = p_in[3]                                  

    # calculate model pedictions of inputs
    # - distances as model parameters
    m_ds=np.array([d1,d2])
    # - focal widths of lens system
    m_fs = f1*f2/(f1+f2-m_ds)
    # - sum of distances of principal planes 
    m_hsums = -m_fs*m_ds*m_ds/(f1*f2)
    # express inputs in terms of model values 
    m_hus = m_hsums - hgs
    m_hgs = m_hsums - hus
    return np.concatenate( (m_fs, m_hus, m_hgs, m_ds) )

 
f1f2Fit = Fit(iData, all_from_f1f2d)
f1f2Fit.model_label = 'all from f1, f2, d'
f1f2Fit.do_fit()

f1f2Fit.report()

f1f2Plot = Plot(f1f2Fit)
f1f2Plot.plot(residual=True)

# the same with PhyPraKit.phyFit.xFit
from PhyPraKit.phyFit import xFit

# define the physics model
#  looks slightly different as data is passed to model as 1st argumen
def _from_f1f2d(data, f1=10, f2=20, d1=10., d2=10.):
   # calulate f, hu, hg (and d)
   #### data = iData.data
    nm = len(data)//4 # expect 4 concatenated arrays as input
    p_in = data.reshape((4, nm))
    fs = p_in[0]    # Brennweiten
    hus = p_in[1]   # Lagen der 1. Hauptebenen
    hgs = p_in[2]   # Lagen der 2. Hauptebenen
    ds = p_in[3]                                  

    # calculate model pedictions of inputs
    # - distances as model parameters
    m_ds=np.array([d1,d2])
    # - focal widths of lens system
    m_fs = f1*f2/(f1+f2-m_ds)
    # - sum of distances of principal planes 
    m_hsums = -m_fs*m_ds*m_ds/(f1*f2)
    # express inputs in terms of model values 
    m_hus = m_hsums - hgs
    m_hgs = m_hsums - hus
    return np.concatenate( (m_fs, m_hus, m_hgs, m_ds) )

print(" d:\n", d, "\n f\n",f, "\n hu\n", hu, "\n hg\n", hg)
f1f2_result = xFit(_from_f1f2d, allp, s=allp_cov,
                  srel=None, sabscor=None, srelcor=None,
                  names=nm*['f'] + nm*['hu'] + nm*['hg'] + nm*['d'],
                # p0=(1., 1.),     
                #  model_kwargs = mpardict,           
                  use_negLogL=True,
                  plot=True,
                  plot_band=True,
                  plot_cor=True,
                  showplots=False,
                  quiet=True,
                  axis_labels=['Index', 'f_i, hu_i, hg_i, d_i / f,hu,hg,d(*par)'], 
                  data_legend = 'Measurements',    
                  model_legend = 'f/hu/hg,d from f1, f2'
                    )
import pprint
pprint.pprint(f1f2_result)
# show all plots
plt.show()
