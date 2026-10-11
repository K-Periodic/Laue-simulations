import numpy as np 
from scipy.spatial.transform import Rotation
from scipy import constants as const
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
### defining lattice paramters 

#lattice parameters for cubic lattice

a = np.array([1,0,0])
b = np.array([0,1,0])
c = np.array([0,0,1])

##detector dims 
length =  168.732 #in mm
height =  179.396 #^

#incident beam direction 

incident_beam = np.array([0,0,1]).T

#ABC matrix def

A = np.array([a,b,c]).T

B = 2*np.pi*np.transpose(np.linalg.inv(A))

axis = np.array([0,0,1])
angle = np.deg2rad(0) #placeholder angle and axis for U matri

U =Rotation.from_rotvec(axis * angle).as_matrix()



####adding mosaicity 

rng = np.random.default_rng(42)


#miller indicies loop
hkl_limit =12

H_list = []

for h in range(-hkl_limit, hkl_limit + 1):
    for k in range(-hkl_limit, hkl_limit +1):
        for l in range(-hkl_limit, hkl_limit +1):
            if (h,k,l) == (0,0,0):

                continue
            H_list.append(np.array([h,k,l]))


H_values = np.array(H_list)




###Quantum efficiency


qe_energies = np.array([5,20,40,60,80,100], dtype= float)
qe_values = np.array([.97, .91, .81 , .90 , .77 , .56],dtype =float)


def quantum_efficiency(energy_keV):
    if not qe_energies[0] <= energy_keV <= qe_energies[-1]:
        raise ValueError("uh energy not good bro")
    return np.interp(energy_keV, qe_energies, qe_values)

print(quantum_efficiency(7))

#to get mosaicity easily from the previous script, I def a function that we will call at the end when we start sampling different rotations 
def spots_for_orientation(U_domain):
    UB = U_domain @ B
    g_list = []

    for H in H_values:
        G = UB @ H
        g_list.append(G)


    G_values = np.array(g_list)

    ###now we enforce the laue condition for this geometry 
    k_list = []
    accepted_H = []
    accepted_G = []
    for H, G in zip(H_values, G_values):
        G_squared = np.dot(G, G)
        G_z = G[2]

        if G_z >= 0:
            continue
        k = -G_squared/(2*G_z)
        k_list.append(k)
        accepted_G.append(G)
        accepted_H.append(H)
        
        # H tells you which Miller index produced this G, just keeping track for later. 

    k_values = np.array(k_list)



    accepted_reflections = []


    #the loop above basically just solves for wavevector k and here we check if this list of k's satisfies the laue energy width 
    for H, G, k in zip(accepted_H, accepted_G, k_list):
        energy_keV = const.hbar * const.c * (k / const.angstrom) / (1000 * const.eV)

        if 5 <= energy_keV <= 90:
            accepted_reflections.append((H, G, k, energy_keV))

    ##now we just need to take the accepted reflections and project them on to an area perpendicular to the incident beam, this will be our "detector"

    detector_distance = 25 #sample to detector distance 
    reflection_coordinates=[]
    for H, G, k, energy_keV in accepted_reflections:
        k_out = G + k*incident_beam
        if k_out[2] <= 0:
            continue  # ray doesn't reach a detector at positive z
        x = detector_distance * k_out[0] / k_out[2]
        y = detector_distance * k_out[1] / k_out[2]
        if -length/2 <= x <= length/2 and -height/2 <= y <= height/2:
            # Position in native 172 micron pixel units
            px = int((x + length/2) / 0.172)
            py = int((y + height/2) / 0.172)

    # Position within a repeating module plus gap region
            local_x = px % 494  # 487 module columns + 7 gap columns
            local_y = py % 212  # 195 module rows + 17 gap rows

    # Skip gaps between modules
            if 242 <= local_x <= 244:
                continue

    # Skip the central one-column CdTe tile seam
            if local_x == 243:
                continue
            qe =quantum_efficiency(energy_keV)
            reflection_coordinates.append(np.array([x,y,qe]))


    ##now to constrain the diffracted rays to the detector 
    return reflection_coordinates



fig, ax = plt.subplots()


all_coordinates = []
###mosaicity 
mosaic_fwhm_deg = 0.2
sigma_rad = np.deg2rad(mosaic_fwhm_deg / 2.355)
#####
for j in range(100):
    small_rotation = rng.normal(0, sigma_rad, size=3)
    U_domain = Rotation.from_rotvec(small_rotation).as_matrix() @ U
    all_coordinates.extend(spots_for_orientation(U_domain))

coords = np.array(all_coordinates).reshape(-1,3)

##detector dimensions

Pix_x =  981
Pix_y = 1043



binning = 1

effective_x = Pix_x // binning 
effective_y = Pix_y // binning

x_edge = np.linspace(-length/2, length/2, effective_x + 1 )
y_edge = np.linspace(-height/2, height/2, effective_y +1 )

## now we need to count every hit to the individual detector pixels 

counts = np.histogram2d(coords[:, 0], coords[:, 1], bins = (x_edge,y_edge), weights = coords[:,2])[0]


##added bitdepth of dectrics pilatus 1m CdTe and overflow counter
detector_image = np.clip(counts.T, 0 , 1048575).astype(np.uint32)

extent = [-length/2, length/2, -height/2, height/2]

# Native detector layout
px = np.arange(Pix_x)
py = np.arange(Pix_y)

panel_x = px % 494

x_active = (
    (panel_x < 487)
    & ((panel_x < 242) | (panel_x > 244)))
y_active = py % 212 < 195
active = np.outer(y_active, x_active)

# Black gaps; very dark gray panels
background = np.where(active, 0.07, 0.0)

ax.imshow(
    background,
    origin="lower", extent=extent, interpolation="nearest",
    cmap="gray", vmin=0, vmax=1,
)

# Gray-to-white spots; zero-signal pixels are transparent
spot_cmap = LinearSegmentedColormap.from_list(
    "spots", ["#808080", "white"]
)
spot_cmap.set_bad(alpha=0)

im = ax.imshow(
    np.ma.masked_equal(detector_image, 0),
    origin="lower", extent=extent, interpolation="nearest",
    cmap=spot_cmap, vmin=0,
)

# Keep gaps black where larger displayed bins straddle them
ax.imshow(
    np.ma.masked_where(active, np.zeros_like(background)),
    origin="lower", extent=extent, interpolation="nearest",
    cmap="gray", vmin=0, vmax=1,
)

ax.set_aspect("equal")
ax.set_xlabel("Detector x (mm)")
ax.set_ylabel("Detector y (mm)")


fig.colorbar(im, ax=ax, label="Simulated hits per pixel")

plt.show()
