import numpy as np 
from scipy.spatial.transform import Rotation
from scipy import constants as const
import matplotlib.pyplot as plt
### defining lattice paramters 

#lattice parameters for cubic lattice

a = np.array([1,0,0])
b = np.array([0,1,0])
c = np.array([0,0,1])

#detector dimensions

length = 250
height = 250


#incident beam direction 

incident_beam = np.array([0,0,1]).T

#ABC matrix def

A = np.array([a,b,c]).T

B = 2*np.pi*np.transpose(np.linalg.inv(A))

axis = np.array([0,0,1])
angle = np.deg2rad(30) #placeholder angle and axis for U matri

U =Rotation.from_rotvec(axis * angle).as_matrix()

UB = U@B



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



#from here we need to turn the miller indicies in to reciprocal lattice vectors


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
        reflection_coordinates.append(np.array([x,y]))


##now to constrain the diffracted rays to the detector 
coords = np.array(reflection_coordinates)

fig, ax = plt.subplots()
ax.scatter(coords[:, 0], coords[:, 1], s=20)

ax.set_xlim(-length / 2, length / 2)
ax.set_ylim(-height / 2, height / 2)
ax.set_aspect("equal")
ax.set_xlabel("Detector x (mm)")
ax.set_ylabel("Detector y (mm)")
ax.set_title("Simulated Laue pattern")

plt.show()



print(reflection_coordinates)