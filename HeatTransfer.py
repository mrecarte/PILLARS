import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
import matplotlib.patches as mpatches

def calculate_temperature_distribution_with_conduction(material_rod, cp_rod, rho_rod, dimensions_rod, emissivity_rod, h_coefficient_rod, T_initial_rod,
                                                       material_fabric, cp_fabric, rho_fabric, dimensions_fabric, emissivity_fabric, h_coefficient_fabric, T_initial_fabric,
                                                       T_exhaust, T_space, time, k_contact, A_contact, d_contact, n_segments):
    temperatures_rod = []
    temperatures_fabric = np.full((n_segments,), T_initial_fabric)

    diameter, length = dimensions_rod
    radius = diameter / 2
    surface_area_conv_rod = np.pi * diameter * length
    surface_area_rad_rod = surface_area_conv_rod + 2 * np.pi * radius**2
    volume_rod = np.pi * radius**2 * length

    length_fabric, height_fabric, thickness_fabric = dimensions_fabric
    segment_length = length_fabric / n_segments
    segment_area = height_fabric * segment_length
    surface_area_conv_fabric = segment_area * n_segments
    surface_area_rad_fabric = segment_area * n_segments
    volume_fabric = length_fabric * height_fabric * thickness_fabric / n_segments

    mass_rod = rho_rod * volume_rod
    mass_fabric_segment = rho_fabric * volume_fabric
    sigma = 5.67e-8
    h_rod = h_coefficient_rod
    h_fabric = h_coefficient_fabric

    T_final_rod = T_initial_rod
    dt = 1  #time step in seconds
    total_steps = time  #total number of time steps
    temperature_history_rod = []
    temperature_history_fabric = []

    for t in range(0, time, dt):
        #heat transfer for the rod
        heat_in_rod = h_rod * surface_area_conv_rod * (T_exhaust - T_final_rod) * dt
        heat_out_rod = emissivity_rod * sigma * surface_area_rad_rod * ((T_final_rod**4) - T_space**4) * dt
        heat_conduction_rod_to_fabric = k_contact * A_contact / d_contact * (temperatures_fabric[0] - T_final_rod) * dt

        delta_T_rod = (heat_in_rod + heat_conduction_rod_to_fabric - heat_out_rod) / (mass_rod * cp_rod)
        T_final_rod += delta_T_rod

        #heat transfer for the fabric
        new_temperatures_fabric = np.copy(temperatures_fabric)

        #heat conduction between rod and the first fabric segment
        heat_conduction_fabric_0 = k_contact * A_contact / d_contact * (T_final_rod - temperatures_fabric[0]) * dt
        delta_T_fabric_0 = (heat_conduction_fabric_0 + h_fabric * segment_area * (T_exhaust - temperatures_fabric[0]) * dt -
                            emissivity_fabric * sigma * segment_area * ((temperatures_fabric[0]**4) - T_space**4) * dt) / (mass_fabric_segment * cp_fabric)
        new_temperatures_fabric[0] += delta_T_fabric_0

        #heat conduction between fabric segments
        for i in range(1, n_segments):
            heat_conduction_fabric = k_contact * segment_area / thickness_fabric * (temperatures_fabric[i-1] - temperatures_fabric[i]) * dt
            delta_T_fabric = (heat_conduction_fabric + h_fabric * segment_area * (T_exhaust - temperatures_fabric[i]) * dt -
                              emissivity_fabric * sigma * segment_area * ((temperatures_fabric[i]**4) - T_space**4) * dt) / (mass_fabric_segment * cp_fabric)
            new_temperatures_fabric[i] += delta_T_fabric

        temperatures_fabric = new_temperatures_fabric

        temperature_history_rod.append(T_final_rod)
        temperature_history_fabric.append(np.copy(temperatures_fabric))

    return temperature_history_rod, temperature_history_fabric

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1)
ax.axis('off')

#num of segments for the Kevlar fabric
n_segments = 100

#dimensions - these ones i made hyothetical for now, but they are easy to cheange for the real life thing
#Aluminum Rod: diameter=0.02 m, length=1.0 m
#Kevlar Fabric: length=1.0 m, height=1.0 m, thickness=0.005 m

#Define rectangles for aluminum and Kevlar
#Positions and sizes are relative for visualization purposes

#Rod
rod_x = 0.1  #x-position
rod_y = 0.2  #y-position
rod_width = 0.1  #width
rod_height = 0.6  #height

#Kevlar
fabric_x = rod_x + rod_width + 0.05  #start after the rod with some gap
fabric_y = 0.2  #same y-position as rod
fabric_width = 0.9  #total width
fabric_height = 0.6  #height same as rod

#rod Rectangle
rect_aluminum = plt.Rectangle((rod_x, rod_y), rod_width, rod_height, color='grey', label='Aluminum Rod', ec='black')
ax.add_patch(rect_aluminum)

#kevlar Fabric Segments
segment_width = fabric_width / n_segments
rect_kevlar_segments = []
for i in range(n_segments):
    segment = plt.Rectangle((fabric_x + i*segment_width, fabric_y), segment_width, fabric_height, 
                            color='grey', ec=None)  #ec=None removes edge color because the edge divisions are too distracting
    ax.add_patch(segment)
    rect_kevlar_segments.append(segment)


rod_label = f"Aluminum Rod\nDiameter: 0.02 m\nLength: 1.0 m"
ax.text(rod_x + rod_width/2, rod_y + rod_height + 0.05, rod_label, color='black', fontsize=10, ha='center', va='bottom')

fabric_label = f"Kevlar Fabric\nLength: 1.0 m\nHeight: 1.0 m\nThickness: 0.005 m"
ax.text(fabric_x + fabric_width/2, fabric_y + fabric_height + 0.05, fabric_label, color='black', fontsize=10, ha='center', va='bottom')

#Color normalization and mapping setup:
#to enhance the visibility, I set normalization based on expected temperature ranges
# For now, I am using initial temp and exhaust temp as bounds

T_initial = 140.15  #initial temperature in K
T_exhaust = 1500  #exhaust temperature in K

norm = Normalize(T_initial, T_exhaust)
sm = ScalarMappable(norm=norm, cmap='inferno')  #changed colormap to 'inferno' for better contrast

#create a colorbar as a key
cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
cb = ColorbarBase(cax, cmap='inferno', norm=norm, orientation='vertical')
cb.set_label('Temperature (K)', fontsize=12)

#temperature data calculation considering conduction between materials
simulation_time = 500  #total simulation time in seconds

T_aluminum_history, T_kevlar_history = calculate_temperature_distribution_with_conduction(
    material_rod="Aluminum", 
    cp_rod=897, 
    rho_rod=2700, 
    dimensions_rod=(0.02, 1.0), 
    emissivity_rod=0.1, 
    h_coefficient_rod=64, 
    T_initial_rod=140.15,
    material_fabric="Kevlar", 
    cp_fabric=1420, 
    rho_fabric=1440, 
    dimensions_fabric=(1.0, 1.0, 0.005), 
    emissivity_fabric=0.85, 
    h_coefficient_fabric=0.04, 
    T_initial_fabric=140.15,
    T_exhaust=1500, 
    T_space=2.7, 
    time=simulation_time, 
    k_contact=0.04, 
    A_contact=0.02, 
    d_contact=0.005, 
    n_segments=n_segments
)

#adjust normalization based on simulation results to enhance contrast
kevlar_temperatures_flat = np.array(T_kevlar_history).flatten()
all_temperatures = np.concatenate((kevlar_temperatures_flat, T_aluminum_history))  #combining all temperatures

min_temp = np.min(all_temperatures)
max_temp = np.max(all_temperatures)
norm = Normalize(min_temp, max_temp)
sm = ScalarMappable(norm=norm, cmap='inferno')
cb.norm = norm

#animation update function
def update(frame):
    rect_aluminum.set_facecolor(sm.to_rgba(T_aluminum_history[frame]))

    for i, rect in enumerate(rect_kevlar_segments):
        rect.set_facecolor(sm.to_rgba(T_kevlar_history[frame][i]))

    ax.set_title(f"Heat Transfer Simulation\nTime Elapsed: {frame} s", fontsize=14)
    return [rect_aluminum] + rect_kevlar_segments

#create and run the animation
ani = FuncAnimation(fig, update, frames=simulation_time, interval=50, blit=True, repeat=False)
ani.save('heat_transfer_animation_with_distribution.mp4', writer='ffmpeg', fps=30)

plt.show()
