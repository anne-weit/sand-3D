#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:49:47 2025

@author: aw2a7c1l
"""

from os import chdir, environ, path, makedirs, sep
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
import math
import queue
import matplotlib.cm as cm
import matplotlib.colors as colors
from data_manip.extraction.telemac_file import TelemacFile
from data_manip.formats.regular_grid import interpolate_on_grid
from data_manip.formats.regular_grid import field_diff_on_grid
from postel.plot1d import plot1d
from postel.plot2d import *
from pathlib import Path
from matplotlib.colors import LogNorm

print("Launch Sand-3D analysis")
path_folder = '/home/aw2a7c1l/Documents/Romanche3D/exploitation2/'
path_result = f'{path_folder}{sep}figures'

# Create results folder
if not path.exists(path_result):
    makedirs(path_result)

# Generate lists of discharge and concentration
q_list = [f'q{i}' for i in range(1,5)]
q_values = [40, 60, 90, 160, 214]
q_dict = {q: q_values[q_idx] for q_idx, q in enumerate(q_list)}

c_list = [f'c{i}' for i in range(1,5)]
c_values = [0.2, 0.5, 1.0, 1.5]
c_dict = {c: c_values[c_idx] for c_idx, c in enumerate(c_list)}


d_list = [f'd{i}' for i in range(1,5)]
d_values = [100, 200, 300, 400]
d_dict = {d: d_values[d_idx] for d_idx, d in enumerate(d_list)}


r_list =[f'r{i}' for i in range(1,3)]
r_values = [2650, 2400]
r_dict = {r: r_values[r_idx] for r_idx, r in enumerate(r_list)}

path_dataset1=path_result+sep+"dataset1"
if not path.exists(path_dataset1):
            makedirs(path_dataset1)

for q in q_list:
    # Create q directory
    
    if not path.exists(path_dataset1+sep+q):
            makedirs(path_dataset1+sep+q)
    for c in c_list:
        #Create c directory
        if not path.exists(path_dataset1+sep+q+sep+c):
            makedirs(path_dataset1+sep+q+sep+c)
            
        current_path_result = path_dataset1+sep+q+sep+c
        
        
        #getting telemac file 
        file_name1 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'r2d_romanche-exp-{q}-{c}-d2-r1.slf')
        file_name2 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'r3d_romanche-exp-{q}-{c}-d2-r1.slf')
        
        file_name3 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'gai_romanche-exp-{q}-{c}-d2-r1.slf')

        res1 = TelemacFile(file_name1)#2D file
        res2 = TelemacFile(file_name2)#3D file
        res3 = TelemacFile(file_name3)#3D file
        
        elevation_z = res1.get_data_value('FREE SURFACE', -1)
        water_depth = res1.get_data_value('WATER DEPTH', -1)
        sand = res1.get_data_value('NCOH SEDIMENT1', -1)
                
        #creating output for all lines 2D
        df_profilesall = pd.read_csv('extraction_points2.csv', sep=";")
        data1 = df_profilesall[['x','y']].values.tolist()
        #print(data1)
        
        x = df_profilesall['x']
        y = df_profilesall['y']

        #poly_coord, abs_curv,values_polylines=res1.get_timeseries_on_polyline('NCOH SEDIMENT1', 
        #                               [np.array([x0,y0]) for x0,y0 in zip(x,y)], 
        #                               discretized_number = [1]*(len(x)-1))
        
        sediment = res1._get_data_on_2d_points('NCOH SEDIMENT1',-1, data1)
        water = res1._get_data_on_2d_points('WATER DEPTH',-1, data1)
        bottom = res1._get_data_on_2d_points('BOTTOM',-1, data1)
        velocity = res1._get_data_on_2d_points('SCALAR VELOCITY',-1, data1)
        bedshearstress = res3._get_data_on_2d_points('BED SHEAR STRESS',-1, data1)
        
        
        df_profilesall['concentration'] = sediment
        df_profilesall['hauteur deau'] = water
        
        # Filter invalid conc values : replace by NaN
        df_profilesall['concentration'][df_profilesall['concentration'] <= 0.00] = np.nan
        # Filter invalid water values : replace by NaN
        df_profilesall['concentration'][df_profilesall['hauteur deau'] <= 0.05] = np.nan
        
        
        # Apply groupby to classes (transect) to find mean concentration
        df_profiles_transverse = df_profilesall.groupby(['transect']).agg({'concentration': 'mean'})
        # Duplicate concentration_mean for each point based on classe number and find matching mean concentration
        df_profilesall['concentration_mean'] = [df_profiles_transverse.loc[i]['concentration'] for i in df_profilesall['transect']]
        # Normalized C_in / C_moy
        df_profilesall['concentration_normalized'] = df_profilesall['concentration']  / df_profilesall['concentration_mean'] 

        #figure 1 all lines 2D
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df_profilesall['y'], df_profilesall['concentration_normalized'],
                 label='sediment concentration', marker='o')
        
        ax.vlines(790.263, min(df_profilesall['concentration_normalized']), 
                  max(df_profilesall['concentration_normalized']), label='bridge')
        ax.legend()
        fig.savefig(f'{current_path_result}_lines_conctration.png', dpi=300)


        #getting data for 3D extraction at different heights
        depth_list = [0.1, 0.25, 0.5, 0.75, 0.9]
        csv_file= path_folder + f'concentration-index-{q}-{c}-d2-r1.csv'
        if not path.exists(csv_file):
            for d in depth_list:
                col_name = f"H{d*100}%"
                df_profilesall[col_name] = bottom + (d * water)
                conc3d = res2.get_data_on_points('NCOH SEDIMENT1',-1, 
                                           df_profilesall[['x','y',col_name]].values.tolist())
                df_profilesall[f"C{d*100}%"] = conc3d
                df_profilesall[f'CN{d*100}%'] = df_profilesall[f"C{d*100}%"]  / df_profilesall['concentration_mean']
            df_profilesall.to_csv(csv_file)
        else:
            df_profilesall.read_csv(csv_file)


        G4=df_profilesall['concentration_normalized']
        #show mesh with transects
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        plot2d_triangle_mesh(ax, res1.tri, x_label='X (m)', y_label='Y (m)', color='k', 
                             linewidth=0.1, zorder=0)
        p = ax.scatter(x, y, c=G4, cmap='RdYlBu_r',vmin=0.0,vmax=2.0, zorder=1, s=5)
        fig.colorbar(p, ax=ax, label='Concentration normalized')
        plt.show()
        fig.savefig(f'{current_path_result}_mesh_concentration.png', dpi=300)

        bs= bedshearstress
        #show mesh with transects
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        plot2d_triangle_mesh(ax, res1.tri, x_label='X (m)', y_label='Y (m)', color='k', 
                             linewidth=0.1, zorder=0)
        p = ax.scatter(x, y, c=bs, cmap='seismic',vmin=0,vmax=50, zorder=1, s=5)
        fig.colorbar(p, ax=ax, label='Bed shear stress')
        plt.show()
        fig.savefig(f'{current_path_result}_mesh_contrainte.png', dpi=300)

        vel= velocity
        #show mesh with transects
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_aspect('equal')
        plot2d_triangle_mesh(ax, res1.tri, x_label='X (m)', y_label='Y (m)', color='k', 
                             linewidth=0.1, zorder=0)
        p = ax.scatter(x, y, c=vel, cmap='jet',vmin=0,vmax=2.5, zorder=1, s=5)
        fig.colorbar(p, ax=ax, label='Scalar velocity (m/s)')
        plt.show()
        fig.savefig(f'{current_path_result}_mesh_vitesse.png', dpi=300)

        # Get mean concentration per transect
        data = df_profiles_transverse['concentration'].values.reshape(1, -1)
        
        # Plot the heatmap of mean concentration by transect
        fig, ax = plt.subplots(figsize=(10,5))
        cax = ax.imshow(data, norm=LogNorm(vmin=0.1, vmax=10),cmap='RdYlBu_r')
        # fig.colorbar(label='Mean concentration')
        #plt.vlines(25.5, -0.5, 10.5, linewidth=3)
        #plt.title('Heatmap using Matplotlib' 
        cb = fig.colorbar(cax, ax=ax)
        plt.yticks([])
        plt.xlabel('Transects (upstream to downstream)')
        plt.show()
        fig.savefig(f'{current_path_result}_heatmap_transect_mean_concentration.png', dpi=300)


        # Add a column for profile id
        profile_id = np.tile(np.arange(0, 11)[::-1], len(np.unique(df_profilesall['transect'])))
        df_profilesall['profile'] = profile_id
        
        vmin = 0
        vmax = np.nanquantile(df_profilesall[[f'C{d*100}%' for d in depth_list]], q=0.9)
        
        data_dict = {'concentration': 'Concentration (g/L)', 
                     'concentration_normalized': 'Concentration norm'}
        data_dict.update({f"C{d*100}%": f'Concentration index at {d} ' for d in depth_list})
        data_dict.update({f"CN{d*100}%": f'Concentration norm at {d} ' for d in depth_list})
        for data_name, title in data_dict.items():
            #data_name='CN10.0%'
            title=data_dict[data_name]
            if data_name[:2] =='CN':
                log_cb=True
            else: 
                log_cb=False
            
            #break        
            # Pivot data and convert to array
            data = df_profilesall.pivot('profile', 'transect', data_name).values
            # Plot the heatmap of mean concentration by transect
            fig, ax = plt.subplots(figsize=(15,5))
           # plt.imshow(data, cmap='RdYlBu_r', aspect='auto')
            if log_cb:
               cax = ax.imshow(data, norm=LogNorm(vmin=0.1, vmax=10), cmap='RdYlBu_r')
            else: 
               cax = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', 
                               vmin=vmin, vmax=vmax)
            cb = fig.colorbar(cax, ax=ax)
  #          cb = plt.colorbar()
            
            cb.set_label(label=title, fontsize=14)
            plt.vlines(25.5, -0.5, 10.5, linewidth=3)
            #plt.title('Heatmap using Matplotlib')
            plt.yticks([])
            plt.xticks(fontsize=12)
            plt.xlabel('Transects (upstream to downstream)', fontsize=14)
            plt.ylabel('Right to Left', fontsize=14)
            plt.ylim(10.5, -0.5)
            plt.show()
            fig.savefig(f'{current_path_result}_heatmap_{data_name}.png', dpi=300)
            
        # Coordinates from transect 27
        points_transect27 = [[679.789,783.239],[682.259,786.386],[684.729,789.533],
                     [687.198,792.679],[689.668,795.826],[692.137,798.973],
                     [694.607,802.119],[697.076,805.266],[699.546,808.413],
                     [702.015,811.559],[704.485,814.706]]       
        fig, ax = plt.subplots(figsize=(10,5))
        cax = ax.imshow(data, norm=LogNorm(vmin=0.1, vmax=10),cmap='RdYlBu_r')
        # fig.colorbar(label='Mean concentration')
        #plt.vlines(25.5, -0.5, 10.5, linewidth=3)
        #plt.title('Heatmap using Matplotlib' 
        cb = fig.colorbar(cax, ax=ax)
        plt.yticks([])
        plt.xlabel('Transects (upstream to downstream)')
        plt.show()
        
        abscisse = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
        
        first = 1
        last = -1
        
        bottom=res1.get_data_on_points('BOTTOM', -1, points_transect27)
        free_surface=res1.get_data_on_points('FREE SURFACE', -1, points_transect27)
        conc = []
        z = []
        abscisse_values = []
        for point_idx, point in enumerate(points_transect27[first:last]):
            conc.append(res2.get_data_on_vertical_segment('NCOH SEDIMENT1', -1, point))
            z_temp = res2.get_data_on_vertical_segment('ELEVATION Z', -1, point)
            z.append(z_temp)
            abscisse_values.append([abscisse[point_idx + first]] * len(z_temp))
 
        df_27 = pd.DataFrame({'abscisse': np.concatenate(abscisse_values),
                              'z': np.concatenate(z),
                              'c_index': np.concatenate(conc)
                              })
            
        df_27['c_normalized'] = df_27['c_index'] / df_27['c_index'].mean()
            
        csv_file= path_folder + f'concentration-transect27-{q}-{c}-d2-r1.csv'
        df_27.to_csv(csv_file)
        
        
        pivot_table = df_27.pivot(index='z', columns='abscisse', values='c_normalized')

        
        fig, ax = plt.subplots(figsize=(10,5))
        plt.plot(abscisse[first:last], bottom[first:last], zorder=10, color='k', linewidth=2)
        plt.plot(abscisse[first:last], free_surface[first:last], zorder=10, color='grey', linewidth=1)
#        plt.imshow(np.log(pivot_table), cmap='viridis', aspect='auto', origin='lower')
        plt.scatter(df_27['abscisse'], df_27['z'], c=df_27['c_normalized'],
                    cmap='RdYlBu_r',vmin=0, vmax=2)
                    #norm=colors.LogNorm(vmin=0.01, vmax=100))
                    #norm=colors.LogNorm(vmin=df_27['c_normalized'].min(), 
                     #                   vmax=df_27['c_normalized'].max()))
                    
        plt.colorbar(label='c normalized (Transect 27)')
        plt.title(f'transect27-{q}-{c}')
        plt.xlabel('Distance (m) from LB to RB)', fontsize=14)
        plt.ylabel('Altitude (m)', fontsize=14)
        plt.show()
        fig.savefig(f'{current_path_result}_transect27_cnorm.png', dpi=300)
        
        
path_dataset2=path_result+sep+"dataset2"
if not path.exists(path_dataset2):
            makedirs(path_dataset2)        

q_save = []
d_save = []
c_save = []

#q_key = "q1"
#q_value = 40
#d_key = "d1"
#d_value = 100
for q_key, q_value in q_dict.items():
    if not path.exists(path_dataset2+sep+q_key):
            makedirs(path_dataset2+sep+q_key)  
    for d_key, d_value in d_dict.items():
        current_path_result = path_dataset2+sep+q_key+sep+d_key
        if not path.exists(current_path_result):
            makedirs(current_path_result)  
        #getting telemac file 
        file_name1 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'r2d_romanche-exp-{q_key}-c3-{d_key}-r1.slf')
        file_name2 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'r3d_romanche-exp-{q_key}-c3-{d_key}-r1.slf')

        res1 = TelemacFile(file_name1)#2D file
        res2 = TelemacFile(file_name2)#3D file
        
        df_profilesall = pd.read_csv('extraction_points2.csv', sep=";")
        data1 = df_profilesall[['x','y']].values.tolist()
        
        point_ref = [[682.259, 786.386, 709.5]]
        c_index = res2._get_data_on_3d_points('NCOH SEDIMENT1', -1, point_ref)
        
        points_transect27 = [[679.789,783.239],[682.259,786.386],[684.729,789.533],
                     [687.198,792.679],[689.668,795.826],[692.137,798.973],
                     [694.607,802.119],[697.076,805.266],[699.546,808.413],
                     [702.015,811.559],[704.485,814.706]]
        
        # Get data for global lineplot
        sediment27 = res1._get_data_on_2d_points('NCOH SEDIMENT1',-1, points_transect27)
        mean_concentration27 = sediment27.mean()        
        q_save.append(q_value)
        d_save.append(d_value)
        c_save.append(c_index/mean_concentration27)
        
        sediment = res1._get_data_on_2d_points('NCOH SEDIMENT1',-1, data1)
        water = res1._get_data_on_2d_points('WATER DEPTH',-1, data1)
        bottom = res1._get_data_on_2d_points('BOTTOM',-1, data1)        
             
        df_profilesall['concentration'] = sediment
        df_profilesall['hauteur deau'] = water
        
        # Filter invalid conc values : replace by NaN
        df_profilesall['concentration'][df_profilesall['concentration'] <= 0.00] = np.nan
        # Filter invalid water values : replace by NaN
        df_profilesall['concentration'][df_profilesall['hauteur deau'] <= 0.05] = np.nan
        
        
        # Apply groupby to classes (transect) to find mean concentration
        df_profiles_transverse = df_profilesall.groupby(['transect']).agg({'concentration': 'mean'})
        # Duplicate concentration_mean for each point based on classe number and find matching mean concentration
        df_profilesall['concentration_mean'] = [df_profiles_transverse.loc[i]['concentration'] for i in df_profilesall['transect']]
        # Normalized C_in / C_moy
        df_profilesall['concentration_normalized'] = df_profilesall['concentration']  / df_profilesall['concentration_mean'] 

        #getting data for 3D extraction at different heights
        depth_list = [0.1, 0.25, 0.5, 0.75, 0.9]
        csv_file= path_folder + f'concentration-index-{q}-c3-{d}-r1.csv'
        if not path.exists(csv_file):
            for d in depth_list:
                col_name = f"H{d*100}%"
                df_profilesall[col_name] = bottom + (d * water)
                conc3d = res2.get_data_on_points('NCOH SEDIMENT1',-1, 
                                           df_profilesall[['x','y',col_name]].values.tolist())
                df_profilesall[f"C{d*100}%"] = conc3d
                df_profilesall[f'CN{d*100}%'] = df_profilesall[f"C{d*100}%"]  / df_profilesall['concentration_mean']
            
            df_profilesall.to_csv(csv_file)
        else:
            df_profilesall = pd.read_csv(csv_file)
            
        # Add a column for profile id
        profile_id = np.tile(np.arange(0, 11)[::-1], len(np.unique(df_profilesall['transect'])))
        df_profilesall['profile'] = profile_id
        
        # Define extrema
        vmin = 0
        vmax = np.nanquantile(df_profilesall[[f'C{d*100}%' for d in depth_list]], q=0.9)
        
        # Define dict for iterative heatmap 
        data_dict = {'concentration': 'Concentration (g/L)', 
                     'concentration_normalized': 'Concentration norm'}
        data_dict.update({f"C{d*100}%": f'Concentration index at {d} ' for d in depth_list})
        data_dict.update({f"CN{d*100}%": f'Concentration norm at {d} ' for d in depth_list})
        for data_name, title in data_dict.items():
            #data_name='CN10.0%'
            title=data_dict[data_name]
            if data_name[:2] =='CN':
                log_cb=True
            else: 
                log_cb=False
        
            # Pivot data and convert to array
            data = df_profilesall.pivot('profile', 'transect', data_name).values
            # Plot the heatmap of mean concentration by transect
            fig, ax = plt.subplots(figsize=(15,5))
            if log_cb:
               cax = ax.imshow(data, norm=LogNorm(vmin=0.1, vmax=10), cmap='RdYlBu_r')
            else: 
               cax = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', 
                               vmin=vmin, vmax=vmax)
            cb = fig.colorbar(cax, ax=ax)            
            cb.set_label(label=title, fontsize=14)
            plt.vlines(25.5, -0.5, 10.5, linewidth=3)
            plt.yticks([])
            plt.xticks(fontsize=12)
            plt.xlabel('Transects (upstream to downstream)', fontsize=14)
            plt.ylabel('Right to Left', fontsize=14)
            plt.ylim(10.5, -0.5)
            plt.show()
            fig.savefig(f'{current_path_result}_heatmap_{data_name}.png', dpi=300)
        

df2 = pd.DataFrame({'q': q_save, 'd': d_save, 'c_index': c_save})

fig, ax = plt.subplots(figsize=(10,5))
for d_value in d_values:
    rows = df2[df2['d'] == d_value]
    plt.plot(rows['q'], rows['c_index'], label=f"{d_value} µm",linestyle= '--', marker='o')
    plt.xlabel('Flow discharge Q (m3/s))', fontsize=14)
    plt.ylabel('C_ind/C_mean', fontsize=14)
    plt.legend()
plt.show()
fig.savefig(f'{path_dataset2}{sep}lineplot_q_Cnorm.png', dpi=300)



path_dataset3=path_result+sep+"dataset3"
if not path.exists(path_dataset3):
            makedirs(path_dataset3)        

q_save = []
r_save = []
c_save = []

for q_key, q_value in q_dict.items():
    if not path.exists(path_dataset2+sep+q_key):
            makedirs(path_dataset2+sep+q_key)  
    for r_key, r_value in r_dict.items():
        current_path_result = path_dataset2+sep+q_key+sep+r_key
        if not path.exists(current_path_result):
            makedirs(current_path_result)  
        #getting telemac file 
        file_name1 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'r2d_romanche-exp-{q_key}-c3-d2-{r_key}.slf')
        file_name2 = path.join('/home','aw2a7c1l','Documents', 
                               'Romanche3D', 'exploitation2',
                               f'r3d_romanche-exp-{q_key}-c3-d2-{r_key}.slf')

        res1 = TelemacFile(file_name1)#2D file
        res2 = TelemacFile(file_name2)#3D file
        
        point_ref = [[682.259, 786.386, 709.5]]
        c_index = res2._get_data_on_3d_points('NCOH SEDIMENT1', -1, point_ref)
        
        points_transect27 = [[679.789,783.239],[682.259,786.386],[684.729,789.533],
                     [687.198,792.679],[689.668,795.826],[692.137,798.973],
                     [694.607,802.119],[697.076,805.266],[699.546,808.413],
                     [702.015,811.559],[704.485,814.706]]
        
        sediment = res1._get_data_on_2d_points('NCOH SEDIMENT1',-1, points_transect27)
        
        mean_concentration = sediment.mean()
        
        
        q_save.append(q_value)
        r_save.append(r_value)
        c_save.append(c_index/mean_concentration)
        

df2 = pd.DataFrame({'q': q_save, 'r': r_save, 'c_index': c_save})

fig, ax = plt.subplots(figsize=(10,5))
for r_value in r_values:
    rows = df2[df2['r'] == r_value]
    plt.plot(rows['q'], rows['c_index'], label=f"{r_value} kg/m3",linestyle= '--', marker='o')
    plt.xlabel('Flow discharge Q (m3/s))', fontsize=14)
    plt.ylabel('C_ind/C_mean', fontsize=14)
    plt.legend()
plt.show()
fig.savefig(f'{path_dataset3}{sep}lineplot_q_Cnorm_r.png', dpi=300)

