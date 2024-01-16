import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axisartist.axislines import Subplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from cfd_class.cfd_class import *
import os
from copy import deepcopy
import pickle
import gzip
import math 
from scipy.integrate import trapezoid
import sys
from typing import List, TextIO
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import scipy.stats as stats
from scipy.optimize import curve_fit
import re
from TexSoup import TexSoup

# matplotlib.use("pgf")
plt.style.use('ggplot')
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "Times New Roman",
    "font.serif": [],                   # blank entries should cause plots 
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 11,               # LaTeX default is 10pt font.
    "font.size": 11,
    "legend.fontsize": 8,               # Make the legend/label fonts 
    "xtick.labelsize": 8,               # a little smaller
    "ytick.labelsize": 8,
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])
    }
matplotlib.rcParams.update(pgf_with_latex)

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("", flush=True, file=out)

def concat_fluid_df(caseloc, mesh_id, irr_id, channel_length: float, gap_width: float):
    prop_dict = {}
    for it1 in ['u', 'v', 'w', 'T', 'P', 'k', 'omega']:
        temp = pd.read_csv(f'{caseloc}\\irr{irr_id}\\case{mesh_id}_irr{irr_id}\\case{mesh_id}_irr{irr_id}_fluid_{it1}.csv'); del temp[temp.columns[0]]
        label_ = np.array([x.split('_') for x in temp.columns]); label_ = np.transpose(label_)
        get_ = pd.DataFrame()
        get_['domain'] = label_[0]; get_['L'] = np.array([int(x) for x in label_[1]]); get_['H'] = np.array([int(x) - 1 for x in label_[2]])
        for x in range(0, temp.shape[0]):
            get_[f't_{x}'] = temp.loc[x,:].values
        prop_dict[it1] = deepcopy(get_)
    prop_dict['vscalar'] = deepcopy(prop_dict['u'])
    for it1 in prop_dict['vscalar'].columns[3:]:
        vscalar = np.array([ np.sqrt(np.sum([prop_dict['u'].loc[x, it1]**2 + prop_dict['v'].loc[x, it1]**2 + \
            prop_dict['w'].loc[x, it1]**2])) for x in range(0, prop_dict['vscalar'].shape[0]) ])
        prop_dict['vscalar'][it1] = deepcopy(vscalar)
    
    geom_coor = deepcopy(prop_dict['u'].loc[:, 'L':'H'])
    nL = np.max(np.unique(geom_coor['L'])); nH = np.max(np.unique(geom_coor['H']))
    deltaL = channel_length / (nL + 1); deltaH = gap_width / (nH + 1)
    geom_coor['y'] = np.array([ deltaL * (it1 + 0.5) for it1 in geom_coor['L'].values ])
    geom_coor['z'] = np.array([ deltaH * (it1 + 0.5) for it1 in geom_coor['H'].values ])
    
    return prop_dict, geom_coor

class analysis:
    def __init__(self, info_, geom_, poolTnew_, poolvnew_, poolpnew_,
                 nlayer_, delta_t, channel_length, channel_width, channel_height, plot = False):
        layer_segment_ = self.get_layer_segments(info_)
        locals_dict = dict.fromkeys(list(range(0, nlayer_)), {'v': {'fluid': [], 'stream': []},
                                                              'T': {'fluid': [], 'stream': [],
                                                                    'glass': [], 'abs': [], 'insul': []},
                                                              'Pgrady': {'fluid': [], 'stream': []},
                                                              'vgradz': {'fluid': []},
                                                              'Tgradz': {'fluid': []}})
        average_dict = {'v': {'fluid': [], 'stream': []},
                        'T': {'fluid': [], 'stream': [],
                                'glass': [], 'abs': [], 'insul': []},
                        'Pgrady': {'fluid': [], 'stream': []},
                        'vgradz': {'fluid': []},
                        'Tgradz': {'fluid': []}}
        
        times_save = []
        
        for it1 in range(0, nlayer_):
            vfluid_, vstream_, \
            Tfluid_, Tstream_, Tglass_, Tabs_, Tinsul_, \
            Pgradyfluid_, Pgradystream_, \
            vgradzfluid_, \
            Tgradzfluid_, \
            times_ = self.calc_locals(info_, geom_, poolTnew_, poolvnew_, poolpnew_, layer_segment_, it1,
                                      delta_t, channel_length, channel_width, channel_height, plot=plot)
            
            locals_dict[it1]['v']['fluid'] = deepcopy(vfluid_)
            locals_dict[it1]['v']['stream'] = deepcopy(vstream_)
            
            locals_dict[it1]['T']['fluid'] = deepcopy(Tfluid_)
            locals_dict[it1]['T']['stream'] = deepcopy(Tstream_)
            locals_dict[it1]['T']['glass'] = deepcopy(Tglass_); locals_dict[it1]['T']['abs'] = deepcopy(Tabs_); locals_dict[it1]['T']['insul'] = deepcopy(Tinsul_)
            
            locals_dict[it1]['Pgrady']['fluid'] = deepcopy(Pgradyfluid_)
            locals_dict[it1]['Pgrady']['stream'] = deepcopy(Pgradystream_)
            
            locals_dict[it1]['vgradz']['fluid'] = deepcopy(vgradzfluid_)
            locals_dict[it1]['Tgradz']['fluid'] = deepcopy(Tgradzfluid_)            
            
            times_save = times_
            
        for it1, it2 in average_dict.items():
            if it1 in ['v', 'k', 'T', 'Pgrady', 'vgradz', 'Tgradz']:
                for it3 in list(average_dict[it1].keys()):
                    average_dict[it1][it3] = np.zeros(shape=(1,len(times_save)))[0]
            else:
                average_dict[it1] = np.zeros(shape=(1,len(times_save)))[0]
                
        for it1, it2 in locals_dict.items():
            average_dict['v']['fluid'] += it2['v']['fluid']
            average_dict['v']['stream'] += it2['v']['stream']
            
            average_dict['T']['fluid'] += it2['T']['fluid']
            average_dict['T']['stream'] += it2['T']['stream']
            
            average_dict['T']['glass'] += it2['T']['glass']
            average_dict['T']['abs'] += it2['T']['abs']
            average_dict['T']['insul'] += it2['T']['insul']
            
            average_dict['Pgrady']['fluid'] += it2['Pgrady']['fluid']
            average_dict['Pgrady']['stream'] += it2['Pgrady']['stream']
            
            average_dict['vgradz']['fluid'] += it2['vgradz']['fluid']            
            average_dict['Tgradz']['fluid'] += it2['Tgradz']['fluid']
            
        for it1, it2 in average_dict.items():
            if it1 in ['v', 'T', 'Pgrady', 'vgradz', 'Tgradz']:
                for it3 in list(average_dict[it1].keys()):
                    average_dict[it1][it3] = average_dict[it1][it3] / nlayer_
            else:
                average_dict[it1] = average_dict[it1] / nlayer_
        
        average_dict['T']['fluid'] = [np.max([x,y]) for x,y in zip(average_dict['T']['fluid'], average_dict['T']['stream'])]
        
        self.__locals = locals_dict
        self.__times = times_save
        self.__averages = average_dict
        
    @staticmethod
    def time_least_square_gradient(info_: Info, geom_: Geom, value_: Value):
        length_times = list(range(1, len(value_.cunit[0].ltime)))
        for it1 in progressbar(range(0, len(list(info_.cells.keys()))), f'Variable {value_.name}', size=60):
            for it2 in length_times:
                a_neigh_coef = np.zeros(shape=(3,3), dtype = float)
                a_bound_coef = np.zeros(shape=(3,3), dtype = float)
                b_neigh_coef = np.zeros(shape=(3,1), dtype = float)
                b_bound_coef = np.zeros(shape=(3,1), dtype = float)
                for it3, it4 in info_.cells[it1].connect.items():
                    w_ = 1 / geom_.dCF.scalar(it1, it3)
                    dCF_ = geom_.dCF.vec(it1, it3)
                    for it5 in range(0, 3):
                        for it6 in range(0, 3):
                            a_neigh_coef[it5][it6] += w_ * dCF_[it5] * dCF_[it6]
                        b_neigh_coef[it5][0] += w_ * dCF_[it5] * (value_.cunit[it3].ltime[it2] - value_.cunit[it1].ltime[it2])
                if len(list(info_.cells[it1].conj.keys())) > 0:
                    for it3, it4 in info_.cells[it1].conj.items():
                        w_ = 1 / geom_.dCf.scalar(it1, it4)
                        dCf_ = geom_.dCf.vec(it1, it4)
                        for it5 in range(0, 3):
                            for it6 in range(0, 3):
                                a_neigh_coef[it5][it6] += w_ * dCf_[it5] * dCf_[it6]
                            b_neigh_coef[it5][0] += w_ * dCf_[it5] * (value_.funit[it4].ltime[it2] - value_.cunit[it1].ltime[it2])
                for it3 in info_.cells[it1].bound:
                    if any([it4 in ['noslip', 'inlet', 'outlet'] for it4 in info_.faces[it3].group]):
                        w_ = 1 / geom_.dCf.scalar(it1, it3)
                        dCf_ = geom_.dCf.vec(it1, it3)
                        for it4 in range(0, 3):
                            for it5 in range(0, 3):
                                a_bound_coef[it4][it5] += w_ * dCf_[it4] * dCf_[it5]
                            b_bound_coef[it4][0] += w_ * dCf_[it4] * (value_.funit[it3].ltime[it2] - value_.cunit[it1].ltime[it2])
                
                a_coef = a_neigh_coef + a_bound_coef; b_coef = b_neigh_coef + b_bound_coef
                
                #try:
                c, low = linalg.cho_factor(a_coef)
                grad_ = linalg.cho_solve((c, low), b_coef)
                grad_ = grad_.ravel()
                #except:
                #    #grad_ = linalg.lstsq(a_coef, b_coef); grad_ = grad_[0]
                #    grad_ = value_.cgrad[it1].current

                value_.cgrad[it1].ltime.append(deepcopy(grad_))
        return value_      

    @staticmethod
    def get_layer_segments(info_: Info):
        k_fluid = 0.025
        ids = {'glass': {}, 'fluid': {}, 'abs': {}, 'insul': {}} # layer dict, tuple in z direction (glass - fluid - abs)
        for it1, it2 in info_.cells.items():
            if it2.layer not in list(ids[it2.group[0]].keys()):
                ids[it2.group[0]][it2.layer] = [it1]
            else:
                ids[it2.group[0]][it2.layer].append(it1)
        length_step = np.max(list(ids['glass'].keys()))
        depth_step = len(ids['glass'][0])
        ids_segment_layer = dict.fromkeys(list(range(0, length_step + 1)), [])
        for it1 in range(0, length_step + 1):
            entry_layer = []; entry_layer.append(ids['glass'][it1])
            entry_fluid = []; depth_fluid = []
            for it2 in ids['fluid'][it1]:
                depth_fluid.append(it2)
                if len(depth_fluid) == depth_step:
                    entry_fluid.append(depth_fluid)
                    depth_fluid = []
            entry_layer.extend(entry_fluid)
            entry_layer.append(ids['abs'][it1])
            entry_layer.append(ids['insul'][it1])
            ids_segment_layer[it1] = deepcopy(entry_layer)
        for it1, it2 in ids_segment_layer.items():
            ids_segment_layer[it1] = np.transpose(it2)
        return ids_segment_layer

    @staticmethod
    def get_face_grad(row_, col_, face_, value_, geom_, dCf_, tstep_, istemp: bool = False):
        if istemp is True:
            gradf_ = (value_.cunit[row_].ltime[tstep_] - value_.cunit[col_].ltime[tstep_]) / dCf_
        else:
            gradf_ = (value_.cunit[row_].ltime[tstep_] - value_.funit[face_].ltime[tstep_]) / dCf_
        eCf_ = geom_.dCf.norm(row_, face_)
        return gradf_ * eCf_

    def calc_locals(self, info_, geom_, poolTnew_, poolvnew_, poolpnew_, layer_segment_, layer_id,
                    delta_t, channel_length, channel_width, channel_height, plot=False):

        length_times = list(range(1, len(poolTnew_.cunit[0].ltime)))
        times_list = [delta_t/2]; times_list.extend([(x - 0.5) * delta_t for x in length_times[1:]])
        
        # averages
        vfluid_l = []; vstream_l = []
        Tfluid_l = []; Tstream_l = []
        Tglass_l = []; Tabs_l = []; Tinsul_l = []
        Pgradyfluid_l = []; Pgradystream_l = []
        vgradzfluid_l = []; Tgradzfluid_l = []

        for tstep_ in length_times:
            vfluid_part = []; vstream_part = []
            Tfluid_part = []; Tstream_part = []
            Tglass_part = []; Tabs_part = []; Tinsul_part = []
            Pgradyfluid_part = []; Pgradystream_part = []
            vgradzfluid_part = []; Tgradzfluid_part = []
            
            for layer_member, layer_set in enumerate(layer_segment_[layer_id]):                
                # find locals: 
                # v_avg, T_avg (calc of q max and functions)
                # Nu, Ra, q (heissler curve, free conv. evaluation)
                
                # find \int -grad^{z} (T)
                # z = 0:z_limit

                # layer_set list from glass [0] -> abs [-1]

                # fluid
                sample_range_ = layer_set[1:-2]
                middle_glass_range_ = math.floor(len(sample_range_)/2) - ( (len(sample_range_) + 1) % 2)
                middle_abs_range_ = math.floor(len(sample_range_)/2)
                
                stream_v_evaluate = [poolvnew_.cunit[x].ltime[tstep_] for x in sample_range_]
                stream_T_evaluate = [poolTnew_.cunit[x].ltime[tstep_] for x in sample_range_]
                
                stream_glass_v_evaluate = [x - np.min(stream_v_evaluate[:middle_glass_range_+1]) for x in stream_v_evaluate[:middle_glass_range_+1]]
                stream_abs_v_evaluate = [x - np.min(stream_v_evaluate[middle_abs_range_:]) for x in stream_v_evaluate[middle_abs_range_:]]
                
                try:
                    stream_glass_v_evaluate = [x for x in stream_glass_v_evaluate if x >= np.max(stream_glass_v_evaluate)]; bound_v_glass = np.min([len(stream_glass_v_evaluate), middle_glass_range_])
                except:
                    bound_v_glass = 0
                try:                    
                    stream_abs_v_evaluate = [x for x in stream_abs_v_evaluate if x < np.max(stream_abs_v_evaluate)]; bound_v_abs = np.max([len(stream_abs_v_evaluate) + middle_abs_range_, middle_abs_range_])
                except:
                    bound_v_abs = len(sample_range_)-1
                
                stream_glass_T_evaluate = [x - np.min(stream_T_evaluate[:middle_glass_range_+1]) for x in stream_T_evaluate[:middle_glass_range_+1]]
                stream_abs_T_evaluate = [x - np.min(stream_T_evaluate[middle_abs_range_:]) for x in stream_T_evaluate[middle_abs_range_:]]
                
                try:
                    stream_glass_T_evaluate = [x for x in stream_glass_T_evaluate if x/np.max(stream_glass_T_evaluate) >= 0.25]; bound_T_glass = np.min([len(stream_glass_T_evaluate), middle_glass_range_])
                except:
                    bound_T_glass = 0
                try:
                    stream_abs_T_evaluate = [x for x in stream_abs_T_evaluate if x/np.max(stream_abs_T_evaluate) < 0.25]; bound_T_abs = np.max([len(stream_abs_T_evaluate) + middle_abs_range_, middle_abs_range_])
                except:
                    bound_T_abs = len(sample_range_)-1
                
                wall_face_glass = list(info_.cells[layer_set[0]].conj.values())[0]
                wall_face_abs = list(info_.cells[layer_set[-2]].conj.values())[0]
                
                wall_face_glass_v_dist = info_.cells[ sample_range_[bound_v_glass] ].loc[2] - info_.faces[wall_face_glass].loc[2]
                wall_face_abs_v_dist = info_.cells[ sample_range_[bound_v_abs] ].loc[2] - info_.faces[wall_face_abs].loc[2]
                
                wall_face_glass_T_dist = info_.cells[ sample_range_[bound_T_glass] ].loc[2] - info_.faces[wall_face_glass].loc[2]
                wall_face_abs_T_dist = info_.cells[ sample_range_[bound_T_abs] ].loc[2] - info_.faces[wall_face_abs].loc[2]
                
                zstream_v = info_.cells[sample_range_[bound_v_abs]].loc[2] - info_.cells[sample_range_[bound_v_glass]].loc[2]
                zstream_T = info_.cells[sample_range_[bound_T_abs]].loc[2] - info_.cells[sample_range_[bound_T_glass]].loc[2]
                zfluid = np.abs(info_.faces[wall_face_abs].loc[2] - info_.faces[wall_face_glass].loc[2])
                
                # v stream trapezoid
                if bound_v_glass != bound_v_abs:
                    loc_v_stream = []
                    vstream_trap = []
                    Pgradystream_trap = []
                    
                    for it1 in range(bound_v_glass, bound_v_abs+1):
                        current_cell_id = sample_range_[it1]
                        loc_v_stream.append(info_.cells[current_cell_id].loc[2])
                        vstream_trap.append(poolvnew_.cunit[current_cell_id].ltime[tstep_])
                        Pgradystream_trap.append(poolpnew_.cgrad[current_cell_id].ltime[tstep_][1])

                    vstream_get = np.abs(trapezoid(vstream_trap, loc_v_stream) / zstream_v)
                    Pgradystream_get = trapezoid(Pgradystream_trap, loc_v_stream) / zstream_v
                else:
                    vstream_get = poolvnew_.cunit[sample_range_[bound_v_glass]].ltime[tstep_]
                    Pgradystream_get = poolpnew_.cgrad[sample_range_[bound_v_glass]].ltime[tstep_][1]
                
                # T stream trapezoid
                if bound_T_glass != bound_T_abs:
                    loc_T_stream = []
                    Tstream_trap = []
                    
                    for it1 in range(bound_T_glass, bound_T_abs+1):
                        current_cell_id = sample_range_[it1]
                        loc_T_stream.append(info_.cells[current_cell_id].loc[2])
                        Tstream_trap.append(poolTnew_.cunit[current_cell_id].ltime[tstep_])
                    
                    Tstream_get = np.abs(trapezoid(Tstream_trap, loc_T_stream) / zstream_T)
                else:
                    Tstream_get = poolTnew_.cunit[sample_range_[bound_T_glass]].ltime[tstep_]

                # v and T fluid trapezoid
                Twall_ref = poolTnew_.funit[wall_face_abs].ltime[tstep_]
                
                loc_fluid = [info_.faces[wall_face_glass].loc[2]]
                vfluid_trap = [0]
                Pgradyfluid_trap = [poolpnew_.fgrad[wall_face_glass].ltime[tstep_][1]]
                Tfluid_trap = [poolTnew_.funit[wall_face_glass].ltime[tstep_]]
                Tgradzfluid_trap = [poolTnew_.funit[wall_face_abs].ltime[tstep_] - poolTnew_.funit[wall_face_glass].ltime[tstep_]]
                
                for it1 in sample_range_:
                    loc_fluid.append(info_.cells[it1].loc[2])
                    vfluid_trap.append(poolvnew_.cunit[it1].ltime[tstep_])
                    Pgradyfluid_trap.append(poolpnew_.cgrad[it1].ltime[tstep_][1])
                    Tfluid_trap.append(poolTnew_.cunit[it1].ltime[tstep_])
                    Tgradzfluid_trap.append(poolTnew_.funit[wall_face_abs].ltime[tstep_] - poolTnew_.cunit[it1].ltime[tstep_])
                
                loc_fluid.append(info_.faces[wall_face_abs].loc[2])
                vfluid_trap.append(0)
                Pgradyfluid_trap.append(poolpnew_.fgrad[wall_face_abs].ltime[tstep_][1])
                Tfluid_trap.append(poolTnew_.funit[wall_face_abs].ltime[tstep_])
                Tgradzfluid_trap.append(0)
                
                # vgradzfluid
                loc_vgradzfluid = [info_.faces[wall_face_glass].loc[2], info_.cells[ sample_range_[bound_v_glass] ].loc[2],
                                   (info_.faces[wall_face_glass].loc[2] + info_.faces[wall_face_abs].loc[2]) / 2,
                                   info_.cells[ sample_range_[bound_v_abs] ].loc[2],
                                   info_.faces[wall_face_abs].loc[2]]
                vgradzfluid_trap = [0, poolvnew_.cunit[ sample_range_[bound_v_glass] ].ltime[tstep_],
                                    vstream_get, poolvnew_.cunit[ sample_range_[bound_v_abs] ].ltime[tstep_], 0]
                
                vgradzfluid_get = (((poolvnew_.cunit[ sample_range_[bound_v_glass] ].ltime[tstep_] + poolvnew_.cunit[ sample_range_[bound_v_abs] ].ltime[tstep_]) * zfluid / 2) - \
                                    np.abs(trapezoid(vgradzfluid_trap, loc_vgradzfluid))) / pow(zfluid, 2)
                # ganti gradzfluid_get jadi rasio
                Tgradzfluid_get = (((poolTnew_.funit[wall_face_abs].ltime[tstep_] - Tstream_get) * zfluid) - np.abs(trapezoid(Tgradzfluid_trap, loc_fluid))) / \
                                    (pow(zfluid, 2) * (poolTnew_.funit[wall_face_abs].ltime[tstep_] - Tstream_get))
                # 2 * pow(zfluid, 2)
                # pow(zfluid, 2)
                
                vfluid_get = np.abs(trapezoid(vfluid_trap, loc_fluid) / zfluid)
                Pgradyfluid_get = np.abs(trapezoid(Pgradyfluid_trap, loc_fluid) / zfluid)
                Tfluid_get = np.max([np.abs(trapezoid(Tfluid_trap, loc_fluid) / zfluid), Tstream_get])
                
                vfluid_part.append(vfluid_get); vstream_part.append(vstream_get)
                Tfluid_part.append(Tfluid_get); Tstream_part.append(Tstream_get)
                Tglass_part.append(poolTnew_.cunit[layer_set[0]].ltime[tstep_]); Tabs_part.append(poolTnew_.cunit[layer_set[-2]].ltime[tstep_]); Tinsul_part.append(poolTnew_.cunit[layer_set[-1]].ltime[tstep_])
                Pgradyfluid_part.append(Pgradyfluid_get); Pgradystream_part.append(Pgradystream_get)
                vgradzfluid_part.append(vgradzfluid_get); Tgradzfluid_part.append(Tgradzfluid_get)
            
            vfluid_l.append(np.mean(vfluid_part)); vstream_l.append(np.mean(vstream_part))
            Tfluid_l.append(np.mean(Tfluid_part)); Tstream_l.append(np.mean(Tstream_part))
            Tglass_l.append(np.mean(Tglass_part)); Tabs_l.append(np.mean(Tabs_part)); Tinsul_l.append(np.mean(Tinsul_part))
            Pgradyfluid_l.append(np.mean(Pgradyfluid_part)); Pgradystream_l.append(np.mean(Pgradystream_part))
            vgradzfluid_l.append(np.mean(vgradzfluid_part)); Tgradzfluid_l.append(np.mean(Tgradzfluid_part))

        return  vfluid_l, vstream_l, \
                Tfluid_l, Tstream_l, \
                Tglass_l, Tabs_l, Tinsul_l, \
                Pgradyfluid_l, Pgradystream_l, \
                vgradzfluid_l, \
                Tgradzfluid_l, \
                times_list

    @property
    def locals(self):
        return self.__locals
    @property
    def averages(self):
        return self.__averages
    @property
    def times(self):
        return self.__times

def table_analysis(caseloc, mesh_id, irr_, csvname):
    columns_analysis = {'v': ['fluid', 'stream'],
                        'T': ['fluid', 'stream', 'glass', 'abs', 'insul'],
                        'Pgrady': ['fluid', 'stream'],
                        'vgradz': ['fluid'],
                        'Tgradz': ['fluid']}
    columns_ = ['case', 'irr', 'times']
    for it1, it2 in columns_analysis.items():
        columns_.extend([f'{it1}_{x}' for x in it2])
    [columns_.append(f'rmsr_{x}') for x in ['u', 'v', 'w', 'T', 'k', 'omega']]
    [columns_.append(f'cfl_{x}') for x in ['u', 'v', 'w', 'T', 'k', 'omega']]
    
    all_entry_ = []
    for it1 in mesh_id:
        analysisdoc = gzip.open(f'{caseloc}\\case{it1}_irr{irr_}\\case{it1}_irr{irr_}_analysis.pklz', 'rb'); analysis_ = pickle.load(analysisdoc); analysisdoc.close()
        # reportdoc = open(f'{caseloc}\\case{it1}_irr{irr_}\\case{it1}_irr{irr_}_report', 'rb'); report_ = pickle.load(reportdoc); reportdoc.close()
        report_ = pd.read_csv(f'{caseloc}\\case{it1}_irr{irr_}\\case{it1}_irr{irr_}_report.csv')
        
        id_entry_ = [np.array([it1]*len(analysis_.times)), np.array([irr_]*len(analysis_.times)), np.array(analysis_.times)]
        for it3, it4 in columns_analysis.items():
            [id_entry_.append(analysis_.averages[it3][x]) for x in it4]
        
        [id_entry_.append(report_.loc[:, f'rmsr_per_time_{x}']) for x in ['u', 'v', 'w', 'T', 'k', 'omega']]
        [id_entry_.append(report_.loc[:, f'max_CFL_{x}']) for x in ['u', 'v', 'w', 'T', 'k', 'omega']]
        
        id_entry_ = np.transpose(id_entry_)
        all_entry_.extend(deepcopy(id_entry_))

    data_ = pd.DataFrame(columns=columns_, data = all_entry_)

    if not os.path.exists(f'{caseloc}\\tabel'):
        try:
            original_umask = os.umask(0)
            os.makedirs(f'{caseloc}\\tabel', 0o777)
        finally:
            os.umask(original_umask)
    os.chdir(f'{caseloc}\\tabel')

    data_.to_csv(f'hasil_{csvname}.csv', index = False)

    return

# exec
caseloc = os.getcwd() + '\\data'
mesh_id = list(range(1,10))
channel_H = [0.27, 0.27, 0.88, 0.88, 1, 0.15, 0.575, 0.575, 0.575]
channel_W = [1.22, 2.28, 1.22, 2.28, 1.75, 1.75, 2.5, 1, 1.75]
channel_L = 2.5
nlayer_ = 10
delta_t = 60
irr_ = 500

for get1, get2 in enumerate(mesh_id):
    print(f'mesh {get2}, irr {irr_}')
    infodoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_info.pklz', 'rb'); info_ = pickle.load(infodoc); infodoc.close()
    geomdoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_geom.pklz', 'rb'); geom_ = pickle.load(geomdoc); geomdoc.close()
    pooldoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_pool.pklz', 'rb'); pool_ = pickle.load(pooldoc); pooldoc.close()
    
    # poolTnew_ = analysis.time_least_square_gradient(info_, geom_, pool_.T)
    # poolvnew_ = analysis.time_least_square_gradient(info_, geom_, pool_.v)
    # poolpnew_ = analysis.time_least_square_gradient(info_, geom_, pool_.P)

    poolTdoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_poolT.pklz', 'rb'); poolTnew_ = pickle.load(poolTdoc); poolTdoc.close()
    poolvdoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_poolv.pklz', 'rb'); poolvnew_ = pickle.load(poolvdoc); poolvdoc.close()
    poolpdoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_poolP.pklz', 'rb'); poolpnew_ = pickle.load(poolpdoc); poolpdoc.close()

    get_ = analysis(info_, geom_, poolTnew_, poolvnew_, poolpnew_, nlayer_, delta_t, channel_L, channel_W[get1], channel_H[get1], plot = False)
    analysisdoc = gzip.open(f'{caseloc}\\case{get2}_irr{irr_}\\case{get2}_irr{irr_}_analysis.pklz', 'wb'); pickle.dump(get_, analysisdoc); analysisdoc.close()

table_analysis(caseloc, mesh_id, irr_, '12-11-23')

