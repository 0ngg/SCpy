##### cfd scheme ##########
import numpy as np; import pandas as pd
import math; import os
import meshio
import scipy.sparse as sparse
from decimal import *
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns

getcontext().prec = 16

def match_duplicates(meshname : str):
    __mesh = meshio.read(os.getcwd() + "\\case\\test\\" + meshname)
    # points aliases
    points_sets = []; points_uniques = []; points_product = []; points_alias = []
    for it1 in __mesh.points:
        points_sets.append(set(it1))
    for it1 in np.transpose(__mesh.points):
        points_uniques.append(list(np.unique(it1)))
    for it1 in itertools.product(*points_uniques):
        points_product.append(set(it1))
    for it1 in points_product:
        dup_id = np.where(np.array(points_sets) == it1)[0]
        if dup_id.shape[0] > 1:
            points_alias.extend([tuple([dup_id[0], dup_id[it2]]) for it2 in range(1, dup_id.shape[0])])
        else:
            pass
    # extract point id reordering dict
    points_reorder = dict({}); ctd = 0
    for it1 in range(0, len(points_sets)):
        check = [it2 for it2 in points_alias if it1 in it2]
        if len(check) != 0:
            if it1 not in list(points_reorder.keys()):
                points_reorder[check[0][0]] = ctd
                points_reorder[check[0][1]] = ctd
                ctd += 1
            else:
                pass       
        else:
            points_reorder[it1] = ctd
            ctd += 1
    # recreate __mesh.points with no duplicates
    points_nodup = []; prev_it2 = -1
    for it1, it2 in points_reorder.items():
        if it2 != prev_it2:
            points_nodup.append(list(__mesh.points[it1]))
            prev_it2 = it2
        else:
            pass
    points_nodup = np.array(points_nodup)
    # reorder __mesh.cells_dict
    cells_dict_nodup = dict({})
    for it1 in list(__mesh.cells_dict.keys()):
        cells_dict_nodup[it1] = []
        for it2 in range(0, __mesh.cells_dict[it1].shape[0]):
            cells_dict_nodup[it1].append([points_reorder[it3] for it3 in __mesh.cells_dict[it1][it2]])
        cells_dict_nodup[it1] = np.array(cells_dict_nodup[it1])
    return points_nodup, cells_dict_nodup, __mesh.cell_sets_dict
def match_physical(*args):
    # args __mesh.cells_dict, __mesh.cell_sets_dict
    # parse physical to points dict
    domain_phys = dict({}) # domain
    bound_phys = dict({}) # boundary
    names = list(args[1].keys())[:-1]
    for it1 in names:
        token = it1.split("_")
        if any([it2 in token for it2 in ["fluid", "solid"]]) is True:
            # domain data only takes one string
            if it1 not in list(domain_phys.keys()):
                domain_phys[it1] = []
            else:
                pass
            for it2, it3 in args[1][it1].items():
                domain_phys[it1].extend(args[0][it2][it3])
        else:
            # boundary data in list
            for it2 in token:
                if it2 not in list(bound_phys.keys()):
                    bound_phys[it2] = []
                else:
                    pass
                for it3, it4 in args[1][it1].items():
                    for it5 in it4:
                        bound_phys[it2].extend(args[0][it3][it5])
    for it1 in list(domain_phys.keys()):
        domain_phys[it1] = list(np.unique(np.array(domain_phys[it1])))
    for it1 in list(bound_phys.keys()):
        bound_phys[it1] = list(np.unique(np.array(bound_phys[it1])))
    return bound_phys, domain_phys
def match_face_cell(*args):
    # args __mesh.cells_dict, __mesh.cell_sets_dict
    # parse faces (triangle and quads only for now)
    face_dict = dict({}) # face obj id: [node list, bound list]
    cell_dict = dict({}) # cell obj id: [node list, face list, neigh list, domain list]
    # physical nodes
    bound_dict, domain_dict = match_physical(args[0], args[1])
    # gmsh specified boundary faces and domain cells
    name_face = [it1 for it1 in list(args[0].keys()) if \
                all([args[0][it1][0].shape[0] >= 3,
                args[0][it1][0].shape[0] <= 4]) is True]
    id_modif = 0
    for it1 in name_face:
        for it2 in range(0, args[0][it1].shape[0]):
            get_nodes = list(args[0][it1][it2])
            get_bound = [it3 for it3 in list(bound_dict.keys()) if \
                        all(list(map(lambda it4: it4 in bound_dict[it3], get_nodes))) is True]
            get_args = [list(args[0][it1][it2]), get_bound]
            face_dict[it2 + id_modif] = get_args
        id_modif += args[0][it1].shape[0]
    name_cell = [it1 for it1 in list(args[0].keys()) if \
                all([args[0][it1][0].shape[0] >= 5]) is True]
    id_modif = 0
    for it1 in name_cell:
        for it2 in range(0, args[0][it1].shape[0]):
            get_nodes = list(args[0][it1][it2])
            get_faces = [it3 for it3, it4 in face_dict.items() if all([it5 in get_nodes for it5 in it4[0]]) is True]
            get_domain = [it3 for it3 in list(domain_dict.keys()) if \
                            all(list(map(lambda it4: it4 in domain_dict[it3], get_nodes))) is True]
            get_args = [list(args[0][it1][it2]), get_faces, get_domain]
            cell_dict[it2 + id_modif] = get_args
        id_modif += args[0][it1].shape[0]
    # partition faces ("none"), found with neighboring cells
    prev_face_len = len(list(face_dict.keys()))
    for it1 in range(0, len(list(cell_dict.keys())) - 1):
        for it2 in range(it1 + 1, len(list(cell_dict.keys()))):
            check_part = [it3 in cell_dict[it2][0] for it3 in cell_dict[it1][0]]
            part_face_nodes = [it4 for it3, it4 in zip(check_part, cell_dict[it1][0]) if it3 is True]
            if len(part_face_nodes) >= 3:
                in_face_dict = False
                for it3 in range(0, prev_face_len):
                    if all([it4 in face_dict[it3][0] for it4 in part_face_nodes]) is True:
                        in_face_dict = True
                        break
                if in_face_dict is False:
                    face_dict[len(list(face_dict.keys()))] = [list(part_face_nodes), ["none"]]
            else:
                continue
    # append partition faces to cell faces
    for it1 in range(0, len(list(cell_dict.keys()))):
        for it2 in range(prev_face_len, len(list(face_dict))):
            if all([it3 in cell_dict[it1][0] for it3 in face_dict[it2][0]]) is True:
                cell_dict[it1][1].append(it2)
    # find cell neighbors and boundary neighbors
    neigh_list = []; bound_list = []
    for it1 in range(0, len(list(cell_dict.keys())) - 1):
        cell_neigh = []
        for it2 in range(it1 + 1, len(list(cell_dict.keys()))):
            is_neigh = [it3 for it3 in cell_dict[it1][1] if it3 in cell_dict[it2][1]]
            if len(is_neigh) != 0:
                cell_neigh.append([it1, it2, is_neigh[0]])
                cell_neigh.append([it2, it1, is_neigh[0]])
        is_bound = [it2 for it2 in cell_dict[it1][1] if it2 not in np.transpose(np.array(cell_neigh))[2]]
        neigh_list.extend(cell_neigh); bound_list.extend([[it1, it2, 0] for it2 in is_bound])
    return face_dict, cell_dict, neigh_list, bound_list
class face:
    def __init__(self, points_dict : dict, *args):
        # args new_nodes : list, new_bound : list
        self.__nodes = args[0]
        self.__centroid, self.__area = self.get_info(points_dict)
        self.__boundary = args[1]
    @property
    def nodes(self):
        return self.__nodes
    @property
    def area(self):
        return self.__area
    @property
    def centroid(self):
        return self.__centroid
    @property
    def boundary(self):
        return self.__boundary

    def get_info(self, points_dict : dict):
        get_area = 0.00
        get_centroid = [0.00, 0.00, 0.00]
        if len(self.nodes) == 3:
            get_coor = [[it1 for it1 in points_dict[it2]] for it2 in self.nodes]
            get_area = np.sqrt(np.sum(list(map(lambda x:x**2, np.cross(np.array(get_coor[1] - get_coor[0]),
                                               np.array(get_coor[2] - get_coor[0])))))) / 2
            get_centroid = (get_coor[0] + get_coor[1] + get_coor[2]) / 3
        elif len(self.nodes) == 4:
            diag1 = [0, 0]; diag2 = [it1 for it1 in self.nodes]; rhs = 0.00
            for it1 in range(0, len(self.nodes) - 1):
                for it2 in range(it1, len(self.nodes)):
                    rhs_check = np.sqrt(np.sum(list(map(lambda x:x**2, np.array(points_dict[self.nodes[it1]] - \
                                np.array(points_dict[self.nodes[it2]]))))))
                    if rhs_check > rhs:
                        rhs = rhs_check; diag1[0] = self.nodes[it1]; diag1[1] = self.nodes[it2]
            diag2.remove(diag1[0]); diag2.remove(diag1[1])
            for it1 in diag1:
                get_coor = np.array([[it2 for it2 in points_dict[it1]], 
                            [it2 for it2 in points_dict[diag2[0]]],
                            [it2 for it2 in points_dict[diag2[1]]]])
                part_area = np.sqrt(np.sum(list(map(lambda x:x**2, np.cross(get_coor[1] - get_coor[0],
                            get_coor[2] - get_coor[0]))))) / 2
                part_centroid = (get_coor[0] + get_coor[1] + get_coor[2]) / 3
                get_area += part_area; get_centroid += part_centroid * part_area
            get_centroid = get_centroid / get_area
        get_centroid = [round(it1, 3) for it1 in get_centroid]; get_area = round(get_area, 3)
        return np.array(get_centroid), get_area
    def print_elements(self):
        print("nodes: {}\ncentroid: {}\narea: {}\nboundary: {}".format(self.nodes, self.centroid, self.area, self.boundary))
        return
class cell:
    def __init__(self, points_dict : dict, face_dict : dict, *args):
        # args new_nodes : list, new_faces : list, new_domain : list
        self.__nodes = args[0]
        self.__faces = args[1]
        self.__centroid, self.__volume = self.get_info(points_dict, face_dict)
        self.__domain = args[2]
    @property
    def nodes(self):
        return self.__nodes
    @property
    def faces(self):
        return self.__faces
    @property
    def volume(self):
        return self.__volume
    @property
    def centroid(self):
        return self.__centroid
    @property
    def domain(self):
        return self.__domain

    def get_info(self, points_dict : dict, face_dict : dict):
        get_volume = 0.00
        get_centroid = [0, 0, 0]
        centre = np.array([0, 0, 0], dtype = float)
        for it1 in self.nodes:
            centre += np.array([it2 for it2 in points_dict[it1]])
        centre = centre / len(self.nodes)
        for it1 in self.faces:
            get_fC = np.array([it2 for it2 in np.array(centre - face_dict[it1].centroid)])
            get_fC_val = np.sqrt(np.sum(list(map(lambda it2: it2**2, get_fC))))
            vec1 = np.array(points_dict[face_dict[it1].nodes[2]] - points_dict[face_dict[it1].nodes[0]])
            vec2 = np.array(points_dict[face_dict[it1].nodes[1]] - points_dict[face_dict[it1].nodes[0]])
            get_Sf = np.array([it2 for it2 in np.cross(vec1, vec2)])
            get_Sf_val = np.sqrt(np.sum(list(map(lambda it2: it2**2, get_Sf))))
            if np.dot(get_Sf, get_fC) > 0.00:
                get_Sf = get_Sf / get_Sf_val
            else:
                get_Sf = -1 * get_Sf / get_Sf_val
            cos = np.dot(get_Sf, get_fC) / get_fC_val
            part_height = cos * get_fC_val
            part_volume = part_height * face_dict[it1].area / 3
            part_centroid = np.array([it2 for it2 in face_dict[it1].centroid]) + 0.25 * get_fC
            get_volume += part_volume; get_centroid += part_centroid * part_volume
        get_centroid = get_centroid / get_volume
        get_centroid = [round(it1, 3) for it1 in get_centroid]; get_volume = round(get_volume, 3)
        return np.array(get_centroid), get_volume
    def print_elements(self):
        print("nodes: {}\nfaces: {}\ncentroid: {}\nvolume: {}\ndomain: {}".format(self.nodes, self.faces, self.centroid, self.volume, self.domain))
        return
class clust:
    def __init__(self, face_dict : dict, face_list : list, clust_area):
        # args face_list : list, clust_area : Decimal()
        self.__faces = face_list
        self.__area = clust_area
        self.__centroid = self.get_info(face_dict)
    @property
    def faces(self):
        return self.__faces
    @property
    def area(self):
        return self.__area
    @property
    def centroid(self):
        return self.__centroid
    @staticmethod
    def match_view(points_dict : dict, face_dict : dict, clust1, clust2):
        view = 0.00
        r = clust2.centroid - clust1.centroid
        r_val = np.sqrt(np.sum(np.array([it1**2 for it1 in r])))
        for it1 in clust1.faces:
            ef1 = np.cross(points_dict[face_dict[it1].nodes[2]] - points_dict[face_dict[it1].nodes[0]], 
                            points_dict[face_dict[it1].nodes[1]] - points_dict[face_dict[it1].nodes[0]])
            ef1 = np.array([it2 for it2 in ef1])
            ef1_val = round(np.sqrt(np.sum(np.array([it2**2 for it2 in ef1]))), 3)
            ef1 = ef1 / ef1_val
            if np.dot(ef1, r) < 0:
                ef1 = -1 * ef1
            cos1 = np.dot(ef1, r) / r_val
            for it2 in clust2.faces:
                ef2 = np.cross(points_dict[face_dict[it2].nodes[2]] - points_dict[face_dict[it2].nodes[0]], 
                                points_dict[face_dict[it2].nodes[1]] - points_dict[face_dict[it2].nodes[0]])
                ef2 = np.array([it2 for it2 in ef2])
                ef2_val = round(np.sqrt(np.sum(np.array([it3**2 for it3 in ef2]))), 3)
                ef2 = ef2 / ef2_val
                if np.dot(ef2, r) > 0.00:
                    ef2 = -1 * ef2
                cos2 = np.dot(ef2, -1 * r) / r_val
                view += cos1 * cos2 / (math.pi * r_val**2)
        view1 = view / clust1.area  
        view2 = view / clust2.area
        return view1, view2

    def get_info(self, face_dict : dict):
        # centroid calc
        centroid = np.array([0.00, 0.00, 0.00])
        for it1 in self.faces:
            centroid += face_dict[it1].centroid * face_dict[it1].area
        centroid = centroid / self.__area
        centroid = np.array([round(it1, 3) for it1 in centroid])
        return centroid
    def print_elements(self):
        print("faces: {}\narea: {}\ncentroid: {}".format(self.faces, self.area, self.centroid))
        return
class connect:
    def __init__(self, cc_args, fc_args):
        for it1 in list(cc_args.keys()):
            cc_args[it1] = np.array(cc_args[it1])
        for it2 in list(fc_args.keys()):
            fc_args[it2] = np.array(fc_args[it2])
        self.__cc = cc_args
        self.__fc = fc_args
    @classmethod
    def copy(self, other):
        self.__cc = other.__cc.copy()
        self.__fc = other.__fc.copy()
        return self
    @property
    def cc(self):
        return self.__cc
    @property
    def fc(self):
        return self.__fc

    def iter(self, which : str, what : str):
        check = str("_connect__" + which)
        if what in list(self.__dict__[check].keys()):
            toarray = np.transpose(self.__dict__[check][what])
            return toarray
        else:
            print("Key not found")
            return 0
    def print_elements(self):
        print("cc:\n")
        print(self.cc)
        print("fc:\n")
        print(self.fc)
        return
class geom:
    def __init__(self, points_dict : dict, face_dict : dict, cell_dict : dict, neigh_args : list):
        Sf_args = [[], [], []]; Ef_args = [[], [], []]; Tf_args = [[], [], []]
        dCf_args = [[], [], []]; eCf_args = [[], [], []]; dCF_args = [[], [], []]; eCF_args = [[], [], []]
        for it1, it2 in cell_dict.items():
            for it3 in it2.faces:
                Sf = np.cross(points_dict[face_dict[it3].nodes[2]] - points_dict[face_dict[it3].nodes[0]], 
                              points_dict[face_dict[it3].nodes[1]] - points_dict[face_dict[it3].nodes[0]])
                Sf = np.array([it4 for it4 in Sf])
                dCf = np.array([it4 for it4 in face_dict[it3].centroid - it2.centroid])
                dCf_val = np.sqrt(np.sum(np.array([it4**2 for it4 in dCf])))
                eCf = dCf / dCf_val
                if np.dot(Sf, dCf) < 0.00:
                    Sf = -1 * Sf
                Sf_args[0].append([it1, it3, Sf[0]]); Sf_args[1].append([it1, it3, Sf[1]]); Sf_args[2].append([it1, it3, Sf[2]])
                dCf_args[0].append([it1, it3, dCf[0]]); dCf_args[1].append([it1, it3, dCf[1]]); dCf_args[2].append([it1, it3, dCf[2]])
                eCf_args[0].append([it1, it3, eCf[0]]); eCf_args[1].append([it1, it3, eCf[1]]); eCf_args[2].append([it1, it3, eCf[2]])
        for it1 in neigh_args:
            dCF = np.array([it2 for it2 in cell_dict[it1[1]].centroid - cell_dict[it1[0]].centroid])
            dCF_val = np.sqrt(np.sum(np.array([it4**2 for it4 in dCf])))
            eCF = np.array([it2 for it2 in dCF / dCF_val])
            dCF_args[0].append([it1[0], it1[1], dCF[0]]); dCF_args[1].append([it1[0], it1[1], dCF[1]]); dCF_args[2].append([it1[0], it1[1], dCF[2]])
            eCF_args[0].append([it1[0], it1[1], eCF[0]]); eCF_args[1].append([it1[0], it1[1], eCF[1]]); eCF_args[2].append([it1[0], it1[1], eCF[2]])
            for it2 in range(0, len(Sf_args[0])):
                if all([Sf_args[0][it2][0] == it1[0], Sf_args[0][it2][1] == it1[2]]) is True:
                    Sf = np.array([Sf_args[0][it2][2], Sf_args[1][it2][2], Sf_args[2][it2][2]])
                    Ef = eCF * np.dot(Sf, Sf) / np.dot(eCF, Sf)
                    Ef = np.array([it3 for it3 in Ef])
                    Tf = Sf - Ef
                    Ef_args[0].append([it1[0], it1[2], Ef[0]]); Ef_args[1].append([it1[0], it1[2], Ef[1]]); Ef_args[2].append([it1[0], it1[2], Ef[2]])
                    Tf_args[0].append([it1[0], it1[2], Tf[0]]); Tf_args[1].append([it1[0], it1[2], Tf[1]]); Tf_args[2].append([it1[0], it1[2], Tf[2]])
        self.__Sf = np.array(Sf_args); self.__Ef = np.array(Ef_args); self.__Tf = np.array(Tf_args)
        self.__dCf = np.array(dCf_args); self.__eCf = np.array(eCf_args); self.__dCF = np.array(dCF_args); self.__eCF = np.array(eCF_args)
    def Sf(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__Sf[0].shape[0]):
            if all([self.__Sf[0][it1][0] == row, self.__Sf[0][it1][1] == col]) is True:
                coor = np.array([self.__Sf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def Ef(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__Ef[0].shape[0]):
            if all([self.__Ef[0][it1][0] == row, self.__Ef[0][it1][1] == col]) is True:
                coor = np.array([self.__Ef[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def Tf(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__Tf[0].shape[0]):
            if all([self.__Tf[0][it1][0] == row, self.__Tf[0][it1][1] == col]) is True:
                coor = np.array([self.__Tf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def dCf(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__dCf[0].shape[0]):
            if all([self.__dCf[0][it1][0] == row, self.__dCf[0][it1][1] == col]) is True:
                coor = np.array([self.__dCf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def eCf(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__eCf[0].shape[0]):
            if all([self.__eCf[0][it1][0] == row, self.__eCf[0][it1][1] == col]) is True:
                coor = np.array([self.__eCf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def dCF(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__dCF[0].shape[0]):
            if all([self.__dCF[0][it1][0] == row, self.__dCF[0][it1][1] == col]) is True:
                coor = np.array([self.__dCF[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def eCF(self, is_unit : bool, row : int, col : int):
        coor = [0, 0, 0]
        for it1 in range(0, self.__eCF[0].shape[0]):
            if all([self.__eCF[0][it1][0] == row, self.__eCF[0][it1][1] == col]) is True:
                coor = np.array([self.__eCF[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))
        else:
            return coor
    def print_elements(self):
        print("Sf: {}".format(self.Sf))
        print("Ef: {}".format(self.Ef))
        print("Tf: {}".format(self.Tf))
        print("dCf: {}".format(self.dCf))
        print("eCf: {}".format(self.eCf))
        print("dCF: {}".format(self.dCF))
        print("eCF: {}".format(self.eCF))
        return
class mesh:
    def __init__(self, meshname : str):
        # element object dict
        points_dict, __cells_dict, __cell_sets_dict = match_duplicates(meshname)
        face_args, cell_args, neigh_args, bound_args = match_face_cell(__cells_dict, __cell_sets_dict)
        face_dict = dict({}); cell_dict = dict({}); clust_dict = dict({})
        for it1, it2 in face_args.items():
            face_dict[it1] = face(points_dict, *it2)
        for it1, it2 in cell_args.items():
            cell_dict[it1] = cell(points_dict, face_dict, *it2)
        clust_args = dict({})
        for it1, it2 in face_dict.items():
            check_clust = np.array(["s2s" in it3 for it3 in it2.boundary])
            if any(check_clust) is True:
                clust_id = int(it2.boundary[np.where(check_clust == True)[0][0]][-1])
                if clust_id not in list(clust_args.keys()):
                    clust_args[clust_id] = [[it1], 0]
                else:
                    clust_args[clust_id][0].append(it1)
        if len(list(clust_args.keys())) > 0:
            for it1, it2 in clust_args.items():
                clust_args[it1][1] = np.sum(np.array([face_dict[it2].area for it2 in it2[0]]))
                clust_dict[it1] = clust(face_dict, clust_args[it1][0], clust_args[it1][1])
        # connect args
        cc_dict = dict({}); fc_dict = dict({})
        for it1 in neigh_args:
            domain1 = cell_dict[it1[0]].domain[0]; domain2 = cell_dict[it1[1]].domain[0]
            check_domain = ["fluid" in it2 for it2 in [domain1, domain2]]
            which_cc = ""
            if any(check_domain) is True:
                if all(check_domain) is True:
                    which_cc = "fluid"
                else:
                    which_cc = "conj"
            else:
                which_cc = "solid"
            if which_cc not in list(cc_dict.keys()):
                cc_dict[which_cc] = [it1]
            else:
                cc_dict[which_cc].append(it1)
        for it1 in bound_args:
            which_fc = [it2 for it2 in ["fluid", "solid"] if it2 in cell_dict[it1[0]].domain[0]]
            which_fc = which_fc[0]
            if which_fc not in list(fc_dict.keys()):
                fc_dict[which_fc] = [it1]
            else:
                fc_dict[which_fc].append(it1)
        if len(list(clust_args.keys())) > 0:
            cc_dict["s2s"] = []
            for it1 in range(0, len(list(clust_dict.keys())) - 1):
                for it2 in range(it1 + 1, len(list(clust_dict.keys()))):
                    view1, view2 = clust.match_view(points_dict, face_dict, clust_dict[list(clust_dict.keys())[it1]], clust_dict[list(clust_dict.keys())[it2]])
                    cc_dict["s2s"].append([list(clust_dict.keys())[it1], list(clust_dict.keys())[it2], view1])
                    cc_dict["s2s"].append([list(clust_dict.keys())[it2], list(clust_dict.keys())[it1], view2])
        connect_obj = connect(cc_dict, fc_dict)
        # geom args
        geom_obj = geom(points_dict, face_dict, cell_dict, neigh_args)
        self.__nodes = points_dict
        self.__faces = face_dict
        self.__cells = cell_dict
        self.__clusts = clust_dict
        self.__templates = connect_obj
        self.__geoms = geom_obj
    @property
    def nodes(self):
        return self.__nodes
    @property
    def faces(self):
        return self.__faces
    @property
    def cells(self):
        return self.__cells
    @property
    def clusts(self):
        return self.__clusts
    @property
    def templates(self):
        return self.__templates
    @property
    def geoms(self):
        return self.__geoms
    
    def visualize_domain(self, what : str):
        unique_points = []
        for it1 in self.templates.cc[what]:
            unique_points.extend(list(self.cells[it1[0]].nodes))
            unique_points.extend(list(self.cells[it1[1]].nodes))
        unique_points = list(np.unique(np.array(unique_points)))
        coor = []
        for it1 in unique_points:
            coor.append(list(self.nodes[it1]))
        coor = np.transpose(np.array(coor))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        ax.scatter(coor[0], coor[1], coor[2])
        fig.show()
        return
    def visualize_boundary(self, what : str):
        unique_points = []
        for it1 in self.templates.fc[what]:
            unique_points.extend(list(self.faces[it1[1]].nodes))
        unique_points = list(np.unique(np.array(unique_points)))
        coor = []
        for it1 in unique_points:
            coor.append(list(self.nodes[it1]))
        coor = np.transpose(np.array(coor))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = "3d")
        ax.scatter(coor[0], coor[1], coor[2])
        fig.show()
        return

##### cfd linear ##########
from numpy import linalg
import scipy.sparse as sparse
from CoolProp.HumidAirProp import HAPropsSI
# import mpi4py

class user:
    def __init__(self, foldername : str):
        # args folder name
        self.__inits = pd.read_csv(os.getcwd() + "\\case\\" + foldername + "\\csv\\init_values.csv")
        self.__solid_props = pd.read_csv(os.getcwd() + "\\case\\" + foldername + "\\csv\\solid_prop_values.csv", index_col = 0)
        self.__constants = pd.read_csv(os.getcwd() + "\\case\\" + foldername + "\\csv\\constant_values.csv")
        self.__sources = pd.read_csv(os.getcwd() + "\\case\\" + foldername + "\\csv\\source_values.csv")
    @property
    def inits(self):
        return self.__inits
    @property
    def solid_props(self):
        return self.__solid_props
    @property
    def constants(self):
        return self.__constants
    @property
    def sources(self):
        return self.__sources
class value:
    def __init__(self, mesh_ : mesh, user_ : user):
        # args mesh : mesh
        # all domains and clusts P, Pcor, u, v, w, k, e, T, q
        face_unit = dict({}); cell_unit = dict({})
        face_grad = dict({}); cell_grad = dict({})
        for it1 in ["P", "Pcor", "u", "v", "w", "k", "e", "T", "q"]:
            # grad is zero at init
            face_unit[it1] = np.full(shape=(len(list(mesh_.faces.keys())), 2), fill_value = np.full(shape=(1,2), fill_value = user_.inits.loc[0, it1]))
            cell_unit[it1] = np.full(shape=(len(list(mesh_.cells.keys())), 2), fill_value = np.full(shape=(1,2), fill_value = user_.inits.loc[0, it1]))
            face_grad[it1] = np.full(shape=(len(list(mesh_.faces.keys())), 2, 3), fill_value = np.full(shape=(2,3), fill_value = 0))
            cell_grad[it1] = np.full(shape=(len(list(mesh_.cells.keys())), 2, 3), fill_value = np.full(shape=(2,3), fill_value = 0))
        self.__cells = dict({"unit": cell_unit, "grad": cell_grad}); self.__faces = dict({"unit": face_unit, "grad": face_grad})
    def cells(self, which : str, what : str):
        return self.__cells[which][what]
    def set_cells(self, which : str, what : str, id : int, value, isprev = False, add = False):
        # only current value
        # args which : str, value : Decimal / list(Decimal), id : int, what : str
        if isprev is True:
            self.__cells[which][what][id][0] = self.__cells[which][what][id][1]
        else:
            if add is True:
                self.__cells[which][what][id][1] += value
            else:
                self.__cells[which][what][id][1] = value
    def faces(self, which : str, what : str):
        return self.__faces[which][what]
    def set_faces(self, which : str, what : str, id : int, value, isprev = False, add = False):
        if isprev is True:
            self.__faces[which][what][id][0] = self.__faces[which][what][id][1]
        else:
            if add is True:
                self.__faces[which][what][id][1] += value
            else:
                self.__faces[which][what][id][1] = value

    def linear_itr(self, which : str, mesh_ : mesh, id_ : list, what : str):
        # id [cell 1 id, cell 2 id, face id]
        # linear face value approx
        # current value only
        # neighbor faces only
        dCf1_ = mesh_.geoms.dCf(True, id_[0], id_[2]); dCf2_ = mesh_.geoms.dCf(True, id_[1], id_[2])
        gC_ = dCf1_ / (dCf1_ + dCf2_)
        return gC_ * self.cells(which, what)[id_[0]][1] + (1 - gC_) * self.cells(which, what)[id_[1]][1]
    def interpolate_grad(self, mesh_ : mesh, id_ : list, what : str):
        # id [cell 1 id, cell 2 id, face id]
        # face grad interpolation
        # current grad only
        # neighbor faces only
        dCF_ = mesh_.geoms.dCF(True, id_[0], id_[1])
        eCF_ = mesh_.geoms.eCF(False, id_[0], id_[1])
        lin_grad_ = self.linear_itr("grad", mesh_, id_, what)
        fgrad_ = lin_grad_ + eCF_ * self.cells("unit", what)[id_[1]][1] - self.cells("unit", what)[id_[0]][1] / dCF_ - np.dot(lin_grad_, eCF_)
        self.set_faces("grad", what, id_[2], fgrad_)
    def QUICK(self, mesh_ : mesh, id_ : list, what : str):
        # id [cell id, face id]
        # QUICK face value
        # current grad only
        # neighbor faces only
        dCf_ = mesh_.geoms.dCf(False, id_[0], id_[1])
        fvalue_ = self.cells("unit", what)[id_[0]][1] + np.dot(self.cells("grad", what)[id_[0]][1] + self.faces("grad", what)[id_[1]][1], dCf_) / 2
        self.set_faces("unit", what, id_[1], fvalue_)
    def least_square_itr(self, mesh_ : mesh, id_ : int, what : str):
        # calculate Cgrad with least square iter since neighbor face values cannot yet be calculated
        # given that fgrad can be interpolated
        # homogenic domains only, conj face values are specified
        lhs_ = np.zeros(shape=(3,3), dtype = float)
        rhs_ = np.zeros(shape=(1,3), dtype = float)[0]
        if "fluid" in mesh_.cells[id_].domain[0]:
            domain_ = "fluid"
        else:
            domain_ = "solid"
        neigh_face_id = np.where(mesh_.templates.iter("cc", domain_)[0] == id_)[0]
        neigh_list = dict(zip(mesh_.templates.iter("cc", domain_)[1][neigh_face_id],
                              mesh_.templates.iter("cc", domain_)[2][neigh_face_id]))
        for it1 in range(0, 3):
            weight_rhs_ = 0
            for it2 in range(0, 3):
                weight_lhs_ = 0
                for it3 in mesh_.cells[id_].faces:
                    dCf_ = mesh_.geoms.dCf(False, id_, it3)
                    weight_lhs_ += 1 / mesh_.geoms.dCf(True, id_, it3) * dCf_[it1] * dCf_[it2]
                lhs_[it1][it2] = float(weight_lhs_)
            for it2 in mesh_.cells[id_].faces:
                dCf_ = mesh_.geoms.dCf(False, id_, it3)
                if it2 in list(neigh_list.keys()):
                    weight_rhs_ += 1 / mesh_.geoms.dCf(True, id_, it3) * dCf_[it1] * \
                                   (self.cells("unit", what)[neigh_list[it2]][1] - self.cells("unit", what)[id_][1])
            rhs_[it1] = float(weight_rhs_)
        grad_, exitCode = sparse.linalg.gmres(lhs_, rhs_, tol = 0.05, maxiter = 500); grad_ = np.array([it1 for it1 in grad_])
        self.set_cells("grad", what, id_, grad_)
    def update_value(self, mesh_ : mesh, what : str, new_values : np.array):
        # new_values in np.array(shape=(1,n), dtype = Decimal)[0]
        print(new_values)
        res_ = 0
        for it1 in range(0, new_values.shape[0]):
            res_ += pow(self.cells("unit", what)[it1][1] - new_values[it1], 2)
        res_ = res_ / self.cells("unit", what).shape[0]
        for it1 in list(mesh_.cells.keys()):
            self.set_cells("unit", what, it1, new_values[it1])
        for it1 in list(mesh_.cells.keys()):
            self.least_square_itr(mesh_, it1, what)
        for it1, it2 in mesh_.faces.items():
            if "none" in it2.boundary:
                neigh_cell = []
                for it3, it4 in mesh_.cells.items():
                    if len(neigh_cell) == 2:
                        break
                    if it1 in it4.faces:
                        neigh_cell.append(it3)
                if len(neigh_cell) == 2:
                    self.interpolate_grad(mesh_, [neigh_cell[0], neigh_cell[1], it1], what)
                    self.QUICK(mesh_, [neigh_cell[0], it1], what)
                    print("neighbor unit: {}, grad: {}".format(self.faces("unit", "T")[it1][1], self.faces("grad", "T")[it1][1]))
                else:
                    self.QUICK(mesh_, [neigh_cell[0], it1], what)
                    print("adiabatic unit: {}, grad: {}".format(self.faces("unit", "T")[it1][1], self.faces("grad", "T")[it1][1]))
        return res_
    def calc_rmsr(self, what : str):
        # new - prev, time rmsr, cell values only
        res_ = 0
        for it1 in self.cells("unit", what):
            res_ += pow(it1[1] - it1[0], 2)
        res_ = res_ / self.cells("unit", what).shape[0]
        return res_
    def calc_prop(self, mesh_ : mesh, user_ : user, prop : str, where : str, id : int):
        val = 0
        if where == "faces":
            P_ = self.faces("unit", "P")[id][1]; T_ = self.faces("unit", "T")[id][1]
        elif where == "cells":
            P_ = self.cells("unit", "P")[id][1]; T_ = self.cells("unit", "T")[id][1]
        if T_ < 195:
            T_ = float(195)
        elif T_ > 623.13:
            T_ = float(623.13)
        W_ = float(user_.constants.loc[0, "Wamb"])
        if prop == "rho":
            rho_ = HAPropsSI("Vha", "P", float(P_), "T", float(T_), "W", float(W_))
            val = 1 / rho_
        elif prop == "miu":
            miu_ = HAPropsSI("mu", "P", float(P_), "T", float(T_), "W", float(W_)) 
            val = miu_
        elif prop == "cp":
            cp_ = HAPropsSI("cp_ha", "P", float(P_), "T", float(T_), "W", float(W_))
            val = cp_
        elif prop == "alpha":
            k_ = HAPropsSI("k", "P", float(P_), "T", float(T_), "W", float(W_))
            rho_ = HAPropsSI("Vha", "P", float(P_), "T", float(T_), "W", float(W_))
            cp_ = HAPropsSI("cp_ha", "P", float(P_), "T", float(T_), "W", float(W_))
            val = k_ * rho_ / cp_
        # solid constant
        elif prop == "k":
            if where == "cells":
                solid_ = mesh_.cells[id].domain[0]
            elif where == "faces":
                solid_ = [it1 for it1 in mesh_.faces[id].boundary if it1 in user_.solid_props.columns][0]
            val = user_.solid_props.loc["k", solid_]
        elif prop == "eps":
            if where == "cells":
                solid_ = mesh_.cells[id].domain[0]
            elif where == "faces":
                solid_ = [it1 for it1 in mesh_.faces[id].boundary if it1 in user_.solid_props.columns][0]
            val = user_.solid_props.loc["eps", solid_]
        return val

class linear:
    def __init__(self, mesh_ : mesh, what = ["fluid", "solid", "conj"]):
        self.__iter = what
        cc_size_lim = len(list(mesh_.cells.keys()))
        # lil matrix generator
        # cells will always have at least one neighboring cell
        # sparse can only store float values. float 0 as init
        self.__lhs = sparse.lil_matrix(np.zeros(shape=(cc_size_lim, cc_size_lim)), dtype = float)
        self.__rhs = sparse.lil_matrix(np.zeros(shape=(cc_size_lim, 1)), dtype = float)
    @property
    def iter(self):
        return self.__iter
    @property
    def lhs(self):
        return self.__lhs
    @property
    def rhs(self):
        return self.__rhs
    def set_lhs(self, id : list, val, add = False):
        if add is True:
            self.__lhs[int(id[0]), int(id[1])] += float(val)
        else:
            self.__lhs[int(id[0]), int(id[1])] = float(val)
    def set_rhs(self, id : list, val, add = False):
        if add is True:
            self.__rhs[int(id[0]), int(id[1])] += float(val)
        else:
            self.__rhs[int(id[0]), int(id[1])] = float(val)
        
    def calc_transient(self, mesh_ : mesh, user_ : user, value_ : value, what : str, time_step, current_time : int):
        # void
        # args mesh : mesh, what : str, time_step : int/double, current_time : int
        lhs_transient_ = deepcopy(self.__lhs)
        rhs_transient_ = deepcopy(self.__rhs)
        if current_time == 0:
            for it1 in range(0, lhs_transient_.shape[0]):
                for it2 in range(0, lhs_transient_.shape[0]):   
                    lhs_transient_[it1, it2] = lhs_transient_[it1, it2] + float(value_.calc_prop(mesh_, user_, "rho", "cells", it1)) \
                                               * float(mesh_.cells[it1].volume) / time_step
                rhs_transient_[it1, 0] = rhs_transient_[it1, 0] + float(mesh_.cells[it1].volume) * user_.inits.loc[0, what] / \
                                         (time_step * HAPropsSI("Vha", "P", float(user_.inits.loc[0, "P"]), "T", float(user_.inits.loc[0, "T"]), "W", float(user_.constants.loc[0, "Wamb"])))
        else:
            for it1 in range(0, lhs_transient_.shape[0]):
                for it2 in range(0, lhs_transient_.shape[0]):   
                    lhs_transient_[it1, it2] = lhs_transient_[it1, it2] + float(value_.calc_prop(mesh_, user_, "rho", "cells", it1)) \
                                               * float(mesh_.cells[it1].volume) / (2 * time_step)
                rhs_transient_[it1, 0] = rhs_transient_[it1, 0] + mesh_.cells[it1].volume * value_.cells("unit", what)[it1][0] / \
                                         (time_step * HAPropsSI("Vha", "P", float(value_.cells("unit", "P")[it1][0]), "T", float(value_.cells("unit", "T")[it1][0]), "W", float(user_.constants.loc[0, "Wamb"])))
        return lhs_transient_, rhs_transient_      
class pcorrect(linear):
    def __init__(self, mesh_ : mesh):
        super().__init__(mesh_, what = ["fluid"])

    def calc_coef(self, mesh_ : mesh, user_ : user, value_ : value, u_ref, v_ref, w_ref):
        aC_ = 0
        bC_ = 0
        cc_ = mesh_.templates.iter("cc", "fluid"); cc_ = sorted(zip(cc_[0], cc_[1], cc_[2]), key = lambda x: x[0])
        prev_row = cc_[0][0]
        for it1, it2, it3 in cc_:
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", it3)
            Df_x_ = ((mesh_.cells[it1].volume / u_ref.lhs[it1, it2]) + (mesh_.cells[it2].volume / u_ref.lhs[it1, it2])) / 2
            Df_y_ = ((mesh_.cells[it1].volume / v_ref.lhs[it1, it2]) + (mesh_.cells[it2].volume / v_ref.lhs[it1, it2])) / 2
            Df_z_ = ((mesh_.cells[it1].volume / w_ref.lhs[it1, it2]) + (mesh_.cells[it2].volume / w_ref.lhs[it1, it2])) / 2
            Sf_ = mesh_.geoms.Sf(False, it1, it3)
            dCF_ = mesh_.geoms.dCF(False, it1, it2)
            Dau_f_ = (pow(Df_x_ * Sf_[0], 2) + pow(Df_y_ * Sf_[1], 2) + pow(Df_z_ * Sf_[2], 2) / \
                     (dCF_[0] * Df_x_ * Sf_[0] + dCF_[1] * Df_y_ * Sf_[1] + dCF_[2] * Df_z_ * Sf_[2]))
            if prev_row == it1:
                aC_ += rho_f_ * Dau_f_
                bC_ += rho_f_ * np.dot(np.array([value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1],
                                       value_.faces("unit", "u")[it3][1]]), Sf_) - \
                                       np.dot(np.array([[Df_x_, 0, 0], [0, Df_y_, 0], [0, 0, Df_z_]]) * \
                                       value_.faces("grad", "u")[it3][1] - value_.linear_itr("grad", mesh_, [it1, it2, it3], "P"), Sf_) 
                self.set_lhs([it1, it2], -1 * rho_f_ * Dau_f_)
                prev_row = it1
            else:
                self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                aC_ = 0; bC_ = 0
                aC_ += rho_f_ * Dau_f_
                bC_ += rho_f_ * np.dot(np.array([value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1],
                                       value_.faces("unit", "w")[it3][1]]), Sf_) - \
                                       np.dot(np.array([[Df_x_, 0, 0], [0, Df_y_, 0], [0, 0, Df_z_]]) * \
                                       value_.faces("grad", "u")[it3][1] - value_.linear_itr("grad", mesh_, [it1, it2, it3], "P"), Sf_) 
                self.set_lhs([it1, it2], -1 * rho_f_ * Dau_f_)
                prev_row = it1
        self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            self.calc_bound(mesh_, user_, value_, it1, it2)
    def calc_bound(self, mesh_ : mesh, user_ : user, value_ : value, row : int, col : int):
        if "inlet" in mesh_.faces[col].boundary:
            rho_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", row)
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", col)
            cvalue_ = rho_f_ * self.lhs[row, row] / rho_C_
            self.set_lhs([row, row], cvalue_, add = True)
            # boundary P' values are set to zero
            # Pin = Pamb
        elif "outlet" in mesh_.faces[col].boundary:
            rho_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", row)
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", col)
            cvalue_ = rho_f_ * self.lhs[row, row] / rho_C_
            self.set_lhs([row, row], cvalue_, add = True)
            # boundary P' values are set to zero
            # Pout = noslip
    def calc_post_bound(self, mesh_ : mesh, user_ : user, value_ : value):
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            if "noslip" in mesh_.faces[it2].boundary:
                rho_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1)
                fvalue_ = value_.cells("unit", "P")[it1][1] + np.dot(value_.cells("grad", "P")[it1][1], mesh_.geoms.Sf(False, it1, it2)) / (self.lhs[it1, it1] / rho_C_)
                value_.set_faces("unit", "P", it2, fvalue_); value_.set_faces("grad", "P", it2, np.array([0, 0, 0]))
            elif "inlet" in mesh_.faces[it2].boundary:
                value_.set_faces("unit", "P", it2, user_.constants.loc[0, "Pamb"]); value_.set_faces("grad", "P", it2, np.array([0, 0, 0]))
            elif "outlet" in mesh_.faces[it2].boundary:
                rho_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1)
                fvalue_ = value_.cells("unit", "P")[it1][1] + np.dot(value_.cells("grad", "P")[it1][1], mesh_.geoms.Sf(False, it1, it2)) / (self.lhs[it1, it1] / rho_C_)
                value_.set_faces("unit", "P", it2, fvalue_); value_.set_faces("grad", "P", it2, np.array([0, 0, 0]))
    def calc_correct(self, mesh_ : mesh, user_ : user, value_ : value, u_ref, v_ref, w_ref):
        for it1 in list(mesh_.cells.keys()):
            rho_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1)
            pcor_C_ = -1 * rho_C_ * mesh_.cells[it1].volume * value_.cells("grad", "Pcor")[it1][1] / self.lhs[it1, it1]
            value_.set_cells("unit", "u", it1, pcor_C_[0], add = True)
            value_.set_cells("unit", "v", it1, pcor_C_[1], add = True)
            value_.set_cells("unit", "w", it1, pcor_C_[2], add = True)
            value_.set_cells("unit", "P", it1, value_.cells("unit", "Pcor")[it1][1], add = True)
        cc_ = mesh_.templates.iter("cc", "fluid")
        for it1, it2, it3 in zip(cc_[0], cc_[1], cc_[2]):
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", it3)
            Df_x_ = (mesh_.cells[it1].volume / u_ref.lhs[it1, it2] + mesh_.cells[it2].volume / u_ref.lhs[it1, it2]) / 2
            Df_y_ = (mesh_.cells[it1].volume / v_ref.lhs[it1, it2] + mesh_.cells[it2].volume / v_ref.lhs[it1, it2]) / 2
            Df_z_ = (mesh_.cells[it1].volume / w_ref.lhs[it1, it2] + mesh_.cells[it2].volume / w_ref.lhs[it1, it2]) / 2
            Sf_ = mesh_.geoms.Sf(False, it1, it3)
            pcor_f_ = -1 * rho_f_ * np.dot(np.array([[Df_x_, 0, 0], [0, Df_y_, 0], [0, 0, Df_z_]]) * value_.faces("grad", "Pcor")[it3][1], Sf_)
            value_.set_faces("unit", "u", it3, pcor_f_[0], add = True)
            value_.set_faces("unit", "v", it3, pcor_f_[1], add = True)
            value_.set_faces("unit", "w", it3, pcor_f_[2], add = True)
    def iter_solve(self, mesh_ : mesh, user_ : user, value_ : value, under_relax_, tol_, max_iter_, time_step_, current_time_, u_ref, v_ref, w_ref):
        self.calc_coef(mesh_, user_, value_, u_ref, v_ref, w_ref)
        lhs_transient_, rhs_transient_ = super().calc_transient(mesh_, user_, value_, "Pcor", time_step_, current_time_)
        under_relax_b = deepcopy(rhs_transient_)
        for it1 in range(0, lhs_transient_.shape[0]):
            lhs_transient_[it1, it1] = lhs_transient_[it1, it1] / under_relax_; under_relax_b[it1, 0] = lhs_transient_[it1, it1] * value_.cells("unit", "Pcor")[it1][1] * (1 - under_relax_)
        A = lambda x: sparse.linalg.spsolve(lhs_transient_, x)
        b = rhs_transient_ + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = tol_, maxiter = max_iter_)
        _ = value_.update_value(mesh_, "Pcor", x)
        self.calc_post_bound(mesh_, user_, value_)
        self.calc_correct(mesh_, user_, value_, u_ref, v_ref, w_ref)
        return
class momentum(linear):
    def __init__(self, mesh_ : mesh, axis_ : int):
        super().__init__(mesh_, what = ["fluid"])
        self.__axis = axis_
    @property
    def axis(self):
        return self.__axis
    
    def calc_coef(self, mesh_ : mesh, user_ : user, value_ : value):
        # fluid only
        aC_ = 0
        bC_ = 0
        coor_dict = {0: "u", 1: "v", 2: "w"}
        axes = np.delete(np.array([0, 1, 2]), self.axis)
        cc_ = mesh_.templates.iter("cc", "fluid"); cc_ = sorted(zip(cc_[0], cc_[1], cc_[2]), key = lambda x: x[0])
        prev_row = cc_[0][0]
        for it1, it2, it3 in cc_:
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", it3)
            miu_f_ = value_.calc_prop(mesh_, user_, "miu", "faces", it3)
            Sf_ = mesh_.geoms.Sf(False, it1, it3)
            dCf_ = mesh_.geoms.dCf(False, it1, it3)
            eCF_ = mesh_.geoms.eCF(False, it1, it2)
            dCF_ = mesh_.geoms.dCF(True, it1, it2)
            v_ = np.array([0, 0, 0])
            v_[self.axis] = value_.faces("unit", coor_dict[self.axis])[it3][1]
            v_[axes[0]] = value_.faces("unit", coor_dict[axes[0]])[it3][1]
            v_[axes[1]] = value_.faces("unit", coor_dict[axes[1]])[it3][1]
            Ret_ = rho_f_ * pow(value_.faces("unit", "k")[it3][1], 2) / (miu_f_ * value_.faces("unit", "e")[it3][1])
            cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_ / 50, 2))
            St_tensor_ = np.array([[value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]]])
            St_tensor_ = (St_tensor_ + np.transpose(St_tensor_)) * 0.5
            St_ = np.sqrt(np.dot(St_tensor_, St_tensor_))
            ts_ = np.min(np.array([value_.faces("unit", "k")[it3][1] / value_.faces("unit", "e")[it3][1], value_.calc_prop(mesh_, user_, "alpha", it3) / (pow(6, 0.5) * cmiu_ * St_)]))
            miut_ = rho_f_ * cmiu_ * value_.faces("unit", "k")[it3][1] * ts_
            graditr_ = value_.linear_itr("grad", mesh_, [it1, it2, it3], coor_dict[self.axis])
            if prev_row == it1:
                aC_ += (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                aC_ += (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                bC += np.dot(np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", coor_dict[self.axis])[it1][1] + graditr_), dCf_) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                bC += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * (miu_f_ + miut_)
                lhs_value = (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                self.set_lhs([it1, it2], lhs_value); self.set_lhs([it1, it2], lhs_value_diff, add = True)
                prev_row = it1
            else:
                if self.axis == 1:
                    bC_ += value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * 9.81 * mesh_.cells[prev_row].volume
                self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                aC_ = 0; bC_ = 0
                aC_ += (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                aC_ += (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                bC += np.dot(np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", coor_dict[self.axis])[it1][1] + graditr_), dCf_) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                bC += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * (miu_f_ + miut_)
                lhs_value = (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                self.set_lhs([it1, it2], lhs_value); self.set_lhs([it1, it2], lhs_value, add = True)
                prev_row = it1
        if self.axis == 1:
            bC_ += value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * 9.81 * mesh_.cells[prev_row].volume
        self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            v_ = np.array([0, 0, 0])
            v_[self.axis] = value_.faces("unit", coor_dict[self.axis])[it2][1]
            v_[axes[0]] = value_.faces("unit", coor_dict[axes[0]])[it2][1]
            v_[axes[1]] = value_.faces("unit", coor_dict[axes[1]])[it2][1]
            self.calc_bound(mesh_, user_, value_, v_, it1, it2)            
    def calc_bound(self, mesh_ : mesh, user_ : user, value_ : value, v_, row : int, col : int):
        coor_dict = dict({0: "u", 1: "v", 2: "w"})
        axes = np.delete(np.array([0,1,2]), self.axis)
        if "noslip" in mesh_.faces[col].boundary:
            rho_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", row)
            miu_f_ = value_.calc_prop(mesh_, user_, "miu", "faces", col)
            miu_C_ = value_.calc_prop(mesh_, user_, "miu", "cells", row)
            cgrad_this = value_.cells("grad", coor_dict[self.axis])[row][1]
            cunit_this = value_.cells("unit", coor_dict[self.axis])[row][1]
            cunit_axes0 = value_.cells("unit", coor_dict[axes[0]])[row][1]
            cunit_axes1 = value_.cells("unit", coor_dict[axes[1]])[row][1]
            funit_this = value_.faces("unit", coor_dict[self.axis])[col][1]
            funit_axes0 = value_.faces("unit", coor_dict[axes[0]])[col][1]
            funit_axes1 = value_.faces("unit", coor_dict[axes[1]])[col][1]
            dperp_ = pow(linalg.norm(cgrad_this)**2 + 2 * cgrad_this, 0.5) - linalg.norm(cgrad_this)
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            lhs_value = miu_f_ * Sf_[self.axis] * (1 - eCf_[self.axis]**2) / dperp_
            rhs_value = (miu_f_ * Sf_[self.axis] / dperp_) * ((funit_this * (1 - eCf_[self.axis]**2)) + \
                        ((cunit_axes0 - funit_axes0) * eCf_[self.axis] * eCf_[axes[0]]) + \
                        ((cunit_axes1 - funit_axes1) * eCf_[self.axis] * eCf_[axes[1]])) - \
                        value_.faces("unit", "P")[col][1] * Sf_[self.axis]
            self.set_lhs([row, row], lhs_value, add = True)
            area_ = np.sum(np.array([mesh_.faces[it1].area for it1 in mesh_.cells[row].faces]))
            Re_ = rho_C_ * np.sqrt(np.sum(np.array([map(lambda x: x**2, v_)]))) * mesh_.cells[row].volume / (area_ * miu_C_)
            tau_ = v_[self.axis] * 8 * rho_C_ / Re_
            rhs_value = -1 * tau_ / (rho_C_ * 2 * mesh_.cells[row].volume / mesh_.faces[col].area)
            self.set_rhs([row, 0], rhs_value, add = True)
        elif "inlet" in mesh_.faces[col].boundary:
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vin_v0_ = value_.cells("grad", coor_dict[self.axis])[row][1] - \
                           (np.dot(value_.cells("grad", coor_dict[self.axis])[row][1], eCf_) * eCf_)
            grad_vin_v1_ = value_.cells("grad", coor_dict[axes[0]])[row][1] - \
                           (np.dot(value_.cells("grad", coor_dict[self.axis])[row][1], eCf_) * eCf_)
            grad_vin_v2_ = value_.cells("grad", coor_dict[axes[1]])[row][1] - \
                           (np.dot(value_.cells("grad", coor_dict[self.axis])[row][1], eCf_) * eCf_)
            vin_v0_ = value_.cells("unit", coor_dict[self.axis])[row][1] + np.dot(grad_vin_v0_, dCf_)
            vin_v1_ = value_.cells("unit", coor_dict[axes[0]])[row][1] + np.dot(grad_vin_v1_, dCf_)
            vin_v2_ = value_.cells("unit", coor_dict[axes[1]])[row][1] + np.dot(grad_vin_v2_, dCf_)
            vin_ = np.array([0, 0, 0])
            vin_[self.axis] = vin_v0_; vin_[axes[0]] = vin_v1_; vin_[axes[1]] = vin_v2_
            lhs_value = value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vin_, Sf_)
            rhs_value = -1 * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vin_, Sf_) * \
                                      np.dot(grad_vin_v0_, dCf_) + user_.constants.loc[0, "Pamb"] * Sf_[self.axis])
            self.set_lhs([row, row], lhs_value, add = True)
            self.set_rhs([row, 0], rhs_value, add = True)
        elif "outlet" in mesh_.faces[col].boundary:
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vout_v0_ = value_.cells("grad", coor_dict[self.axis])[row][1] - \
                           (np.dot(value_.cells("grad", coor_dict[self.axis])[row][1], eCf_) * eCf_)
            grad_vout_v1_ = value_.cells("grad", coor_dict[axes[0]])[row][1] - \
                           (np.dot(value_.cells("grad", coor_dict[self.axis])[row][1], eCf_) * eCf_)
            grad_vout_v2_ = value_.cells("grad", coor_dict[axes[1]])[row][1] - \
                           (np.dot(value_.cells("grad", coor_dict[self.axis])[row][1], eCf_) * eCf_)
            vout_v0_ = value_.cells("unit", coor_dict[self.axis])[row][1] + np.dot(grad_vout_v0_, dCf_)
            vout_v1_ = value_.cells("unit", coor_dict[axes[0]])[row][1] + np.dot(grad_vout_v1_, dCf_)
            vout_v2_ = value_.cells("unit", coor_dict[axes[1]])[row][1] + np.dot(grad_vout_v2_, dCf_)
            vout_ = np.array([0, 0, 0])
            vout_[self.axis] = vout_v0_; vout_[axes[0]] = vout_v1_; vout_[axes[1]] = vout_v2_
            pout_ = value_.faces("unit", "P")[col][1]
            lhs_value = value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_)
            rhs_value = -1 * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_) * \
                                      np.dot(grad_vout_v0_, dCf_) + pout_ * Sf_[self.axis])
            self.set_lhs([row, row], lhs_value, add = True)
            self.set_rhs([row, 0], rhs_value, add = True)
        else:
            pass
    def calc_post_bound(self, mesh_ : mesh, user_ : user, value_ : value):
        coor_dict = dict({0: "u", 1 :"v", 2: "w"})
        axes = np.delete(np.array([0,1,2]), self.axis)
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            if "noslip" in mesh_.faces[it2].boundary:
                value_.set_faces("unit", coor_dict[self.axis], it2, 0)
                value_.set_faces("grad", coor_dict[self.axis], it2, np.array([0, 0, 0]))
            elif "inlet" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_vin_v0_ = value_.cells("grad", coor_dict[self.axis])[it1][1] - \
                            (np.dot(value_.cells("grad", coor_dict[self.axis])[it1][1], eCf_) * eCf_)
                grad_vin_v1_ = value_.cells("grad", coor_dict[axes[0]])[it1][1] - \
                            (np.dot(value_.cells("grad", coor_dict[self.axis])[it1][1], eCf_) * eCf_)
                grad_vin_v2_ = value_.cells("grad", coor_dict[axes[1]])[it1][1] - \
                            (np.dot(value_.cells("grad", coor_dict[self.axis])[it1][1], eCf_) * eCf_)
                vin_v0_ = value_.cells("unit", coor_dict[self.axis])[it1][1] + np.dot(grad_vin_v0_, dCf_)
                vin_v1_ = value_.cells("unit", coor_dict[axes[0]])[it1][1] + np.dot(grad_vin_v1_, dCf_)
                vin_v2_ = value_.cells("unit", coor_dict[axes[1]])[it1][1] + np.dot(grad_vin_v2_, dCf_)
                value_.set_faces("unit", coor_dict[self.axis], it2, vin_v0_); value_.set_faces("grad", coor_dict[self.axis], it2, grad_vin_v0_)
                value_.set_faces("unit", coor_dict[axes[0]], it2, vin_v1_); value_.set_faces("grad", coor_dict[axes[0]], it2, grad_vin_v1_)
                value_.set_faces("unit", coor_dict[axes[1]], it2, vin_v2_); value_.set_faces("grad", coor_dict[axes[1]], it2, grad_vin_v2_)
            elif "outlet" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_vout_v0_ = value_.cells("grad", coor_dict[self.axis])[it1][1] - \
                            (np.dot(value_.cells("grad", coor_dict[self.axis])[it1][1], eCf_) * eCf_)
                grad_vout_v1_ = value_.cells("grad", coor_dict[axes[0]])[it1][1] - \
                            (np.dot(value_.cells("grad", coor_dict[self.axis])[it1][1], eCf_) * eCf_)
                grad_vout_v2_ = value_.cells("grad", coor_dict[axes[1]])[it1][1] - \
                            (np.dot(value_.cells("grad", coor_dict[self.axis])[it1][1], eCf_) * eCf_)
                vout_v0_ = value_.cells("unit", coor_dict[self.axis])[it1][1] + np.dot(grad_vout_v0_, dCf_)
                vout_v1_ = value_.cells("unit", coor_dict[axes[0]])[it1][1] + np.dot(grad_vout_v1_, dCf_)
                vout_v2_ = value_.cells("unit", coor_dict[axes[1]])[it1][1] + np.dot(grad_vout_v2_, dCf_)
                value_.set_faces("unit", coor_dict[self.axis], it2, vout_v0_); value_.set_faces("grad", coor_dict[self.axis], it2, grad_vout_v0_)
                value_.set_faces("unit", coor_dict[axes[0]], it2, vout_v1_); value_.set_faces("grad", coor_dict[axes[0]], it2, grad_vout_v1_)
                value_.set_faces("unit", coor_dict[axes[1]], it2, vout_v2_); value_.set_faces("grad", coor_dict[axes[1]], it2, grad_vout_v2_)
    def calc_wall(self, mesh_ : mesh, user_ : user, value_ : value):
        coor_dict = dict({0: "u", 1 : "v", 2: "w"})
        if "conj" in list(mesh_.templates.cc.keys()):
            conj_ = mesh_.templates.iter("cc", "conj")
            for it1, it2, it3 in zip(conj_[0], conj_[1], conj_[2]):
                if "fluid" in mesh_.cells[it1].domain[0]:
                    v_ = np.array([0, 0, 0])
                    v_[0] = value_.cells("unit", "u")[it1][1]
                    v_[1] = value_.cells("unit", "v")[it1][1]
                    v_[2] = value_.cells("unit", "w")[it1][1]
                    v_val_ = np.sqrt(np.sum(np.array([map(lambda x: x**2, v_)])))
                    Sf_wall_ = -1 * mesh_.geoms.Sf(False, it1, it3)
                    v_parallel_ = np.cross(np.cross(v_, Sf_wall_), Sf_wall_)
                    v_parallel_val_ = np.sqrt(np.sum(np.array([map(lambda x: x**2), v_parallel_])))
                    check = np.dot(v_val_, v_parallel_)
                    if check < 0:
                        v_parallel_ = -1 * v_parallel_
                    theta = math.acos(np.dot(v_, v_parallel_) / (v_val_ * v_parallel_val_))
                    v_val_ = v_val_ * math.sin(theta)
                    Ret_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1) * pow(value_.cells("unit", "k")[it1][1], 2) / \
                        (value_.calc_prop(mesh_, user_, "miu", "cells", it1) * value_.cells("unit", "e")[it1][1])
                    cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_/50, 2))
                    gradCfluid_ = value_.cells("grad", coor_dict[self.axis])[it1][1]
                    dperp_ = (np.sqrt(2 * value_.cells("unit", coor_dict[self.axis]) - 1) * \
                            np.sqrt(np.sum(np.array([map(lambda x:x**2, gradCfluid_)]))))
                    dCplus_ = dperp_ * pow(cmiu_, 0.25) * np.sqrt(value_.cells("unit", "k")[it1][1]) * \
                            value_.calc_prop(mesh_, user_, "rho", "cells", it1) / value_.calc_prop(mesh_, user_, "miu", "cells", it1)
                    miutau_ = v_val_ * 0.41 / (np.log(dCplus_) + 5.25)
                    dplusv_ = dperp_ * miutau_ * value_.calc_prop(mesh_, user_, "rho", "cells", it1) / value_.calc_prop(mesh_, user_, "miu", "cells", it1)
                    vplus_ = np.log(dplusv_) / 0.41 + 5.25
                    value_.set_cells("unit", coor_dict[self.axis], it1, vplus_)   
        else:
            pass
    def iter_solve(self, mesh_ : mesh, user_ : user, value_ : value, under_relax_, tol_, max_iter_, time_step_, current_time_):
        coor_dict = dict({0: 'u', 1: 'v', 2: 'w'})
        self.calc_coef(mesh_, user_, value_)
        lhs_transient_, rhs_transient_ = super().calc_transient(mesh_, user_, value_, coor_dict[self.axis], time_step_, current_time_)
        under_relax_b = deepcopy(rhs_transient_)
        for it1 in range(0, lhs_transient_.shape[0]):
            lhs_transient_[it1, it1] = lhs_transient_[it1, it1] / under_relax_
            under_relax_[it1, 0] = lhs_transient_[it1, it1] * value_.cells("unit", coor_dict[self.axis])[it1][1] * (1 - under_relax_)
        A = lambda x: sparse.linalg.spsolve(lhs_transient_, x) 
        b = rhs_transient_ + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = tol_, maxiter = max_iter_)
        return x
class turb_k(linear):
    def __init__(self, mesh_ : mesh):
        super().__init__(mesh_, what = ["fluid"])
    
    def calc_coef(self, mesh_ : mesh, user_ : user, value_ : value):
        # fluid only
        aC_ = 0
        bC_ = 0
        cc_ = mesh_.templates.iter("cc", "fluid"); cc_ = sorted(zip(cc_[0], cc_[1], cc_[2]), key = lambda x: x[0])
        prev_row = cc_[0][0]
        for it1, it2, it3 in cc_:
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", it3)
            miu_f_ = value_.calc_prop(mesh_, user_, "miu", "faces", it3)
            Sf_ = mesh_.geoms.Sf(False, it1, it3)
            dCf_ = mesh_.geoms.dCf(False, it1, it3)
            eCF_ = mesh_.geoms.eCF(False, it1, it2)
            dCF_ = mesh_.geoms.dCF(True, it1, it2)
            v_ = np.array([value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]])
            Ret_ = rho_f_ * pow(value_.faces("unit", "k")[it3][1], 2) / (miu_f_ * value_.faces("unit", "e")[it3][1])
            cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + float(Ret_) / 50, 2))
            St_tensor_ = np.array([[value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]]])
            St_tensor_ = (St_tensor_ + np.transpose(St_tensor_))
            St_ = np.sqrt(np.dot(St_tensor_, St_tensor_))
            try:
                with np.errstate(divide='raise'):
                    ts_ = np.min(np.array([value_.faces("unit", "k")[it3][1] / value_.faces("unit", "e")[it3][1], value_.calc_prop(mesh_, user_, "alpha", "faces", it3) / (pow(6, 0.5) * cmiu_ * St_)]))
                    miut_ = rho_f_ * cmiu_ * value_.faces("unit", "k")[it3][1] * ts_
            except FloatingPointError:
                miut_ = 0
            graditr_ = value_.linear_itr("grad", mesh_, [it1, it2, it3], "k")
            if prev_row == it1:
                aC_ += (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                aC_ += (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                print(np.dot(graditr_, eCF_)); print(np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", "k")[it1][1] + graditr_), dCf_))
                print(np.dot(np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", "k")[it1][1] + graditr_), dCf_) / 2, dCf_)); print(np.dot(v_, Sf_))
                bC_ += np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", "k")[it1][1] + graditr_) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * (miu_f_ + miut_)
                lhs_value = (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                self.set_lhs([it1, it2], lhs_value); self.set_lhs([it1, it2], lhs_value_diff, add = True)
                prev_row = it1
            else:
                gradC_u_ = value_.cells("grad", "u")[prev_row][1]
                gradC_v_ = value_.cells("grad", "v")[prev_row][1]
                gradC_w_ = value_.cells("grad", "w")[prev_row][1]
                phi_v_ = 2 * (gradC_u_[0]**2 + gradC_v_[1]**2 + gradC_w_[2]**2) + pow(gradC_u_[1] + gradC_v_[0], 2) \
                         + pow(gradC_u_[2] + gradC_w_[0], 2) + pow(gradC_v_[2] + gradC_w_[1], 2)
                Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * pow(value_.cells("unit", "k")[prev_row][1], 2) / \
                         (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([value_.cells("grad", "u")[prev_row][1], value_.cells("grad", "v")[prev_row][1], value_.cells("grad", "w")[prev_row][1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(np.dot(St_tensor_C_, St_tensor_C_))
                try:
                    with np.errstate(divide='raise'):
                        ts_C_ = np.min(np.array([value_.cells("unit", "k")[prev_row][1] / value_.cells("unit", "e")[prev_row][1], \
                                value_.calc_prop(mesh_, user_, "alpha", "cells", prev_row) / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                        miut_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * cmiu_C_ * value_.cells("unit", "k")[prev_row][1] * ts_C_
                except FloatingPointError:
                    miut_C_ = 0
                bC_ += (miut_C_ * phi_v_ - value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1]) \
                       * mesh_.cells[prev_row].volume
                print(aC_, bC_)
                self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                aC_ = 0; bC_ = 0
                aC_ += (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                aC_ += (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                bC_ += np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", "k")[it1][1] + graditr_) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * (miu_f_ + miut_)
                lhs_value = (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_)
                self.set_lhs([it1, it2], lhs_value); self.set_lhs([it1, it2], lhs_value_diff, add = True)
                prev_row = it1
        gradC_u_ = value_.cells("grad", "u")[prev_row][1]
        gradC_v_ = value_.cells("grad", "v")[prev_row][1]
        gradC_w_ = value_.cells("grad", "w")[prev_row][1]
        phi_v_ = 2 * (gradC_u_[0]**2 + gradC_v_[1]**2 + gradC_w_[2]**2) + pow(gradC_u_[1] + gradC_v_[0], 2) \
                    + pow(gradC_u_[2] + gradC_w_[0], 2) + pow(gradC_v_[2] + gradC_w_[1], 2)
        Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * pow(value_.cells("unit", "k")[prev_row][1], 2) / \
                    (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([value_.cells("grad", "u")[prev_row][1], value_.cells("grad", "v")[prev_row][1], value_.cells("grad", "w")[prev_row][1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(np.dot(St_tensor_C_, St_tensor_C_))
        try:
            with np.errstate(divide='raise'):
                ts_C_ = np.min(np.array([value_.cells("unit", "k")[prev_row][1] / value_.cells("unit", "e")[prev_row][1], \
                        value_.calc_prop(mesh_, user_, "alpha", "cells", prev_row) / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * cmiu_C_ * value_.cells("unit", "k")[prev_row][1] * ts_C_
        except FloatingPointError:
            miut_C_ = 0
        bC_ += (miut_C_ * phi_v_ - value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1]) \
                * mesh_.cells[prev_row].volume
        print(aC_, bC_)
        self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            v_ = np.array([value_.faces("unit", "u")[it2][1], value_.faces("unit", "v")[it2][1], value_.faces("unit", "w")[it2][1]])
            self.calc_bound(mesh_, user_, value_, v_, it1, it2)
    def calc_bound(self, mesh_ : mesh, user_ : user, value_ : value, v_, row : int, col : int):
        if "inlet" in mesh_.faces[col].boundary:
            # specified value; zero gradient at inlet
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vin_v0_ = value_.cells("grad", "u")[row][1] - \
                           (np.dot(value_.cells("grad", "u")[row][1], eCf_) * eCf_)
            grad_vin_v1_ = value_.cells("grad", "v")[row][1] - \
                           (np.dot(value_.cells("grad", "v")[row][1], eCf_) * eCf_)
            grad_vin_v2_ = value_.cells("grad", "w")[row][1] - \
                           (np.dot(value_.cells("grad", "w")[row][1], eCf_) * eCf_)
            vin_v0_ = value_.cells("unit", "u")[row][1] + np.dot(grad_vin_v0_, dCf_)
            vin_v1_ = value_.cells("unit", "v")[row][1] + np.dot(grad_vin_v1_, dCf_)
            vin_v2_ = value_.cells("unit", "w")[row][1] + np.dot(grad_vin_v2_, dCf_)
            vin_ = np.array([vin_v0_, vin_v1_, vin_v2_])
            grad_kin_ = value_.cells("grad", "k")[row][1] - \
                        (np.dot(value_.cells("grad", "k")[row][1], eCf_) * eCf_)
            kin_ = 0.5 * np.dot(vin_, vin_) * 0.01**2      
            rhs_value = -1 * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vin_, Sf_)) * (kin_ + np.dot(grad_kin_, dCf_))
            self.set_rhs([row, 0], rhs_value, add = True)
        elif "outlet" in mesh_.faces[col].boundary:
            # fully developed flow; zero gradient at outlet
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vout_v0_ = value_.cells("grad", "u")[row][1] - \
                           (np.dot(value_.cells("grad", "u")[row][1], eCf_) * eCf_)
            grad_vout_v1_ = value_.cells("grad", "v")[row][1] - \
                           (np.dot(value_.cells("grad", "v")[row][1], eCf_) * eCf_)
            grad_vout_v2_ = value_.cells("grad", "w")[row][1] - \
                           (np.dot(value_.cells("grad", "w")[row][1], eCf_) * eCf_)
            vout_v0_ = value_.cells("unit", "u")[row][1] + np.dot(grad_vout_v0_, dCf_)
            vout_v1_ = value_.cells("unit", "v")[row][1] + np.dot(grad_vout_v1_, dCf_)
            vout_v2_ = value_.cells("unit", "w")[row][1] + np.dot(grad_vout_v2_, dCf_)
            vout_ = np.array([vout_v0_, vout_v1_, vout_v2_])
            grad_kout_ = value_.cells("grad", "k")[row][1] - \
                         (np.dot(value_.cells("grad", "k")[row][1], eCf_) * eCf_)
            lhs_value = value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_)
            rhs_value = -1 * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_) * \
                                      np.dot(grad_kout_, dCf_))
            self.set_lhs([row, row], lhs_value, add = True)
            self.set_rhs([row, 0], rhs_value, add = True)
        else:
            pass
    def calc_post_bound(self, mesh_ : mesh, user_ : user, value_ : value):
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            if "inlet" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_vin_v0_ = value_.cells("grad", "u")[it1][1] - \
                            (np.dot(value_.cells("grad", "u")[it1][1], eCf_) * eCf_)
                grad_vin_v1_ = value_.cells("grad", "v")[it1][1] - \
                            (np.dot(value_.cells("grad", "v")[it1][1], eCf_) * eCf_)
                grad_vin_v2_ = value_.cells("grad", "w")[it1][1] - \
                            (np.dot(value_.cells("grad", "w")[it1][1], eCf_) * eCf_)
                vin_v0_ = value_.cells("unit", "u")[it1][1] + np.dot(grad_vin_v0_, dCf_)
                vin_v1_ = value_.cells("unit", "v")[it1][1] + np.dot(grad_vin_v1_, dCf_)
                vin_v2_ = value_.cells("unit", "w")[it1][1] + np.dot(grad_vin_v2_, dCf_)
                vin_ = np.array([vin_v0_, vin_v1_, vin_v2_])
                grad_kin_ = value_.cells("grad", "k")[it1][1] - \
                            (np.dot(value_.cells("grad", "k")[it1][1], eCf_) * eCf_)
                kin_ = 0.5 * np.dot(vin_, vin_) * 0.01**2
                value_.set_faces("unit", "k", it2, kin_); value_.set_faces("grad", "k", it2, grad_kin_)
            elif "outlet" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_vout_v0_ = value_.cells("grad", "u")[it1][1] - \
                            (np.dot(value_.cells("grad", "u")[it1][1], eCf_) * eCf_)
                grad_vout_v1_ = value_.cells("grad", "v")[it1][1] - \
                            (np.dot(value_.cells("grad", "v")[it1][1], eCf_) * eCf_)
                grad_vout_v2_ = value_.cells("grad", "w")[it1][1] - \
                            (np.dot(value_.cells("grad", "w")[it1][1], eCf_) * eCf_)
                vout_v0_ = value_.cells("unit", "u")[it1][1] + np.dot(grad_vout_v0_, dCf_)
                vout_v1_ = value_.cells("unit", "v")[it1][1] + np.dot(grad_vout_v1_, dCf_)
                vout_v2_ = value_.cells("unit", "w")[it1][1] + np.dot(grad_vout_v2_, dCf_)
                vout_ = np.array([vout_v0_, vout_v1_, vout_v2_])
                grad_kout_ = value_.cells("grad", "k")[it1][1] - \
                            (np.dot(value_.cells("grad", "k")[it1][1], eCf_) * eCf_)
                kout_ = value_.cells("unit", "k")[it1][1] + np.dot(grad_kout_, dCf_)
                value_.set_faces("unit", "k", it2, kout_); value_.set_faces("grad", "k", it2, grad_kout_)
    def calc_wall(self, mesh_ : mesh, user_ : user, value_ : value):
        if "conj" in list(mesh_.templates.cc.keys()):
            conj_ = mesh_.templates.iter("cc", "conj")
            for it1, it2, it3 in zip(conj_[0], conj_[1], conj_[2]):
                if "fluid" in mesh_.cells[it1].domain[0]:
                    Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1) * pow(value_.cells("unit", "k")[it1][1], 2) / \
                            (value_.calc_prop(mesh_, user_, "miu", "cells", it1) * value_.cells("unit", "e")[it1][1])
                    cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                    value_.set_cells("unit", "k", it1, 1 / np.sqrt(cmiu_C_))
        else:
            pass
    def iter_solve(self, mesh_ : mesh, user_ : user, value_ : value, under_relax_, tol_, max_iter_, time_step_, current_time_):
        self.calc_coef(mesh_, user_, value_)
        lhs_transient_, rhs_transient_ = super().calc_transient(mesh_, user_, value_, "k", time_step_, current_time_)
        under_relax_b = deepcopy(rhs_transient_)
        for it1 in range(0, lhs_transient_.shape[0]):
            lhs_transient_[it1, it1] = lhs_transient_[it1, it1] / under_relax_
        for it1 in range(0, rhs_transient_.shape[0]):
            under_relax_b[it1, 0] = lhs_transient_[it1, it1] * float(value_.cells("unit", "k")[it1][1]) * (1 - under_relax_)
        A = lhs_transient_.tocsr()
        b = rhs_transient_ + under_relax_b; b = b.tocsr()
        M_x = lambda x: sparse.linalg.spsolve(A, x)
        M_ = sparse.linalg.LinearOperator(A.shape, M_x)
        x, exitCode = sparse.linalg.gmres(A, b.toarray(), tol = tol_, maxiter = max_iter_, M = M_)
        print(A); print(b); print(x)
        return x
class turb_e(linear):
    def __init__(self, mesh_ : mesh):
        super().__init__(mesh_, what = ["fluid"])
    
    def calc_coef(self, mesh_ : mesh, user_ : user, value_ : value):
        # fluid only
        aC_ = 0
        bC_ = 0
        cc_ = mesh_.templates.iter("cc", "fluid"); cc_ = sorted(zip(cc_[0], cc_[1], cc_[2]), key = lambda x: x[0])
        prev_row = cc_[0][0]
        for it1, it2, it3 in cc_:
            rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", it3)
            miu_f_ = value_.calc_prop(mesh_, user_, "miu", "faces", it3)
            Sf_ = mesh_.geoms.Sf(False, it1, it3)
            dCf_ = mesh_.geoms.dCf(False, it1, it3)
            eCF_ = mesh_.geoms.eCF(False, it1, it2)
            dCF_ = mesh_.geoms.dCF(True, it1, it2)
            v_ = np.array([value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]])
            Ret_ = rho_f_ * pow(value_.faces("unit", "k")[it3][1], 2) / (miu_f_ * value_.faces("unit", "e")[it3][1])
            cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_ / 50, 2))
            St_tensor_ = np.array([[value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]]])
            St_tensor_ = (St_tensor_ + np.transpose(St_tensor_))
            St_ = np.sqrt(np.dot(St_tensor_, St_tensor_))
            try:
                with np.errstate(divide='raise'):
                    ts_ = np.min(np.array([value_.faces("unit", "k")[it3][1] / value_.faces("unit", "e")[it3][1], value_.calc_prop(mesh_, user_, "alpha", "faces", it3) / (pow(6, 0.5) * cmiu_ * St_)]))
                    miut_ = rho_f_ * cmiu_ * value_.faces("unit", "k")[it3][1] * ts_
            except FloatingPointError:
                miut_ = 0
            graditr_ = value_.linear_itr("grad", mesh_, [it1, it2, it3], "e")
            if prev_row == it1:
                aC_ += (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                aC_ += (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_ / 1.3)
                bC_ += np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", "e")[it1][1] + graditr_) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * (miu_f_ + miut_ / 1.3)
                lhs_value = (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_ / 1.3)
                self.set_lhs([it1, it2], lhs_value); self.set_lhs([it1, it2], lhs_value_diff, add = True)
                prev_row = it1
            else:
                gradC_u_ = value_.cells("grad", "u")[prev_row][1]
                gradC_v_ = value_.cells("grad", "v")[prev_row][1]
                gradC_w_ = value_.cells("grad", "w")[prev_row][1]
                phi_v_ = 2 * (gradC_u_[0]**2 + gradC_v_[1]**2 + gradC_w_[2]**2) + pow(gradC_u_[1] + gradC_v_[0], 2) \
                         + pow(gradC_u_[2] + gradC_w_[0], 2) + pow(gradC_v_[2] + gradC_w_[1], 2)
                Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * pow(value_.cells("unit", "k")[prev_row][1], 2) / \
                         (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([value_.cells("grad", "u")[prev_row][1], value_.cells("grad", "v")[prev_row][1], value_.cells("grad", "w")[prev_row][1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(np.dot(St_tensor_C_, St_tensor_C_))
                try:
                    with np.errstate(divide='raise'):
                        ts_C_ = np.min(np.array([value_.cells("unit", "k")[prev_row][1] / value_.cells("unit", "e")[prev_row][1], \
                                value_.calc_prop(mesh_, user_, "alpha", "cells", prev_row) / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                        miut_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * cmiu_C_ * value_.cells("unit", "k")[prev_row][1] * ts_C_
                        ceps2_ = 1.92 * (1 - 0.3 * math.exp(-1 * Ret_C_**2))
                        bC_ += (1.44 * miut_C_ * phi_v_ / ts_C_ - ceps2_ * value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1] / ts_C_) \
                            * mesh_.cells[prev_row].volume
                except FloatingPointError:
                    pass
                self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                aC_ = 0; bC_ = 0
                aC_ += (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                aC_ += (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_ / 1.3)
                bC_ += np.dot(np.dot(graditr_, eCF_) * eCF_ - (value_.cells("grad", "e")[it1][1] + graditr_) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * (miu_f_ + miut_ / 1.3)
                lhs_value = (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * (miu_f_ + miut_ / 1.3)
                self.set_lhs([it1, it2], lhs_value); self.set_lhs([it1, it2], lhs_value_diff, add = True)
                prev_row = it1
        gradC_u_ = value_.cells("grad", "u")[prev_row][1]
        gradC_v_ = value_.cells("grad", "v")[prev_row][1]
        gradC_w_ = value_.cells("grad", "w")[prev_row][1]
        phi_v_ = 2 * (gradC_u_[0]**2 + gradC_v_[1]**2 + gradC_w_[2]**2) + pow(gradC_u_[1] + gradC_v_[0], 2) \
                    + pow(gradC_u_[2] + gradC_w_[0], 2) + pow(gradC_v_[2] + gradC_w_[1], 2)
        Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * pow(value_.cells("unit", "k")[prev_row][1], 2) / \
                    (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([value_.cells("grad", "u")[prev_row][1], value_.cells("grad", "v")[prev_row][1], value_.cells("grad", "w")[prev_row][1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(np.dot(St_tensor_C_, St_tensor_C_))
        try:
            with np.errstate(divide='raise'):
                ts_C_ = np.min(np.array([value_.cells("unit", "k")[prev_row][1] / value_.cells("unit", "e")[prev_row][1], \
                        value_.calc_prop(mesh_, user_, "alpha", "cells", prev_row) / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * cmiu_C_ * value_.cells("unit", "k")[prev_row][1] * ts_C_
                ceps2_ = 1.92 * (1 - 0.3 * math.exp(-1 * Ret_C_**2))
                bC_ += (1.44 * miut_C_ * phi_v_ / ts_C_ - ceps2_ * value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1] / ts_C_) \
                        * mesh_.cells[prev_row].volume
        except FloatingPointError:
            pass
        self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            v_ = np.array([value_.faces("unit", "u")[it2][1], value_.faces("unit", "v")[it2][1], value_.faces("unit", "w")[it2][1]])
            self.calc_bound(mesh_, user_, value_, v_, it1, it2)
    def calc_bound(self, mesh_ : mesh, user_ : user, value_ : value, v_, row : int, col : int):
        if "inlet" in mesh_.faces[col].boundary:
            # specified value; zero gradient at inlet
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vin_v0_ = value_.cells("grad", "u")[row][1] - \
                           (np.dot(value_.cells("grad", "u")[row][1], eCf_) * eCf_)
            grad_vin_v1_ = value_.cells("grad", "v")[row][1] - \
                           (np.dot(value_.cells("grad", "v")[row][1], eCf_) * eCf_)
            grad_vin_v2_ = value_.cells("grad", "w")[row][1] - \
                           (np.dot(value_.cells("grad", "w")[row][1], eCf_) * eCf_)
            vin_v0_ = value_.cells("unit", "u")[row][1] + np.dot(grad_vin_v0_, dCf_)
            vin_v1_ = value_.cells("unit", "v")[row][1] + np.dot(grad_vin_v1_, dCf_)
            vin_v2_ = value_.cells("unit", "w")[row][1] + np.dot(grad_vin_v2_, dCf_)
            vin_ = np.array([vin_v0_, vin_v1_, vin_v2_])
            grad_ein_ = value_.cells("grad", "e")[row][1] - \
                        (np.dot(value_.cells("grad", "e")[row][1], eCf_) * eCf_)
            ein_ =  pow(0.5 * np.dot(vin_, vin_) * 0.01**2, 1.5) * 0.09 / (0.1 * mesh_.cells[row].volume)     
            rhs_value = -1 * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vin_, Sf_)) * (ein_ + np.dot(grad_ein_, dCf_))
            self.set_rhs([row, 0], rhs_value, add = True)
        elif "outlet" in mesh_.faces[col].boundary:
            # fully developed flow; zero gradient at outlet
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vout_v0_ = value_.cells("grad", "u")[row][1] - \
                           (np.dot(value_.cells("grad", "u")[row][1], eCf_) * eCf_)
            grad_vout_v1_ = value_.cells("grad", "v")[row][1] - \
                           (np.dot(value_.cells("grad", "v")[row][1], eCf_) * eCf_)
            grad_vout_v2_ = value_.cells("grad", "w")[row][1] - \
                           (np.dot(value_.cells("grad", "w")[row][1], eCf_) * eCf_)
            vout_v0_ = value_.cells("unit", "u")[row][1] + np.dot(grad_vout_v0_, dCf_)
            vout_v1_ = value_.cells("unit", "v")[row][1] + np.dot(grad_vout_v1_, dCf_)
            vout_v2_ = value_.cells("unit", "w")[row][1] + np.dot(grad_vout_v2_, dCf_)
            vout_ = np.array([vout_v0_, vout_v1_, vout_v2_])
            grad_eout_ = value_.cells("grad", "e")[row][1] - \
                         (np.dot(value_.cells("grad", "e")[row][1], eCf_) * eCf_)
            lhs_value = value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_)
            rhs_value = -1 * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_) * \
                                      np.dot(grad_eout_, dCf_))
            self.set_lhs([row, row], lhs_value, add = True)
            self.set_rhs([row, 0], rhs_value, add = True)
        else:
            pass
    def calc_post_bound(self, mesh_ : mesh, user_ : user, value_ : value):
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            if "inlet" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_vin_v0_ = value_.cells("grad", "u")[it1][1] - \
                            (np.dot(value_.cells("grad", "u")[it1][1], eCf_) * eCf_)
                grad_vin_v1_ = value_.cells("grad", "v")[it1][1] - \
                            (np.dot(value_.cells("grad", "v")[it1][1], eCf_) * eCf_)
                grad_vin_v2_ = value_.cells("grad", "w")[it1][1] - \
                            (np.dot(value_.cells("grad", "w")[it1][1], eCf_) * eCf_)
                vin_v0_ = value_.cells("unit", "u")[it1][1] + np.dot(grad_vin_v0_, dCf_)
                vin_v1_ = value_.cells("unit", "v")[it1][1] + np.dot(grad_vin_v1_, dCf_)
                vin_v2_ = value_.cells("unit", "w")[it1][1] + np.dot(grad_vin_v2_, dCf_)
                vin_ = np.array([vin_v0_, vin_v1_, vin_v2_])
                grad_ein_ = value_.cells("grad", "e")[it1][1] - \
                            (np.dot(value_.cells("grad", "e")[it1][1], eCf_) * eCf_)
                ein_ =  pow(0.5 * np.dot(vin_, vin_) * 0.01**2, 1.5) * 0.09 / (0.1 * mesh_.cells[it1].volume)  
                value_.set_faces("unit", "e", it2, ein_); value_.set_faces("grad", "e", it2, grad_ein_)
            elif "outlet" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_vout_v0_ = value_.cells("grad", "u")[it1][1] - \
                            (np.dot(value_.cells("grad", "u")[it1][1], eCf_) * eCf_)
                grad_vout_v1_ = value_.cells("grad", "v")[it1][1] - \
                            (np.dot(value_.cells("grad", "v")[it1][1], eCf_) * eCf_)
                grad_vout_v2_ = value_.cells("grad", "w")[it1][1] - \
                            (np.dot(value_.cells("grad", "w")[it1][1], eCf_) * eCf_)
                vout_v0_ = value_.cells("unit", "u")[it1][1] + np.dot(grad_vout_v0_, dCf_)
                vout_v1_ = value_.cells("unit", "v")[it1][1] + np.dot(grad_vout_v1_, dCf_)
                vout_v2_ = value_.cells("unit", "w")[it1][1] + np.dot(grad_vout_v2_, dCf_)
                vout_ = np.array([vout_v0_, vout_v1_, vout_v2_])
                grad_eout_ = value_.cells("grad", "e")[it1][1] - \
                         (np.dot(value_.cells("grad", "e")[it1][1], eCf_) * eCf_)
                eout_ = value_.cells("unit", "e")[it1][1] + np.dot(grad_eout_, dCf_)
                value_.set_faces("unit", "e", it2, eout_); value_.set_faces("grad", "e", it2, grad_eout_)
    def calc_wall(self, mesh_ : mesh, user_ : user, value_ : value):
        if "conj" in list(mesh_.templates.cc.keys()):
            conj_ = mesh_.templates.iter("cc", "conj")
            for it1, it2, it3 in zip(conj_[0], conj_[1], conj_[2]):
                if "fluid" in mesh_.cells[it1].domain[0]:
                    v_ = np.array([0, 0, 0])
                    v_[0] = value_.cells("unit", "u")[it1][1]
                    v_[1] = value_.cells("unit", "v")[it1][1]
                    v_[2] = value_.cells("unit", "w")[it1][1]
                    v_val_ = np.sqrt(np.sum(np.array([map(lambda x: x**2, v_)])))
                    Ret_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1) * pow(value_.cells("unit", "k")[it1][1], 2) / \
                        (value_.calc_prop(mesh_, user_, "miu", "cells", it1) * value_.cells("unit", "e")[it1][1])
                    cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_/50, 2))
                    gradCfluid_ = value_.cells("grad", "e")[it1][1]
                    dperp_ = (np.sqrt(2 * value_.cells("unit", "e") - 1) * \
                            np.sqrt(np.sum(np.array([map(lambda x:x**2, gradCfluid_)]))))
                    dCplus_ = dperp_ * pow(cmiu_, 0.25) * np.sqrt(value_.cells("unit", "k")[it1][1]) * \
                            value_.calc_prop(mesh_, user_, "rho", "cells", it1) / value_.calc_prop(mesh_, user_, "miu", "cells", it1)
                    miutau_ = v_val_ * 0.41 / (np.log(dCplus_) + 5.25)
                    eplus_ = value_.calc_prop(mesh_, user_, "miu", "cells", it1) / (value_.calc_prop(mesh_, user_, "rho", it1) * miutau_ * 0.41 * dperp_)
                    value_.set_cells("unit", "e", it1, eplus_) 
        else:
            pass
    def iter_solve(self, mesh_ : mesh, user_ : user, value_ : value, under_relax_, tol_, max_iter_, time_step_, current_time_):
        self.calc_coef(mesh_, user_, value_)
        lhs_transient_, rhs_transient_ = super().calc_transient(mesh_, user_, value_, "e", time_step_, current_time_)
        under_relax_b = deepcopy(rhs_transient_)
        for it1 in range(0, lhs_transient_.shape[0]):
            lhs_transient_[it1, it1] = lhs_transient_[it1, it1] / under_relax_
        for it1 in range(0, rhs_transient_.shape[0]):
            under_relax_b[it1, 0] = lhs_transient_[it1, it1] * float(value_.cells("unit", "e")[it1][1]) * (1 - under_relax_)
        A = lhs_transient_.tocsr()
        b = rhs_transient_ + under_relax_b; b = b.tocsr()
        M_x = lambda x: sparse.linalg.spsolve(A, x)
        M_ = sparse.linalg.LinearOperator(A.shape, M_x)
        x, exitCode = sparse.linalg.gmres(A, b.toarray(), tol = tol_, maxiter = max_iter_, M = M_)
        print(A); print(b); print(x)
        return x
class energy(linear):
    def __init__(self, mesh_ : mesh):
        what_ = [it1 for it1 in list(mesh_.templates.cc.keys()) if it1 != "s2s"]
        if "conj" in what_:
            what_.remove("conj"); what_.append("conj")
        super().__init__(mesh_, what = what_)

    def calc_coef(self, mesh_ : mesh, user_ : user, value_ : value):
        for it1 in self.iter:
            prev_row = 0
            if it1 == "fluid":
                aC_ = 0
                bC_ = 0
                cc_ = mesh_.templates.iter("cc", "fluid"); cc_ = sorted(zip(cc_[0], cc_[1], cc_[2]), key = lambda x: x[0])
                prev_row = cc_[0][0]
                for it1, it2, it3 in cc_:
                    rho_f_ = value_.calc_prop(mesh_, user_, "rho", "faces", it3)
                    miu_f_ = value_.calc_prop(mesh_, user_, "miu", "faces", it3)
                    Sf_ = mesh_.geoms.Sf(False, it1, it3)
                    dCf_ = mesh_.geoms.dCf(False, it1, it3)
                    eCF_ = mesh_.geoms.eCF(False, it1, it2)
                    dCF_ = mesh_.geoms.dCF(True, it1, it2)
                    v_ = np.array([value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]])
                    Ret_ = rho_f_ * pow(value_.faces("unit", "k")[it3][1], 2) / (miu_f_ * value_.faces("unit", "e")[it3][1])
                    cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_ / 50, 2))
                    St_tensor_ = np.array([[value_.faces("unit", "u")[it3][1], value_.faces("unit", "v")[it3][1], value_.faces("unit", "w")[it3][1]]])
                    St_tensor_ = (St_tensor_ + np.transpose(St_tensor_)) * 0.5; St_tensor_T = np.transpose(St_tensor_)
                    St_ = np.sqrt(np.dot(St_tensor_[0], St_tensor_T[0] + np.dot(St_tensor_[1], St_tensor_T[1]) + np.dot(St_tensor_[2], St_tensor_T[2])))
                    try:
                        with np.errstate(divide='raise'):
                            ts_ = np.min(np.array([value_.faces("unit", "k")[it3][1] / value_.faces("unit", "e")[it3][1], value_.calc_prop(mesh_, user_, "alpha", "faces", it3) / (pow(6, 0.5) * cmiu_ * St_)]))
                            miut_ = rho_f_ * cmiu_ * value_.faces("unit", "k")[it3][1] * ts_
                    except FloatingPointError:
                        miut_ = 0
                    graditr_ = value_.linear_itr("grad", mesh_, [it1, it2, it3], "T")
                    Pr_ = miu_f_ / (rho_f_ * value_.calc_prop(mesh_, user_, "alpha", "faces", it3))
                    if prev_row == it1:
                        aC_ += value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                        aC_ += (np.dot(eCF_, Sf_) / dCF_) *  value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (miu_f_ / Pr_ + miut_ / 0.9); 
                        bC_ += value_.calc_prop(mesh_, user_, "cp", "faces", it3) * np.dot((np.dot(graditr_, eCF_) * eCF_ - value_.cells("grad", "T")[it1][1]) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                        bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (miu_f_ / Pr_ + miut_ / 0.9)
                        lhs_value = value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                        lhs_value_diff = -1 * (np.dot(eCF_, Sf_) / dCF_) * value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (miu_f_ / Pr_ + miut_ / 0.9)
                        self.set_lhs([it1, it2], lhs_value + lhs_value_diff, add = True)
                        prev_row = it1
                    else:
                        gradC_u_ = value_.cells("grad", "u")[prev_row][1]
                        gradC_v_ = value_.cells("grad", "v")[prev_row][1]
                        gradC_w_ = value_.cells("grad", "w")[prev_row][1]
                        phi_v_ = 2 * (gradC_u_[0]**2 + gradC_v_[1]**2 + gradC_w_[2]**2) + pow(gradC_u_[1] + gradC_v_[0], 2) \
                                + pow(gradC_u_[2] + gradC_w_[0], 2) + pow(gradC_v_[2] + gradC_w_[1], 2)
                        Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * pow(value_.cells("unit", "k")[prev_row][1], 2) / \
                                (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1])
                        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                        St_tensor_C_ = np.array([value_.cells("grad", "u")[prev_row][1], value_.cells("grad", "v")[prev_row][1], value_.cells("grad", "w")[prev_row][1]])
                        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5; St_tensor_C_T = np.transpose(St_tensor_C_)
                        St_C_ = np.sqrt(np.dot(St_tensor_C_[0], St_tensor_C_T[0]) + np.dot(St_tensor_C_[1], St_tensor_C_T[1]) + np.dot(St_tensor_C_[2], St_tensor_C_T[2]))
                        try:
                            with np.errstate(divide='raise'):
                                ts_C_ = np.min(np.array([value_.cells("unit", "k")[prev_row][1] / value_.cells("unit", "e")[prev_row][1], value_.calc_prop(mesh_, user_, "alpha", "cells", prev_row) / (pow(6, 0.5) * cmiu_C_ * St_C_)]))
                                miut_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * cmiu_C_ * value_.cells("unit", "k")[prev_row][1] * ts_C_
                        except FloatingPointError:
                            miut_C_ = 0
                        bC_ += (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) + miut_C_) * phi_v_ * mesh_.cells[prev_row].volume
                        self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                        aC_ += value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (1 - np.dot(eCF_, eCF_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                        aC_ += (np.dot(eCF_, Sf_) / dCF_) *  value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (miu_f_ / Pr_ + miut_ / 0.9); 
                        bC_ += value_.calc_prop(mesh_, user_, "cp", "faces", it3) * np.dot((np.dot(graditr_, eCF_) * eCF_ - value_.cells("grad", "T")[it1][1]) / 2, dCf_) * rho_f_ * np.dot(v_, Sf_)
                        bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (miu_f_ / Pr_ + miut_ / 0.9)
                        lhs_value = value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (np.dot(eCF_, dCf_) / (2 * dCF_)) * rho_f_ * np.dot(v_, Sf_)
                        lhs_value_diff = -1 * (np.dot(eCF_, Sf_) / dCF_) * value_.calc_prop(mesh_, user_, "cp", "faces", it3) * (miu_f_ / Pr_ + miut_ / 0.9)
                        self.set_lhs([it1, it2], lhs_value + lhs_value_diff, add = True)
                        prev_row = it1
                gradC_u_ = value_.cells("grad", "u")[prev_row][1]
                gradC_v_ = value_.cells("grad", "v")[prev_row][1]
                gradC_w_ = value_.cells("grad", "w")[prev_row][1]
                phi_v_ = 2 * (gradC_u_[0]**2 + gradC_v_[1]**2 + gradC_w_[2]**2) + pow(gradC_u_[1] + gradC_v_[0], 2) \
                        + pow(gradC_u_[2] + gradC_w_[0], 2) + pow(gradC_v_[2] + gradC_w_[1], 2)
                Ret_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * pow(value_.cells("unit", "k")[prev_row][1], 2) / \
                        (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) * value_.cells("unit", "e")[prev_row][1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([value_.cells("grad", "u")[prev_row][1], value_.cells("grad", "v")[prev_row][1], value_.cells("grad", "w")[prev_row][1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5; St_tensor_C_T = np.transpose(St_tensor_C_)
                St_C_ = np.sqrt(np.dot(St_tensor_C_[0], St_tensor_C_T[0]) + np.dot(St_tensor_C_[1], St_tensor_C_T[1]) + np.dot(St_tensor_C_[2], St_tensor_C_T[2]))
                try:
                    with np.errstate(divide='raise'):
                        ts_C_ = np.min(np.array([value_.cells("unit", "k")[prev_row][1] / value_.cells("unit", "e")[prev_row][1], value_.calc_prop(mesh_, user_, "alpha", "cells", prev_row) / (pow(6, 0.5) * cmiu_C_ * St_C_)]))
                        miut_C_ = value_.calc_prop(mesh_, user_, "rho", "cells", prev_row) * cmiu_C_ * value_.cells("unit", "k")[prev_row][1] * ts_C_
                except FloatingPointError:
                    miut_C_ = 0
                bC_ += (value_.calc_prop(mesh_, user_, "miu", "cells", prev_row) + miut_C_) * phi_v_ * mesh_.cells[prev_row].volume
                self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                fc_ = mesh_.templates.iter("fc", "fluid")
                for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
                    v_ = np.array([value_.faces("unit", "u")[it2][1], value_.faces("unit", "v")[it2][1], value_.faces("unit", "w")[it2][1]])
                    self.calc_bound(mesh_, user_, value_, v_, it1, it2)
            elif it1 == "solid":
                aC_ = 0
                bC_ = 0
                cc_ = mesh_.templates.iter("cc", "solid"); cc_ = sorted(zip(cc_[0], cc_[1], cc_[2]), key = lambda x: x[0])
                prev_row = cc_[0][0]
                for it1, it2, it3 in cc_:
                    Sf_ = mesh_.geoms.Sf(False, it1, it3)
                    dCf_ = mesh_.geoms.dCf(False, it1, it3)
                    eCF_ = mesh_.geoms.eCF(False, it1, it2)
                    dCF_ = mesh_.geoms.dCF(True, it1, it2)
                    graditr_ = value_.linear_itr("grad", mesh_, [it1, it2, it3], "T")
                    if prev_row == it1:
                        aC_ += (np.dot(eCF_, Sf_) / dCF_) * value_.calc_prop(mesh_, user_, "k", "cells", it1)
                        bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * value_.calc_prop(mesh_, user_, "k", "cells", it1)
                        lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * value_.calc_prop(mesh_, user_, "k", "cells", it1)
                        self.set_lhs([it1, it2], lhs_value_diff)
                        prev_row = it1
                    else:
                        self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                        aC_ = 0; bC_ = 0
                        aC_ += (np.dot(eCF_, Sf_) / dCF_) * value_.calc_prop(mesh_, user_, "k", "cells", it1)
                        bC_ += np.dot(graditr_ - (np.dot(graditr_, eCF_) * eCF_), Sf_) * value_.calc_prop(mesh_, user_, "k", "cells", it1)
                        lhs_value_diff = (-1) * (np.dot(eCF_, Sf_) / dCF_) * value_.calc_prop(mesh_, user_, "k", "cells", it1)
                        self.set_lhs([it1, it2], lhs_value_diff)
                        prev_row = it1
                self.set_lhs([prev_row, prev_row], aC_); self.set_rhs([prev_row, 0], bC_)
                fc_ = mesh_.templates.iter("fc", "solid")
                for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
                    v_ = np.array([0, 0, 0])
                    self.calc_bound(mesh_, user_, value_, v_, it1, it2)           
            elif it1 == "conj":
                cc_ = mesh_.templates.iter("cc", "conj")
                for it1, it2, it3 in zip(cc_[0], cc_[1], cc_[2]):
                    self.calc_conj(mesh_, user_, value_, [it1, it2, it3])
    def calc_bound(self, mesh_ : mesh, user_ : user, value_ : value, v_, row : int, col : int):
        if "hamb" in mesh_.faces[col].boundary:
            Tamb_ = user_.constants.loc[0, "Tamb"]
            Tsky_ = 0.0552 * pow(Tamb_, 1.5)
            Tf_ = value_.faces("unit", "T")[col][1]
            try:
                with np.errstate(divide='raise'):
                    hsky_ = 5.67 * pow(10, -8) * value_.calc_prop(mesh_, user_, "eps", "cells", row) * \
                            (Tf_ + Tsky_) * (Tf_**2 + Tsky_**2) * (Tf_ - Tsky_) / (Tf_ - Tamb_)
            except FloatingPointError:
                hsky_ = 0.00
            Tfilm_ = (Tf_ + Tamb_) / 2
            rho_film_ = 1 / HAPropsSI("Vha", "P", float(user_.constants.loc[0, "Pamb"]), "T", float(Tfilm_), "W", float(user_.constants.loc[0, "Wamb"]))
            miu_film_ = HAPropsSI("mu", "P", float(user_.constants.loc[0, "Pamb"]), "T", float(Tfilm_), "W", float(user_.constants.loc[0, "Wamb"]))
            cp_film_ = HAPropsSI("cp_ha", "P", float(user_.constants.loc[0, "Pamb"]), "T", float(Tfilm_), "W", float(user_.constants.loc[0, "Wamb"]))  
            k_film_ = HAPropsSI("k", "P", float(user_.constants.loc[0, "Pamb"]), "T", float(Tfilm_), "W", float(user_.constants.loc[0, "Wamb"]))  
            alpha_film_ = k_film_ * rho_film_ / cp_film_
            RaL_ = 9.81 * (Tf_ - Tamb_) * pow(mesh_.faces[col].area, 1.5) * rho_film_ / (Tfilm_ * miu_film_ * alpha_film_)             
            if RaL_ <= pow(10, 7):
                Nu_N_ = 0.54 * pow(RaL_, 0.25)
            else:
                Nu_N_ = 0.15 * pow(RaL_, 0.25)
            hconv_ = Nu_N_ * k_film_ / np.sqrt(mesh_.faces[col].area)
            print(RaL_, Nu_N_, k_film_, hsky_, hconv_)
            rhs_value = -1 * (hsky_ + hconv_) * (Tf_ - Tamb_) * mesh_.geoms.Sf(True, row, col)
            print("hamb", rhs_value)
            self.set_rhs([row, 0], rhs_value, add = True)
            print("hamb", self.rhs[row, 0])
        elif any(["irr" in it1 for it1 in mesh_.faces[col].boundary]) is True:
            # Von Neumann
            source_id = [it1 for it1 in mesh_.faces[col].boundary if "irr" in it1][0]
            rhs_value = user_.sources.loc[0, source_id] * mesh_.geoms.Sf(True, row, col)
            self.set_rhs([row, 0], rhs_value, add = True)
            print("irr", self.rhs[row, 0])
        elif any(["s2s" in it1 for it1 in mesh_.faces[col].boundary]) is True:
            # Von Neumann
            rhs_value = value_.faces("unit", "q")[col][1] * mesh_.geoms.Sf(True, row, col)
            self.set_rhs([row, 0], rhs_value, add = True)
            print("s2s", self.rhs[row, 0])
        elif "inlet" in mesh_.faces[col].boundary:
            # specified value; zero gradient at inlet
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vin_v0_ = value_.cells("grad", "u")[row][1] - \
                           (np.dot(value_.cells("grad", "u")[row][1], eCf_) * eCf_)
            grad_vin_v1_ = value_.cells("grad", "v")[row][1] - \
                           (np.dot(value_.cells("grad", "v")[row][1], eCf_) * eCf_)
            grad_vin_v2_ = value_.cells("grad", "w")[row][1] - \
                           (np.dot(value_.cells("grad", "w")[row][1], eCf_) * eCf_)
            vin_v0_ = value_.cells("unit", "u")[row][1] + np.dot(grad_vin_v0_, dCf_)
            vin_v1_ = value_.cells("unit", "v")[row][1] + np.dot(grad_vin_v1_, dCf_)
            vin_v2_ = value_.cells("unit", "w")[row][1] + np.dot(grad_vin_v2_, dCf_)
            vin_ = np.array([vin_v0_, vin_v1_, vin_v2_])
            grad_Tin_ = value_.cells("grad", "T")[row][1] - \
                        (np.dot(value_.cells("grad", "T")[row][1], eCf_) * eCf_)
            Tin_ =  user_.constants.loc[0, "Tamb"] 
            rhs_value = -1 * value_.calc_prop(mesh_, user_, "cp", "faces", col) * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vin_, Sf_)) * (Tin_ + np.dot(grad_Tin_, dCf_))
            self.set_rhs([row, 0], rhs_value, add = True)
            print("inlet", self.rhs[row, 0])
        elif "outlet" in mesh_.faces[col].boundary:
            # fully developed flow; zero gradient at outlet
            Sf_ = mesh_.geoms.Sf(False, row, col)
            eCf_ = mesh_.geoms.eCf(False, row, col)
            dCf_ = mesh_.geoms.dCf(False, row, col)
            grad_vout_v0_ = value_.cells("grad", "u")[row][1] - \
                           (np.dot(value_.cells("grad", "u")[row][1], eCf_) * eCf_)
            grad_vout_v1_ = value_.cells("grad", "v")[row][1] - \
                           (np.dot(value_.cells("grad", "v")[row][1], eCf_) * eCf_)
            grad_vout_v2_ = value_.cells("grad", "w")[row][1] - \
                           (np.dot(value_.cells("grad", "w")[row][1], eCf_) * eCf_)
            vout_v0_ = value_.cells("unit", "u")[row][1] + np.dot(grad_vout_v0_, dCf_)
            vout_v1_ = value_.cells("unit", "v")[row][1] + np.dot(grad_vout_v1_, dCf_)
            vout_v2_ = value_.cells("unit", "w")[row][1] + np.dot(grad_vout_v2_, dCf_)
            vout_ = np.array([vout_v0_, vout_v1_, vout_v2_])
            grad_Tout_ = value_.cells("grad", "T")[row][1] - \
                         (np.dot(value_.cells("grad", "T")[row][1], eCf_) * eCf_)
            lhs_value = value_.calc_prop(mesh_, user_, "cp", "faces", col) * value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_)
            rhs_value = -1 * value_.calc_prop(mesh_, user_, "cp", "faces", col) * (value_.calc_prop(mesh_, user_, "rho", "faces", col) * np.dot(vout_, Sf_) * \
                                      np.dot(grad_Tout_, dCf_))
            self.set_lhs([row, row], lhs_value, add = True)
            self.set_rhs([row, 0], rhs_value, add = True)
            print("outlet", self.rhs[row, 0])
        else:
            pass
    def calc_conj(self, mesh_, user_ : user, value_ : value, id_ : list):
        if "fluid" in mesh_.cells[id_[0]].domain[0]:
            fluid_id = id_[0]
            solid_id = id_[1]
        else:
            fluid_id = id_[1]
            solid_id = id_[0]
        v_ = np.array([value_.faces("unit", "u")[fluid_id][1], value_.faces("unit", "v")[fluid_id][1], value_.faces("unit", "w")[fluid_id][1]])
        rho_ = value_.calc_prop(mesh_, user_, "rho", "cells", fluid_id)
        miu_ = value_.calc_prop(mesh_, user_, "miu", "cells", fluid_id)
        cp_ = value_.calc_prop(mesh_, user_, "cp", "cells", fluid_id)
        Ret_ = rho_ * pow(value_.cells("unit", "k")[fluid_id][1], 2) / (miu_ * value_.cells("unit", "e")[fluid_id][1])
        cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_/50, 2))
        hb_ = rho_ * cp_ * pow(cmiu_, 0.25) * np.sqrt(value_.cells("unit", "k")[fluid_id][1]) / value_.cells("unit", "T")[fluid_id][1]
        graditr_ = value_.linear_itr("grad", mesh_, id_, "T")
        if fluid_id == id_[0]:
            lhs_value = hb_ * np.dot(mesh_.geoms.eCF(False, fluid_id, solid_id), mesh_.geoms.dCf(False, fluid_id, id_[2])) * mesh_.geoms.Sf(True, fluid_id, id_[2]) / mesh_.geoms.dCF(True, fluid_id, solid_id)
            rhs_value = hb_ * np.dot((graditr_ - np.dot(graditr_, mesh_.geoms.eCF(False, fluid_id, solid_id)) * mesh_.geoms.eCF(False, fluid_id, solid_id)), mesh_.geoms.dCf(False, fluid_id, solid_id))
            print(id_, lhs_value, rhs_value)
            self.set_lhs([fluid_id, fluid_id], lhs_value, add = True)
            self.set_lhs([fluid_id, solid_id], -1 * lhs_value, add = True)
            self.set_rhs([fluid_id, 0], rhs_value, add = True)
        else:
            lhs_value = hb_ * np.dot(mesh_.geoms.eCF(False, solid_id, fluid_id), mesh_.geoms.dCf(False, solid_id, id_[2])) * mesh_.geoms.Sf(True, solid_id, id_[2]) / mesh_.geoms.dCF(True, solid_id, fluid_id)
            rhs_value = hb_ * np.dot((graditr_ - np.dot(graditr_, mesh_.geoms.eCF(False, solid_id, fluid_id)) * mesh_.geoms.eCF(False, solid_id, fluid_id)), mesh_.geoms.dCf(False, solid_id, fluid_id))
            print(id_, lhs_value, rhs_value)
            self.set_lhs([solid_id, solid_id], lhs_value, add = True)
            self.set_lhs([solid_id, fluid_id], -1 * lhs_value, add = True)
            self.set_rhs([solid_id, 0], rhs_value, add = True) 
    def calc_post_bound(self, mesh_ : mesh, user_ : user, value_ : value):
        fc_ = mesh_.templates.iter("fc", "fluid")
        for it1, it2, it3 in zip(fc_[0], fc_[1], fc_[2]):
            if "hamb" in mesh_.faces[it2].boundary:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_Thamb_ = value_.cells("grad", "T")[it1][1] - \
                         (np.dot(value_.cells("grad", "T")[it1][1], eCf_) * eCf_)
                Thamb_ = value_.cells("unit", "T")[it1][1] + np.dot(grad_Thamb_, dCf_)
                value_.set_faces("unit", "T", it2, Thamb_); value_.set_faces("grad", "T", it2, grad_Thamb_)
            elif "inlet" in mesh_.faces[it2].boundary:
                # specified value; zero gradient at inlet
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_Tin_ = value_.cells("grad", "T")[it1][1] - \
                            (np.dot(value_.cells("grad", "T")[it1][1], eCf_) * eCf_)
                Tin_ =  user_.constants.loc[0, "Tamb"] 
                value_.set_faces("unit", "T", it2, Tin_); value_.set_faces("grad", "T", it2, grad_Tin_)
            elif "outlet" in mesh_.faces[it2].boundary:
                # fully developed flow; zero gradient at outlet
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                grad_Tout_ = value_.cells("grad", "T")[it1][1] - \
                            (np.dot(value_.cells("grad", "T")[it1][1], eCf_) * eCf_)
                Tout_ = value_.cells("unit", "T")[it1][1] + np.dot(grad_Tout_, dCf_)
                value_.set_faces("unit", "T", it2, Tout_); value_.set_faces("grad", "T", it2, grad_Tout_)
        neigh_ = mesh_.templates.iter("cc", "conj")
        for it1, it2, it3 in zip(neigh_[0], neigh_[1], neigh_[2]):
            if "solid" in mesh_.cells[it1].domain[0]:
                Sf_ = mesh_.geoms.Sf(False, it1, it2)
                eCf_ = mesh_.geoms.eCf(False, it1, it2)
                dCf_ = mesh_.geoms.dCf(False, it1, it2)
                change_ = np.dot(value_.cells("grad", "T")[it1][1], eCf_)
                grad_Tconj_ = value_.cells("grad", "T")[it1][1] - np.array([it1 * change_ for it1 in eCf_])
                Tconj_ = value_.cells("unit", "T")[it1][1] + np.dot(grad_Tconj_, dCf_)
                value_.set_faces("unit", "T", it2, Tconj_); value_.set_faces("grad", "T", it2, grad_Tconj_)
    def calc_wall(self, mesh_ : mesh, user_ : user, value_ : value):
        if "conj" in list(mesh_.templates.cc.keys()):
            conj_ = mesh_.templates.iter("cc", "conj")
            for it1, it2, it3 in zip(conj_[0], conj_[1], conj_[2]):
                if "fluid" in mesh_.cells[it1].domain[0]:
                    v_ = np.array([0, 0, 0])
                    v_[0] = value_.cells("unit", "u")[it1][1]
                    v_[1] = value_.cells("unit", "v")[it1][1]
                    v_[2] = value_.cells("unit", "w")[it1][1]
                    v_val_ = np.sqrt(np.sum(np.array([pow(it4, 2) for it4 in v_])))
                    Ret_ = value_.calc_prop(mesh_, user_, "rho", "cells", it1) * pow(value_.cells("unit", "k")[it1][1], 2) / \
                        (value_.calc_prop(mesh_, user_, "miu", "cells", it1) * value_.cells("unit", "e")[it1][1])
                    cmiu_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_/50, 2))
                    gradCfluid_ = value_.cells("grad", "T")[it1][1]; val_gradCfluid = np.sqrt(np.sum(np.array([pow(it4, 2) for it4 in gradCfluid_])))
                    dperp_ = pow(2 * value_.cells("unit", "T")[it1][1] - 1, 0.5) * val_gradCfluid
                    dCplus_ = dperp_ * pow(cmiu_, 0.25) * np.sqrt(value_.cells("unit", "T")[it1][1]) * \
                            value_.calc_prop(mesh_, user_, "rho", "cells", it1) / value_.calc_prop(mesh_, user_, "miu", "cells", it1)
                    try:
                        with np.errstate(divide='raise'):
                            miutau_ = v_val_ * 0.41 / math.log(dCplus_) + 5.25
                            dplusT_ = dperp_ * miutau_ * value_.calc_prop(mesh_, user_, "rho", "cells", it1) / value_.calc_prop(mesh_, user_, "miu", "cells", it1)
                            Pr_ = value_.calc_prop(mesh_, user_, "miu", "cells", it1) / (value_.calc_prop(mesh_, user_, "rho", "cells", it1) * value_.calc_prop(mesh_, user_, "alpha", "cells", it1))
                            beta_ = pow(3.85 * pow(Pr_, 1/3) - 1.3, 2) + 2.12 * math.log(Pr_)
                            Tplus_ = 2.21 * math.log(abs(dplusT_)) + beta_ * Pr_
                            value_.set_cells("unit", "T", it1, Tplus_) 
                    except FloatingPointError:
                        pass
        else:
            pass
    def iter_solve(self, mesh_ : mesh, user_ : user, value_ : value, under_relax_, tol_, max_iter_, time_step_, current_time_):
        self.calc_coef(mesh_, user_, value_)
        lhs_transient_, rhs_transient_ = super().calc_transient(mesh_, user_, value_, "T", time_step_, current_time_)
        under_relax_b = deepcopy(rhs_transient_)
        for it1 in range(0, lhs_transient_.shape[0]):
            lhs_transient_[it1, it1] = lhs_transient_[it1, it1] / under_relax_
        for it1 in range(0, rhs_transient_.shape[0]):
            under_relax_b[it1, 0] = lhs_transient_[it1, it1] * float(value_.cells("unit", "T")[it1][1]) * (1 - under_relax_)
        A = lhs_transient_.tocsr()
        b = rhs_transient_ + under_relax_b; b = b.tocsr()
        M_x = lambda x: sparse.linalg.spsolve(A, x)
        M_ = sparse.linalg.LinearOperator(A.shape, M_x)
        x, exitCode = sparse.linalg.gmres(A, b.toarray(), tol = tol_, maxiter = max_iter_, M = M_)
        print(A); print(b); print(x)
        return x
class s2s(linear):
    def __init__(self, mesh_ : mesh):
        super().__init__(mesh_, what = ["s2s"])
    
    def calc_coef(self, mesh_ : mesh, user_ : user, value_ : value):
        cc_ = mesh_.templates.iter("cc", "s2s")
        prev_row = cc_[0][0]
        for it1, it2, it3 in zip(cc_[0], cc_[1], cc_[2]):
            if prev_row == it1:
                Tclust_C_ = 0
                rho_clust_F_ = 0
                for it4 in mesh_.clusts[it1].faces:
                    Tclust_C_ += value_.faces("unit", "T")[it4][1] * mesh_.faces[it4].area
                    eps_clust_C_ = value_.calc_prop(mesh_, user_, "eps", "faces", it4)
                for it4 in mesh_.clusts[it2].faces:
                    rho_clust_F_ += value_.calc_prop(mesh_, user_, "rho", "faces", it4) * mesh_.faces[it4].area
                Tclust_C_ = Tclust_C_ / mesh_.clusts[it1].area
                rho_clust_F_ = rho_clust_F_ / mesh_.clusts[it1].area        
                lhs_value = rho_clust_F_ * it3
                self.set_lhs([it1, it2], lhs_value)
                prev_row = it1
            else:
                self.set_lhs([prev_row, prev_row], 1)
                rhs_value = eps_clust_C_ * 5.67 *pow(10, -8) * pow(Tclust_C_, 4)
                self.set_rhs([prev_row, 0], rhs_value)
                Tclust_C_ = 0
                rho_clust_F_ = 0
                for it4 in mesh_.clusts[it1].faces:
                    Tclust_C_ += value_.faces("unit", "T")[it4][1] * mesh_.faces[it4].area
                    eps_clust_C_ = value_.calc_prop(mesh_, user_, "eps", "faces", it4)
                for it4 in mesh_.clusts[it2].faces:
                    rho_clust_F_ += value_.calc_prop(mesh_, user_, "rho", "faces", it4) * mesh_.faces[it4].area
                Tclust_C_ = Tclust_C_ / mesh_.clusts[it1].area
                rho_clust_F_ = rho_clust_F_ / mesh_.clusts[it1].area       
                lhs_value = rho_clust_F_ * it3
                self.set_lhs([it1, it2], lhs_value)
                prev_row = it1
        self.set_lhs([prev_row, prev_row], 1)
        rhs_value = eps_clust_C_ * 5.67 * pow(10, -8) * pow(Tclust_C_, 4)
        self.set_rhs([prev_row, 0], rhs_value)
    def update_source(self, mesh_ : mesh, value_ : value, new_val_):
        for it1 in range(0, new_val_.shape[0]):
            for it2 in mesh_.clusts[it1].faces:
                value_.set_faces("unit", "q", it2, new_val_[it1])
    def iter_solve(self, mesh_ : mesh, user_ : user, value_ : value, tol_, max_iter_):
        self.calc_coef(mesh_, user_, value_)
        A = self.lhs.tocsr()
        b = self.rhs.tocsr()
        x, exitCode = sparse.linalg.gmres(A, b.toarray(), tol = tol_, maxiter = max_iter_)
        x = x[:len(list(mesh_.clusts.keys()))]
        print("source: {}".format(x))
        self.update_source(mesh_, value_, x)
        return

##### cfd solver #########
class result:
    def __init__(self, mesh_ : mesh):
        self.__val_face = dict({}); self.__val_cell = dict({})
        for it1 in list(mesh_.faces.keys()):
            self.__val_face[it1] = dict({"P": [], "u": [], "v": [], "w": [], "k": [], "e": [], "T": []})
        for it1 in list(mesh_.cells.keys()):
            self.__val_cell[it1] = dict({"P": [], "u": [], "v": [], "w": [], "k": [], "e": [], "T": []})
        self.__res_loop = dict({"P": [], "u": [], "v": [], "w": [], "k": [], "e": [], "T": []})
        self.__res_time = dict({"P": [], "u": [], "v": [], "w": [], "k": [], "e": [], "T": []})
    
    def val_face(self):
        return self.__val_face
    def set_val_face(self, id_ : int, what_ : str, val_):
        self.__val_face[id_][what_].append(val_)
    def val_cell(self):
        return self.__val_cell
    def set_val_cell(self, id_ : int, what_ : str, val_):
        self.__val_cell[id_][what_].append(val_)
    def res_loop(self):
        return self.__res_loop
    def set_res_loop(self, what_ : str, val_):
        self.__res_loop[what_].append(val_)
    def res_time(self):
        return self.__res_time
    def set_res_time(self, what_ : str, val_):
        self.__res_time[what_].append(val_)
    @staticmethod
    def export(self):
        # mesh, result to database
        return
class solver:
    def __init__(self, mesh_ : mesh, user_ : user, value_ : value, under_relax, tol, max_iter, max_loop, time_step, max_time_step):
        self.__eq = dict({"Pcor": pcorrect(mesh_), "u": momentum(mesh_, 0), "v": momentum(mesh_, 1), "w": momentum(mesh_, 2), \
                    "k": turb_k(mesh_), "e": turb_e(mesh_), "T": energy(mesh_), "s2s": s2s(mesh_)})
        self.__under_relax = under_relax
        self.__tol = tol
        self.__max_iter = max_iter
        self.__max_loop = max_loop
        self.__time_step = time_step
        self.__max_time_step = max_time_step
        self.__current_time = 0
    
    def eq(self, what_ : str):
        return self.__eq[what_]
    @property
    def under_relax(self):
        return self.__under_relax
    @property
    def tol(self):
        return self.__tol
    @property
    def max_iter(self):
        return self.__max_iter
    @property
    def max_loop(self):
        return self.__max_loop
    @property
    def time_step(self):
        return self.__time_step
    @property
    def max_time_step(self):
        return self.__max_time_step
    @property
    def current_time(self):
        return self.__current_time
    @current_time.setter
    def current_time(self, new_time : bool):
        if new_time is True:
            self.__current_time += 1
    
    def forward_time(self, value_ : value, result_ : result):
        check_res_time = []
        for it1 in ["P", "Pcor", "u", "v", "w", "k", "e", "T", "q"]:
            for it2 in range(0, value_.cells("unit", it1).shape[0]):
                value_.set_cells("unit", it1, it2, 0, isprev = True)
                value_.set_cells("grad", it1, it2, 0, isprev = True)
            for it2 in range(0, value_.faces("unit", it1).shape[0]):
                value_.set_faces("unit", it1, it2, 0, isprev = True)
                value_.set_faces("grad", it1, it2, 0, isprev = True)
        for it1 in ["P", "u", "v", "w", "k", "e", "T"]:
            for it2 in list(result_.val_cell().keys()):
                result_.set_val_cell(it2, it1, value_.cells("unit", it1)[it2][0])
            for it2 in list(result_.val_face().keys()):
                result_.set_val_face(it2, it1, value_.faces("unit", it1)[it2][0])
        for it1 in ["u", "v", "w", "k", "e", "T"]:
            result_.set_res_time(it1, result_.res_loop()[it1][-1])
            check_res_time.append(result_.res_loop()[it1][-1] < self.tol)
        self.current_time(True)
        return all(check_res_time)
    def energy_loop(self, mesh_ : mesh, user_ : user, value_ : value, result_ : result):
        # stop condition -> all equations return rmsr value < tol
        print("energy loop")
        ctrl = 0; passes_total = 0; passes_turb = 0
        check = [False, False]
        while ctrl < self.max_loop:
            passes_total = 0; passes_turb = 0
            check = [False, False]
            while passes_total < self.max_loop:
                if check[0] is True:
                    break
                print("calc s2s")
                self.eq("s2s").iter_solve(mesh_, user_, value_, self.tol, self.max_iter)
                print("calc energy")
                x_T_ = self.eq("T").iter_solve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time)
                rmsr_t_ = value_.update_value(mesh_, "T", x_T_)
                self.eq("T").calc_post_bound(mesh_, user_, value_)
                self.eq("T").calc_wall(mesh_, user_, value_)
                result_.set_res_loop("T", rmsr_t_)
                check[0] = rmsr_t_ < self.tol
                passes_total += 1
            while passes_turb < self.max_loop:
                if check[1] is True:
                    break
                print("calc turb_k")
                x_k_ = self.eq("k").iter_solve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time)
                rmsr_k_ = value_.update_value(mesh_, "k", x_k_)
                self.eq("k").calc_post_bound(mesh_, user_, value_)
                self.eq("k").calc_wall(mesh_, user_, value_)
                result_.set_res_loop("k", rmsr_k_)
                print("calc turb_e")
                x_e_ = self.eq("e").iter_solve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time)
                rmsr_e_ = value_.update_value(mesh_, "e", x_e_)
                self.eq("e").calc_post_bound(mesh_, user_, value_)
                self.eq("e").calc_wall(mesh_, user_, value_)
                result_.set_res_loop("e", rmsr_e_)
                check[1] = all([i < self.__tol for i in [rmsr_k_, rmsr_e_]])
                passes_turb += 1
            ctrl += 1
            if all([passes_total + passes_turb < 4]) is True:
                print("energy done")
                break
        return ctrl
    def SIMPLE_loop(self, mesh_ : mesh, user_ : user, value_ : value, result_ : result):
        # stop condition -> all equations return rmsr value < tol
        print("SIMPLE loop")
        ctrl = 0; passes_v = 0
        check = [False, False]
        while ctrl < self.max_iter:
            passes_v = 0
            check = [False, False]
            while passes_v < self.max_iter:
                if all(check) is True:
                    break
                print("calc v")
                x_u_ = self.eq("u").iter_solve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time)
                x_v_ = self.eq("v").iter_solve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time)
                x_w_ = self.eq("w").iter_solve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time)
                rmsr_u_ = value_.update_value(mesh_, "u", x_u_); rmsr_v_ = value_.update_value(mesh_, "v", x_v_); rmsr_w_ = value_.update_value(mesh_, "w", x_w_)
                self.eq("u").calc_post_bound(mesh_, user_, value_); self.eq("v").calc_post_bound(mesh_, user_, value_); self.eq("w").calc_post_bound(mesh_, user_, value_)
                self.eq("u").calc_wall(mesh_, user_, value_); self.eq("v").calc_wall(mesh_, user_, value_); self.eq("w").calc_wall(mesh_, user_, value_)
                result_.set_res_loop("u", rmsr_u_); result_.set_res_loop("v", rmsr_v_); result_.set_res_loop("w", rmsr_w_)
                print("calc pcor")
                self.eq["Pcor"].itersolve(mesh_, user_, value_, self.under_relax, self.tol, self.max_iter, self.time_step, self.current_time, self.eq("u"), self.eq("v"), self.eq("w"))
                res_pcor = np.sum(np.array([value_.cells("unit", "Pcor")[it1][1] for it1 in range(0, value_.cells("unit", "Pcor").shape[0])])) / len(list(mesh_.cells.keys()))
                check[0] = all([it1 < self.tol for it1 in [rmsr_u_, rmsr_v_, rmsr_w_]])
                check[1] = all([res_pcor < self.tol])
                passes_v += 1
            ctrl += 1
            if all([passes_v < 2]) is True:
                print("SIMPLE done")
                break
        return ctrl
    def time_loop(self, mesh_ : mesh, user_ : user, value_ : value, result_ : result):
        # stop condition per loop groups -> all loops return rmsr value < tol for a determined number of passes
        print("time loop")
        ctrl_all = 0; passes_all = 0
        passes_energy = 10; passes_momentum = 10
        while ctrl_all < self.max_loop:
            if all([passes_energy + passes_momentum < 4]) is True:
                break
            passes_energy = self.energy_loop(mesh_, user_, value_, result_)
            passes_momentum = self.SIMPLE_loop(mesh_, user_, value_, result_)
            ctrl_all += 1
        print("Time {} done. ({}/{} iterations)".format(self.current_time, ctrl_all, self.max_iter))
        check_time = self.forward_time(value_, result_)
        return check_time
    def steady_loop(self, mesh_ : mesh, user_ : user, value_ : value, result_ : result):
        # stop condition if res_time [-1] < tol
        print("steady loop")
        ctrl_time = 0
        while ctrl_time < self.max_time_step:
            check_time = self.time_loop(mesh_, user_, value_, result_)
            if check_time is True:
                print("Done. Steady state achieved ({}s)".format(self.time_step * ctrl_time))
                return
            ctrl_time += 1
        print("Steady state not achieved ({}s)".format(self.time_step * self.max_time_step))
        return

user_test = user("test"); mesh_test = mesh("sc.msh"); value_test = value(mesh_test, user_test)
result_test = result(mesh_test); solver_test = solver(mesh_test, user_test, value_test, 0.2, 0.005, 1000, 10, 0.05, 10)

solver_test.steady_loop(mesh_test, user_test, value_test, result_test)

# TO DO
# export to database
# extract from database
# analysis