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
    __mesh = meshio.read(os.getcwd() + "\\problem\\test\\" + meshname)
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
        get_area = Decimal(0)
        get_centroid = [Decimal(0), Decimal(0), Decimal(0)]
        if len(self.nodes) == 3:
            get_coor = [[Decimal(it1) for it1 in points_dict[it2]] for it2 in self.nodes]
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
                get_coor = np.array([[Decimal(it2) for it2 in points_dict[it1]], 
                            [Decimal(it2) for it2 in points_dict[diag2[0]]],
                            [Decimal(it2) for it2 in points_dict[diag2[1]]]])
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
        get_volume = Decimal(0)
        get_centroid = [Decimal(0), Decimal(0), Decimal(0)]
        centre = np.array([Decimal(0), Decimal(0), Decimal(0)])
        for it1 in self.nodes:
            centre += np.array([Decimal(it2) for it2 in points_dict[it1]])
        centre = centre / len(self.nodes)
        for it1 in self.faces:
            get_fC = np.array([Decimal(it2) for it2 in np.array(centre - face_dict[it1].centroid)])
            get_fC_val = np.sqrt(np.sum(list(map(lambda it2: it2**2, get_fC))))
            vec1 = np.array(points_dict[face_dict[it1].nodes[2]] - points_dict[face_dict[it1].nodes[0]])
            vec2 = np.array(points_dict[face_dict[it1].nodes[1]] - points_dict[face_dict[it1].nodes[0]])
            get_Sf = np.array([Decimal(it2) for it2 in np.cross(vec1, vec2)])
            get_Sf_val = np.sqrt(np.sum(list(map(lambda it2: it2**2, get_Sf))))
            if np.dot(get_Sf, get_fC) > 0.00:
                get_Sf = get_Sf / get_Sf_val
            else:
                get_Sf = -1 * get_Sf / get_Sf_val
            cos = np.dot(get_Sf, get_fC) / get_fC_val
            part_height = cos * get_fC_val
            part_volume = part_height * face_dict[it1].area / 3
            part_centroid = np.array([Decimal(it2) for it2 in face_dict[it1].centroid]) + (Decimal(0.25) * get_fC)
            get_volume += part_volume; get_centroid += part_centroid * part_volume
        get_centroid = get_centroid / get_volume
        get_centroid = [round(it1, 3) for it1 in get_centroid]; get_volume = round(get_volume, 3)
        return np.array(get_centroid), get_volume
    def print_elements(self):
        print("nodes: {}\nfaces: {}\ncentroid: {}\nvolume: {}\ndomain: {}".format(self.nodes, self.faces, self.centroid, self.volume, self.domain))
        return
class clust:
    def __init__(self, face_dict : dict, face_list : list, clust_area : Decimal()):
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
        view = Decimal(0)
        r = clust2.centroid - clust1.centroid
        r_val = round(Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in r])))), 3)
        for it1 in clust1.faces:
            ef1 = np.cross(points_dict[face_dict[it1].nodes[2]] - points_dict[face_dict[it1].nodes[0]], 
                            points_dict[face_dict[it1].nodes[1]] - points_dict[face_dict[it1].nodes[0]])
            ef1 = np.array([round(Decimal(it2), 3) for it2 in ef1])
            ef1_val = round(Decimal(np.sqrt(np.sum(np.array([it3**2 for it3 in ef1])))), 3); ef1 = ef1 / ef1_val
            if np.dot(ef1, r) < 0.00:
                ef1 = Decimal(-1) * ef1
            cos1 = np.dot(ef1, r) / r_val
            for it2 in clust2.faces:
                ef2 = np.cross(points_dict[face_dict[it2].nodes[2]] - points_dict[face_dict[it2].nodes[0]], 
                                points_dict[face_dict[it2].nodes[1]] - points_dict[face_dict[it2].nodes[0]])
                ef2 = np.array([round(Decimal(it2), 3) for it2 in ef2])
                ef2_val = round(Decimal(np.sqrt(np.sum(np.array([it3**2 for it3 in ef2])))), 3); ef2 = ef2 / ef2_val
                if np.dot(ef2, r) > 0.00:
                    ef2 = Decimal(-1) * ef2
                cos2 = Decimal(np.dot(ef2, (-1) * r) / r_val)
                view += cos1 * cos2 / (Decimal(math.pi) * r_val**2)
        view1 = round(Decimal(view / clust1.area), 3)   
        view2 = round(Decimal(view / clust2.area), 3)
        return view1, view2

    def get_info(self, face_dict : dict):
        # centroid calc
        centroid = np.array([Decimal(0), Decimal(0), Decimal(0)])
        for it1 in self.faces:
            centroid += face_dict[it1].centroid * face_dict[it1].area
        centroid = centroid / self.__area
        centroid = np.array([round(it1, 3) for it1 in centroid])
        return centroid
    def print_elements(self):
        print("faces: {}\narea: {}\ncentroid: {}".format(self.faces, self.area, self.centroid))
        return
class connect:
    def __init__(self, cc_args : dict, fc_args : dict):
        self.__cc = cc_args
        self.__fc = fc_args
    @classmethod
    def copy(self, other):
        self.__cc = deepcopy(other.__cc)
        self.__fc = deepcopy(other.__fc)
        return self
    @property
    def cc(self):
        return self.__cc
    @property
    def fc(self):
        return self.__fc

    def get_coo(self, var : str, which : str):
        check = str("_connect__" + var)
        if check in dir(self):
            return self.__dict__[check][which].tocoo()
        else:
            print("Variable not found")
        return
    def get_csr(self, var : str, which : str):
        check = str("_connect__" + var)
        if check in dir(self):
            return self.__dict__[check][which].tocsr()
        else:
            print("Variable not found")
        return
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
                Sf = np.array([round(Decimal(it4), 3) for it4 in Sf])
                dCf = np.array([round(Decimal(it4), 3) for it4 in face_dict[it3].centroid - it2.centroid])
                dCf_val = np.sqrt(np.sum(np.array([it4**2 for it4 in dCf])))
                eCf = dCf / dCf_val
                if np.dot(Sf, dCf) < 0.00:
                    Sf = Decimal(-1) * Sf
                Sf_args[0].append([it1, it3, Sf[0]]); Sf_args[1].append([it1, it3, Sf[1]]); Sf_args[2].append([it1, it3, Sf[2]])
                dCf_args[0].append([it1, it3, dCf[0]]); dCf_args[1].append([it1, it3, dCf[1]]); dCf_args[2].append([it1, it3, dCf[2]])
                eCf_args[0].append([it1, it3, eCf[0]]); eCf_args[1].append([it1, it3, eCf[1]]); eCf_args[2].append([it1, it3, eCf[2]])
        for it1 in neigh_args:
            dCF = np.array([round(Decimal(it2), 3) for it2 in cell_dict[it1[1]].centroid - cell_dict[it1[0]].centroid])
            dCF_val = np.sqrt(np.sum(np.array([it4**2 for it4 in dCf])))
            eCF = np.array([round(Decimal(it2), 3) for it2 in dCF / dCF_val])
            dCF_args[0].append([it1[0], it1[1], dCF[0]]); dCF_args[1].append([it1[0], it1[1], dCF[1]]); dCF_args[2].append([it1[0], it1[1], dCF[2]])
            eCF_args[0].append([it1[0], it1[1], eCF[0]]); eCF_args[1].append([it1[0], it1[1], eCF[1]]); eCF_args[2].append([it1[0], it1[1], eCF[2]])
            for it2 in range(0, len(Sf_args[0])):
                if all([Sf_args[0][it2][0] == it1[0], Sf_args[0][it2][1] == it1[2]]) is True:
                    Sf = np.array([Sf_args[0][it2][2], Sf_args[1][it2][2], Sf_args[2][it2][2]])
                    Ef = eCF * np.dot(Sf, Sf) / np.dot(eCF, Sf)
                    Ef = np.array([round(Decimal(it3), 3) for it3 in Ef])
                    Tf = Sf - Ef
                    Ef_args[0].append([it1[0], it1[2], Ef[0]]); Ef_args[1].append([it1[0], it1[2], Ef[1]]); Ef_args[2].append([it1[0], it1[2], Ef[2]])
                    Tf_args[0].append([it1[0], it1[2], Tf[0]]); Tf_args[1].append([it1[0], it1[2], Tf[1]]); Tf_args[2].append([it1[0], it1[2], Tf[2]])
        self.__Sf = np.array(Sf_args); self.__Ef = np.array(Ef_args); self.__Tf = np.array(Tf_args)
        self.__dCf = np.array(dCf_args); self.__eCf = np.array(eCf_args); self.__dCF = np.array(dCF_args); self.__eCF = np.array(eCF_args)
    @property
    def Sf(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__Sf[0].shape[0]):
            if all([self.__Sf[0][it1][0] == id[1][0], self.__Sf[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__Sf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
        else:
            return coor
    @property
    def Ef(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__Ef[0].shape[0]):
            if all([self.__Ef[0][it1][0] == id[1][0], self.__Ef[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__Ef[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
        else:
            return coor
    @property
    def Tf(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__Tf[0].shape[0]):
            if all([self.__Tf[0][it1][0] == id[1][0], self.__Tf[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__Tf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
        else:
            return coor
    @property
    def dCf(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__dCf[0].shape[0]):
            if all([self.__dCf[0][it1][0] == id[1][0], self.__dCf[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__dCf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
        else:
            return coor
    @property
    def eCf(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__eCf[0].shape[0]):
            if all([self.__eCf[0][it1][0] == id[1][0], self.__eCf[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__eCf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
        else:
            return coor
    @property
    def dCF(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__dCF[0].shape[0]):
            if all([self.__dCF[0][it1][0] == id[1][0], self.__dCF[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__dCF[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
        else:
            return coor
    @property
    def eCF(self, is_unit : bool, id : list):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self.__eCF[0].shape[0]):
            if all([self.__eCF[0][it1][0] == id[1][0], self.__eCF[0][it1][1] == id[1][1]]) is True:
                coor = np.array([self.__eCF[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor]))))
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
                    clust_args[clust_id] = [[it1], Decimal(0)]
                else:
                    clust_args[clust_id][0].append(it1)
        if len(list(clust_args.keys())) > 0:
            for it1, it2 in clust_args.items():
                clust_args[it1][1] = round(Decimal(np.sum(np.array([face_dict[it2].area for it2 in it2[0]]))), 3)
                clust_dict[it1] = clust(face_dict, clust_args[it1][0], clust_args[it1][1])
        # connect args
        cc_dict = dict({}); fc_dict = dict({})
        for it1 in neigh_args:
            domain1 = cell_dict[it1[0]].domain[0]; domain2 = cell_dict[it1[1]].domain[0]
            print(domain1, domain2)
            check_domain = ["fluid" in it2 for it2 in [domain1, domain2]]
            which_cc = ""
            if any(check_domain) is True:
                if all(check_domain) is True:
                    which_cc = "fluid"
                    print("fluid")
                else:
                    which_cc = "conj"
            else:
                which_cc = "solid"
            if which_cc not in list(cc_dict.keys()):
                cc_dict[which_cc] = [it1]
            else:
                cc_dict[which_cc].append(it1)
        for it1, it2 in cc_dict.items():
            cc_dict[it1] = sparse.lil_matrix(it2)
        for it1 in bound_args:
            which_fc = [it2 for it2 in ["fluid", "solid"] if it2 in cell_dict[it1[0]].domain[0]]
            which_fc = which_fc[0]
            if which_fc not in list(fc_dict.keys()):
                fc_dict[which_fc] = [it1]
            else:
                fc_dict[which_fc].append(it1)
        for it1, it2 in fc_dict.items():
            fc_dict[it1] = sparse.lil_matrix(it2)
        if len(list(clust_args.keys())) > 0:
            cc_dict["s2s"] = []
            for it1 in range(0, len(list(clust_dict.keys())) - 1):
                for it2 in range(it1 + 1, len(list(clust_dict.keys()))):
                    view1, view2 = clust.match_view(points_dict, face_dict, clust_dict[list(clust_dict.keys())[it1]], clust_dict[list(clust_dict.keys())[it2]])
                    cc_dict["s2s"].append([it1, it2, view1])
                    cc_dict["s2s"].append([it2, it1, view2])
            cc_dict["s2s"] = sparse.lil_matrix(cc_dict["s2s"])
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
import numpy as np
from numpy import linalg
import scipy.sparse as sparse
from CoolProp.HumidAirProp import HAPropsSI
import math
# import mpi4py
# import cfd_scheme

def gauss_seidel(A, b, x = np.array([0, 0, 0], dtype = Decimal()), max_iterations = 50, tolerance = 0.005):
    #x is the initial condition
    iter1 = 0
    #Iterate
    for k in range(max_iterations):
        iter1 = iter1 + 1
        print ("The solution vector in iteration", iter1, "is:", x)    
        x_old  = x.copy()
        #Loop over rows
        for i in range(A.shape[0]):
            x[i] = (b[i] - np.dot(A[i,:i], x[:i]) - np.dot(A[i,(i+1):], x_old[(i+1):])) / A[i ,i]
        #Stop condition 
        #LnormInf corresponds to the absolute value of the greatest element of the vector.
        LnormInf = max(abs((x - x_old)))/max(abs(x_old))   
        print ("The L infinity norm in iteration", iter1,"is:", LnormInf)
        if  LnormInf < tolerance:
            break
    return x
class user:
    def __init__(self, *args):
        # args init_value : str, solid_props : str, const_value : str
        self.__inits = pd.read_csv(os.getcwd() + "\\problem\\test\\" + args[0])
        self.__solid_props = pd.read_csv(os.getcwd() + "\\problem\\test\\" + args[1], index_col = 0)
        self.__constants = pd.read_csv(os.getcwd() + "\\problem\\test\\" + args[2])
    @property
    def inits(self):
        pass
    @inits.getter
    def inits(self):
        return self.__inits
    @inits.deleter
    def inits(self):
        del self.__inits
    @property
    def solid_props(self):
        pass
    @solid_props.getter
    def solid_props(self):
        return self.__solid_props
    @solid_props.deleter
    def solid_props(self):
        return self.__solid_props
    @property
    def constants(self):
        pass
    @constants.getter
    def constants(self):
        return self.__constants
    @constants.deleter
    def constants(self):
        del self.__constants
class value:
    def __init__(self, mesh_ : mesh, user_ : user):
        # args mesh : mesh
        # all domains and clusts P, Pcor, u, v, w, k, e, T, q
        face_unit = dict({}); cell_unit = dict({})
        face_grad = dict({}); cell_grad = dict({})
        P_init = Decimal(user_.inits.loc[0, "P"]); Pcor_init = Decimal(user_.inits.loc[0, "Pcor"])
        u_init = Decimal(user_.inits.loc[0, "u"]); v_init = Decimal(user_.inits.loc[0, "v"])
        w_init = Decimal(user_.inits.loc[0, "w"]); k_init = Decimal(user_.inits.loc[0, "k"])
        e_init = Decimal(user_.inits.loc[0, "e"]); T_init = Decimal(user_.inits.loc[0, "T"])
        q_init = Decimal(user_.inits.loc[0, "q"])
        for it1, it2 in mesh_.cells.items():
            cell_unit[it1] = dict({"P": [Decimal(P_init), Decimal(P_init)],
                                     "Pcor": [Decimal(Pcor_init), Decimal(Pcor_init)],
                                     "u": [Decimal(u_init), Decimal(u_init)],
                                     "v": [Decimal(v_init), Decimal(v_init)],
                                     "w": [Decimal(w_init), Decimal(w_init)],
                                     "k": [Decimal(k_init), Decimal(k_init)],
                                     "e": [Decimal(e_init), Decimal(e_init)],
                                     "T": [Decimal(T_init), Decimal(T_init)],
                                     "q": [Decimal(q_init), Decimal(q_init)]})
            cell_grad[it1] = dict({"P": [[Decimal(P_init), Decimal(P_init), Decimal(P_init)],
                                         [Decimal(P_init), Decimal(P_init), Decimal(P_init)]],
                                    "Pcor": [[Decimal(Pcor_init), Decimal(Pcor_init), Decimal(Pcor_init)],
                                         [Decimal(Pcor_init), Decimal(Pcor_init), Decimal(Pcor_init)]],
                                     "u": [[Decimal(u_init), Decimal(u_init), Decimal(u_init)],
                                           [Decimal(u_init), Decimal(u_init), Decimal(u_init)]],
                                     "v": [[Decimal(v_init), Decimal(v_init), Decimal(v_init)],
                                           [Decimal(v_init), Decimal(v_init), Decimal(v_init)]],
                                     "w": [[Decimal(w_init), Decimal(w_init), Decimal(w_init)],
                                           [Decimal(w_init), Decimal(w_init), Decimal(w_init)]],
                                     "k": [[Decimal(k_init), Decimal(k_init), Decimal(k_init)],
                                           [Decimal(k_init), Decimal(k_init), Decimal(k_init)]],
                                     "e": [[Decimal(k_init), Decimal(e_init), Decimal(e_init)],
                                           [Decimal(k_init), Decimal(e_init), Decimal(e_init)]],
                                     "T": [[Decimal(T_init), Decimal(T_init), Decimal(T_init)],
                                           [Decimal(T_init), Decimal(T_init), Decimal(T_init)]],
                                     "q": [[Decimal(q_init), Decimal(q_init), Decimal(q_init)],
                                           [Decimal(q_init), Decimal(q_init), Decimal(q_init)]]})            
        for it1, it2 in mesh_.faces.items():
            face_unit[it1] = dict({"P": [Decimal(P_init), Decimal(P_init)],
                                     "Pcor": [Decimal(Pcor_init), Decimal(Pcor_init)],
                                     "u": [Decimal(u_init), Decimal(u_init)],
                                     "v": [Decimal(v_init), Decimal(v_init)],
                                     "w": [Decimal(w_init), Decimal(w_init)],
                                     "k": [Decimal(k_init), Decimal(k_init)],
                                     "e": [Decimal(e_init), Decimal(e_init)],
                                     "T": [Decimal(T_init), Decimal(T_init)],
                                     "q": [Decimal(q_init), Decimal(q_init)]})
            face_grad[it1] = dict({"P": [[Decimal(P_init), Decimal(P_init), Decimal(P_init)],
                                         [Decimal(P_init), Decimal(P_init), Decimal(P_init)]],
                                    "Pcor": [[Decimal(Pcor_init), Decimal(Pcor_init), Decimal(Pcor_init)],
                                         [Decimal(Pcor_init), Decimal(Pcor_init), Decimal(Pcor_init)]],
                                     "u": [[Decimal(u_init), Decimal(u_init), Decimal(u_init)],
                                           [Decimal(u_init), Decimal(u_init), Decimal(u_init)]],
                                     "v": [[Decimal(v_init), Decimal(v_init), Decimal(v_init)],
                                           [Decimal(v_init), Decimal(v_init), Decimal(v_init)]],
                                     "w": [[Decimal(w_init), Decimal(w_init), Decimal(w_init)],
                                           [Decimal(w_init), Decimal(w_init), Decimal(w_init)]],
                                     "k": [[Decimal(k_init), Decimal(k_init), Decimal(k_init)],
                                           [Decimal(k_init), Decimal(k_init), Decimal(k_init)]],
                                     "e": [[Decimal(k_init), Decimal(e_init), Decimal(e_init)],
                                           [Decimal(k_init), Decimal(e_init), Decimal(e_init)]],
                                     "T": [[Decimal(T_init), Decimal(T_init), Decimal(T_init)],
                                           [Decimal(T_init), Decimal(T_init), Decimal(T_init)]],
                                     "q": [[Decimal(q_init), Decimal(q_init), Decimal(q_init)],
                                           [Decimal(q_init), Decimal(q_init), Decimal(q_init)]]})   
        self.__cells = [cell_unit, cell_grad]; self.__faces = [face_unit, face_grad]
    @property
    def cells(self, what : str):
        return self.__cells[what]
    @cells.setter
    def cells(self, *args):
        # args value : Decimal / list(Decimal), id : int, what : str, is_prev : bool
        if args[2] is True:
            self.__cells[args[1]][args[2]][0] = args[0]
        else:
            self.__cells[args[1]][args[2]][1] = args[0]
    @property
    def faces(self, what : str):
        return self.__faces[what]
    @faces.getter
    def faces(self, *args):
        # args value : Decimal / list(Decimal), id : int, what : str, is_prev : bool
        if args[2] is True:
            self.__faces[args[1]][args[2]][0] = args[0]
        else:
            self.__faces[args[1]][args[2]][1] = args[0]

class linear:
    def __init__(self, mesh_ : mesh, what = ["fluid", "solid", "conj"]):
        self.__lhs = dict({}); self.__rhs = dict({})
        for it1 in what:
            self.__lhs[it1] = mesh_.templates.cc[it1]
            self.__rhs[it1] = np.zeros(shape=(mesh_.templates.cc[it1].shape[0], 1), dtype = Decimal)
    @staticmethod
    def linear_itr(*args):
        # args, mesh : mesh, value : value, id : [cell1, cell2, face], what : [variable name, dict key]
        gC = args[0].geoms.dCf(True, [args[2][0], args[2][2]]) / \
            (args[0].geoms.dCf(True, [args[2][0], args[2][2]]) + \
            args[0].geoms.dCf(True, [args[2][1], args[2][2]]))
        return gC * args[1].cells[args[3][0]]
        
        gC = args[0].geoms("dCf", args[3], args[4], True) / \
             (args[0].geoms("dCf", args[3], args[4], True) + \
             args[0].geoms("dCf", args[2], args[4], True))
        return gC * args[0].cells[args[2]](args[1][0])[args[1][1]][-1] + \
               (1 - gC) * args[0].cells[args[3]](args[1][0])[args[1][1]][-1]
    def QUICK_grad(self, *args):
        # args mesh : mesh, what : dict key, cell id 1 : int, cell id 2 : int, face id : int
        grad__ = self.linearitr(args[0], ["grad", args[1]], args[2], args[3], args[4])
        dCF__ = args[0].geoms("dCF", args[2], args[3], True)
        eCF__ = args[0].geoms("eCF", args[2], args[3], False)
        return grad__ + ((args[0].cells[args[3]].value[args[1]][-1] + \
               args[0].cells[args[2]].value[args[1]][-1]) / dCF__) - \
               (np.dot(grad__, eCF__)) * eCF__
    def QUICK_value(*args):
        # args mesh : mesh, what : dict key, cell id : int, face id : int
        return args[0].cells[args[2]].value[args[1]][-1] + 0.5 * np.dot(
               (args[0].cells[args[2]].grad[args[1]][-1] +
               args[0].faces[args[3]].grad[args[1]][-1]),
               args[0].geoms("dCf", args[2], args[3], False))
    def least_square_itr(self, *args):
        # mesh : mesh, cell id : int, what : str
        check = args[0].templates.neighbor["all"].toarray()[args[1], :]
        neighbor__ = [i for i in range(0, check.shape[0]) if check[i] != 0]
        lhs_leastsquare = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype = float)
        rhs_leastsquare = np.array([[0], [0], [0]])
        for i in range(0, 3):
            rhs_coef__ = 0.00
            for j in range(0, 3):
                lhs_coef__ = 0.00
                for k in neighbor__:
                    dCF__ = args[0].geoms("dCF", args[1], k, False)
                    dCF_val__ = args[0].geoms("dCF", args[1], k, True)
                    lhs_coef__ += dCF__[i] * dCF__[j] / dCF_val__
            for k in neighbor__:
               dCF__ = args[0].geoms("dCF", args[1], k, False)
               dCF_val__ = args[0].geoms("dCF", args[1], k, True)
               rhs_coef__ += dCF__[i] * (self.cells[k].value[args[2]][-1] - \
                             self.cells[i].value[args[2]][-1]) / dCF_val__
            lhs_leastsquare[i][j] = lhs_coef__
            rhs_leastsquare[i][0] = rhs_coef__
        grad__ = gauss_seidel(lhs_leastsquare, rhs_leastsquare)
        return grad__
    def update_value(self, *args):
        # args mesh : mesh, what : str, new_values : np.array([])
        # C and F value
        for i in args[0].cells.keys():
            args[0].cells.value[args[1]][-1] = args[2][i]
        # gradC least square itr based of C and F values
        for i in args[0].cells.keys():
            args[0].cells.grad[args[1]][-1] = self._leastsquareitr(i, args[1])
        # gradf and fvalue
        face_list = list(self.faces.keys())
        check = args[0].templates.neighbor["all"].tocoo()
        for i, j, k in zip(check.row, check.col, check.data):
            if k in check:
                args[0].faces[k].grad[args[1]][-1] = self.QUICKgrad(args[0], args[1], i, j, k)
                args[0].faces[k].value[args[1]][-1] = self.QUICKvalue(args[0], args[1], i, k)
                face_list.remove(k)
        return
    def calc_rmsr(*args):
        # args value new : np.array([]), value prev : np.array([])
        res__ = np.sum(np.array([pow(args[0][i] - args[1][i], 2) for i in range(0, args[0].shape[0])]))
        res__ = np.sqrt(res__ / args[0].shape[0])
        return res__
    def calc_transient(self, *args):
        # void
        # args mesh : mesh, what : str, time_step : int/double, current_time : int
        lhs_transient__ = self.lhs
        rhs_transient__ = self.rhs
        if args[3] == 0:
            for i in range(0, self.lhs.shape[0]):
                for j in range(0, self.lhs.shape[1]):
                    lhs_transient__[i][j] = lhs_transient__[i][j] + args[0].cells[i].prop.rhs[-1] * args[0].cells[i].volume / \
                                            (args[2])
            for i in range(0, len(self.rhs)):
                rhs_transient__[i][0] = self.rhs[i][0] - args[0].cells[i].prop.rhs[-2] * \
                                        args[0].cells[i].volume * args[0].cells[i].value[args[1]][-2] / \
                                        args[2]
        else:
            for i in self.lhs.shape[0]:
                for j in self.lhs.shape[1]:
                    lhs_transient__[i][j] = lhs_transient__[i][j] + args[0].cells[i].prop.rhs[-1] * args[0].cells[i].volume / \
                                            (2 * args[2])
            for i in range(0, len(self.rhs)):
                rhs_transient__[i][0] = self.rhs[i][0] - args[0].cells[i].prop.rhs[-2] * \
                                        args[0].cells[i].volume * args[0].cells[i].value[args[1]][-2] / \
                                        (2 * args[2])
        return lhs_transient__, rhs_transient__
    
class pcorrect(linear):
    def __init__(self, *args):
        # args mesh : mesh
        super().__init__(args[0])
    def calccoef(self, *args):
        # args mesh : mesh, u_ref : momentum, v_ref : momentum, w_ref : momentum, what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        neigh__ = args[0].templates.neighbor["fluid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Df_x__ = ((args[0].cells[i].volume / args[1].lhs[i][j]) + \
                     (args[0].cells[j].volume / args[1].lhs[i][j])) / 2
            Df_y__ = ((args[0].cells[i].volume / args[2].lhs[i][j]) + \
                     (args[0].cells[j].volume / args[2].lhs[i][j])) / 2
            Df_z__ = ((args[0].cells[i].volume / args[3].lhs[i][j]) + \
                     (args[0].cells[j].volume / args[3].lhs[i][j])) / 2
            Sf__  = args[0].geoms("Sf", i, k, False)
            dCF__ = args[0].geoms("dCF", i, j, False)
            Dau_f__ = (pow(Df_x__ * Sf__[0], 2) + pow(Df_y__ * Sf__[1], 2) +
                      pow(Df_z__ * Sf__[2], 2)) / (dCF__[0] * Df_x__ * Sf__[0] +
                      dCF__[1] * Df_y__ * Sf__[1] + dCF__[2] * Df_z__ * Sf__[2])
            if prev_row == i:
                aC__ += args[0].faces[k].prop.rhs[-1] * Dau_f__
                bC__ += args[0].faces[k].prop.rhs[-1] * \
                        np.dot(np.array([args[1].faces[k].value["u"][-1],
                        args[1].faces[k].value["v"][-1],
                        args[1].faces[k].value["w"][-1]]), Sf__) - \
                        np.dot(np.array([[Df_x__, 0, 0], [0, Df_y__, 0], [0, 0, Df_z__]]) * \
                        args[0].faces[k].grad["P"] - super().linearitr(args[0], ["grad", "P"], i, j, k), \
                        Sf__)
                self.lhs[i][j] = -args[0].faces[k].prop.rhs[-1] * Dau_f__
                prev_row = i
            else:
                self.lhs[prev_row][prev_row] = aC__
                self.rhs[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += args[0].faces[k].prop.rhs[-1] * Dau_f__
                bC__ += args[0].faces[k].prop.rhs[-1] * \
                        np.dot(np.array([args[1].faces[k].value["u"][-1],
                        args[1].faces[k].value["v"][-1],
                        args[1].faces[k].value["w"][-1]]), Sf__) - \
                        np.dot(np.array([[Df_x__, 0, 0], [0, Df_y__, 0], [0, 0, Df_z__]]) * \
                        args[0].faces[k].grad["P"] - super().linearitr(args[0], ["grad", "P"], i, j, k), \
                        Sf__)
                self.lhs[i][j] = -args[0].faces[k].prop.rhs[-1] * Dau_f__
                prev_row = i
        self.lhs[prev_row][prev_row] = aC__
        self.rhs[prev_row][0] = bC__
        bound__ = args[0].templates.boundary["fluid"].tocoo()
        for i, j, k in zip(bound__.row, bound__.col, bound__.data):
            for l in args[0].faces[j].bound:
                self.calcbound(args[0], l, i, j)
        return
    def calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int
        if "noslip" in args[1]:
            args[0].faces[args[3]].value["P"][-1] = args[0].cells[args[2]].value["P"][-1] - \
                                                          np.dot(args[0].cells[args[2]].grad["P"][-1], \
                                                          args[0].geoms("Sf", args[2], args[3], False)) - \
                                                          np.dot(args[0].faces[args[3]].grad["P"][-1], \
                                                          args[0].geoms("Tf", args[2], args[3], False)) / \
                                                          (self.lhs[args[2]][args[2]] / \
                                                          args[0].cells[args[2]].prop.rhs[-1])
        elif "inlet" in args[1]:
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.rhs[-1] * \
                                             self.lhs[args[2]][args[2]] / \
                                             args[0].cells[args[2]].prop.rhs[-1]
        elif "outlet" in args[1]:
            self.lhs[args[2]][0] += args[0].faces[args[3]].prop.rhs[-1] * \
                                       self.lhs[args[2]][args[2]] / \
                                       args[0].cells[args[2]].prop.rhs[-1]
        else:
            pass
        return
    def calccorrect(self, *args):
        # void
        # args mesh : mesh, u_ref : momentum, v_ref : momentum, w_ref : momentum
        for i in args[0].cells.keys():
            pcor_C__ = -args[0].cells[i].prop.rhs[-1] * args[0].cells[i].volume * \
                        args[0].cells[i].grad["Pcor"][-1] / self.lhs[i][i] 
            args[0].cells[i].value["u"][-1] += pcor_C__[0]
            args[0].cells[i].value["v"][-1] += pcor_C__[1]
            args[0].cells[i].value["w"][-1] += pcor_C__[2]
            args[0].cells[i].value["P"][-1] += args[0].cells[i].value["Pcor"][-1]
        neigh__ = args[0].templates.neighbor["fluid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Df_x__ = ((args[0].cells[i].volume / args[1].lhs[i][j]) + \
                     (args[0].cells[j].volume / args[1].lhs[i][j])) / 2
            Df_y__ = ((args[0].cells[i].volume / args[2].lhs[i][j]) + \
                     (args[0].cells[j].volume / args[2].lhs[i][j])) / 2
            Df_z__ = ((args[0].cells[i].volume / args[3].lhs[i][j]) + \
                     (args[0].cells[j].volume / args[3].lhs[i][j])) / 2
            Sf__  = args[0].geoms("Sf", i, k, False)
            pcor_f__ = -args[0].faces[k].prop.rhs[-1] * \
                       np.dot(np.array([[Df_x__, 0, 0], [0, Df_y__, 0], [0, 0, Df_z__]]) \
                       * args[0].faces[k].grad["Pcor"][-1], Sf__)
            args[0].faces[k].value["u"][-1] += pcor_f__[0]
            args[0].faces[k].value["v"][-1] += pcor_f__[1]
            args[0].faces[k].value["w"][-1] += pcor_f__[2]  
        return
    def itersolve(self, *args):
        # GMRES
        # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, u :momentum, v : momentum, w : momentum, current_time : int
        self.calccoef(args[5], args[6], args[7])
        lhs_transient, rhs_transient = super().calctransient(args[0], "Pcor", args[4], args[8])
        for i in range(0, lhs_transient.shape[0]):
            lhs_transient[i][i] = lhs_transient[i][i] / args[1]
        A = lambda x: sparse.linalg.spsolve(lhs_transient, x)
        under_relax_b = np.transpose(np.array([[lhs_transient[i][i] * args[0].cells[i].value["Pcor"][-1] * (1 - args[1]) \
                        for i in range(0, lhs_transient.shape[0])]]))
        b = self.rhs + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = args[2], maxiter = args[3])
        prev_x = np.array([args[0].cells[i].value["Pcor"]][-1] for i in args[0].cells.keys())
        rmsr__ = super().calcrmsr(np.flatten(x), prev_x)
        print("{} [{}]; status: {}, prev. value RMSR = {}".format("Pcor", args[8], exitCode, rmsr__))
        # update value and grad
        self.updatevalue(args[0], "Pcor", np.flatten(x))
        self.calccorrect(args[0], args[5], args[6], args[7])
        return rmsr__

class momentum(linear):
    def __init__(self, *args):
        # args mesh : mesh, axis : int
        super().__init__(args[0])
        self.axis = args[1]
    def calccoef(self, *args):
        # args mesh : mesh, user : user, what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        coor_dict = {0: "u", 1: "v", 2: "w"}
        axes = np.delete(np.array([0, 1, 2]), self.axis)
        neigh__ = args[0].templates.neighbor["fluid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Sf__  = args[0].geoms("Sf", i, k, False)
            dCf__ = args[0].geoms("dCf", i, k, False)
            eCF__ = args[0].geoms("eCF", i, j, False)
            dCF__ = args[0].geoms("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[self.axis] = args[0].faces[k].value[coor_dict[self.axis]][-1]
            v__[axes[0]] = args[0].faces[k].value[coor_dict[axes[0]]][-1]
            v__[axes[1]] = args[0].faces[k].value[coor_dict[axes[1]]][-1]
            Ret__ = args[0].faces[k].prop.rhs[-1] * pow(args[0].faces[k].value["k"][-1], 2) / \
                    (args[0].faces[k].prop.miu[-1] * args[0].faces[k].value["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0].faces[k].grad["u"][-1], args[0].faces[k].grad["v"][-1], \
                          args[0].faces[k].grad["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0].faces[k].value["k"][-1] / args[0].faces[k].value["e"][-1], \
                   args[0].faces[k].prop.alpha[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0].faces[k].prop.rhs[-1] * cmiu__ * args[0].faces[k].value["k"][-1] * ts__
            graditr__  = super().linearitr(args[0], ["grad", coor_dict(self.axis)], i, j, k)
            if prev_row == i:
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad[coor_dict[self.axis]][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0].faces[k].prop.miu[-1] \
                        + miut__)
                bC__ += -args[0].faces[k].value["P"][-1] + (2 * args[0].faces[k].prop.rhs[-1] * \
                         args[0].faces[k].value["k"][-1] / 3) * Sf__[self.axis]
                self.lhs[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                prev_row = i
            else:
                if self.axis == 1:
                    bC__ += args[0].cells[prev_row].prop.rhs[-1] * 9.81 * args[0].cells[prev_row].volume
                self.lhs[prev_row][prev_row] = aC__
                self.rhs[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad[coor_dict[self.axis]][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0].faces[k].prop.miu[-1] \
                        + miut__)
                bC__ += -args[0].faces[k].value["P"][-1] + (2 * args[0].faces[k].prop.rhs[-1] * \
                         args[0].faces[k].value["k"][-1] / 3) * Sf__[self.axis]
                self.lhs[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                prev_row = i
        if self.axis == 1:
            bC__ += args[0].cells[prev_row].prop.rhs[-1] * 9.81 * args[0].cells[prev_row].volume
        self.lhs[prev_row][prev_row] = aC__
        self.rhs[prev_row][0] = bC__
        bound__ = args[0].templates.boundary["fluid"].tocoo()
        for i, j, k in zip(bound__.row, bound__.col, bound__.data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[self.axis] = args[0].faces[k].value[coor_dict[self.axis]][-1]
            v__[axes[0]] = args[0].faces[k].value[coor_dict[axes[0]]][-1]
            v__[axes[1]] = args[0].faces[k].value[coor_dict[axes[1]]][-1]
            for l in args[0].faces[j].bound:
                self.calcbound(args[0], l, i, j, v__, args[1])
        return
    def calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        coor_dict = dict({0: "u", 1: "v", 2: "w"})
        axes = np.delete(np.array([0, 1, 2]), self.axis)
        if "noslip" in args[1]:
            dperp__ = pow(linalg.norm(args[0].cells[args[2]].grad[coor_dict[self.axis]][-1])**2 + \
                      2 * args[0].cells[args[2]].value[coor_dict[self.axis]][-1], 0.5) - \
                      linalg.norm(args[0].cells[args[2]].grad[coor_dict[self.axis]][-1])
            Sf__ = args[0]("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.miu * \
                                             Sf__[self.axis] * (1 - eCf__[self.axis]**2) / dperp__
            self.rhs[args[2]][0] += (args[0].faces[args[3]].prop.miu[-1] * Sf__[self.axis] / dperp__) * \
                                       ((args[0].faces[args[3]].value[coor_dict[self.axis]][-1] * (1 - eCf__[self.axis**2]))
                                       + ((args[0].cells[args[2]].value[axes[0]][-1] - args[0].faces[args[3]].value[axes[0]][-1]) * eCf__[self.axis] * eCf__[axes[0]])
                                       + ((args[0].cells[args[2]].value[axes[1]][-1] - args[0].faces[args[3]].value[axes[1]][-1]) * eCf__[self.axis] * eCf__[axes[1]])) \
                                       - (args[0].faces[args[3]].value["P"][-1] * Sf__[self.axis])
            Re__ = args[0].cells[args[2]].prop.rhs[-1] * np.sqrt(np.sum(np.array([map(lambda x: x^2, args[4])]))) * \
                   pow(args[0].cells[args[2]].volume / args[0].cells[args[2]].prop.miu[-1])
            tau__ = args[4][self.axis] * 8 * args[0].cells[args[2]].prop.rhs[-1] / Re__
            self.rhs[args[2]][0] += -tau__ / (args[0].cells[args[2]].prop.rhs[-1] * 2 * args[0].cells[args[2]].volume / args[0].faces[args[3]].area)
        elif "inlet" in args[1]:
            # specified static pressure and velocity direction
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0].cells[args[2]].grad[coor_dict[self.axis]][-1] - \
                          (np.dot(args[0].cells[args[2]].grad[coor_dict[self.axis]][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0].cells[args[2]].grad[axes[0]][-1] - \
                          (np.dot(args[0].cells[args[2]].grad[axes[0]][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0].cells[args[2]].grad[axes[1]][-1] - \
                          (np.dot(args[0].cells[args[2]].grad[axes[1]][-1], eCf__) * eCf__))
            vin_v0_ = args[0].cells[args[2]].value[coor_dict[self.axis]][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0].cells[args[2]].value[axes[0]][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0].cells[args[2]].value[axes[1]][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([0.00, 0.00, 0.00], dtype = float)
            vin__[self.axis] = vin_v0_
            vin__[axes[0]] = vin_v1_
            vin__[axes[1]] = vin_v2_
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * np.dot(grad_vin_v0_, dCf__) -\
                                        args[5].inits.loc[0, "P"] * Sf__[self.axis]
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0].cells[args[2]].grad[coor_dict[self.axis]][-1] - \
                          (np.dot(args[0].cells[args[2]].grad[coor_dict[self.axis]][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0].cells[args[2]].grad[axes[0]][-1] - \
                          (np.dot(args[0].cells[args[2]].grad[axes[0]][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0].cells[args[2]].grad[axes[1]][-1] - \
                          (np.dot(args[0].cells[args[2]].grad[axes[1]][-1], eCf__) * eCf__))
            vout_v0_ = args[0].cells[args[2]].value[coor_dict[self.axis]][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0].cells[args[2]].value[axes[0]][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0].cells[args[2]].value[axes[1]][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([0.00, 0.00, 0.00], dtype = float)
            vout__[self.axis] = vout_v0_
            vout__[axes[0]] = vout_v1_
            vout__[axes[1]] = vout_v2_
            pout__ = args[0].cells[args[2]].value["P"][-1] + np.dot(args[0].cells[args[2]].grad["P"][-1], dCf__)
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__) * np.dot(grad_vout_v0_, dCf__) -\
                                        pout__ * Sf__[self.axis]
        else:
            pass
        return
    def calcwall(self, *args):
        # args mesh : mesh
        coor_dict = {0: "u", 1: "v", 2: "w"}
        for i in args[0].cells.keys():
            if args[0].cells[i].conj_id >= 0:
                v__ = np.array([0.00, 0.00, 0.00], dtype = float)
                v__[0] = args[0].cells[i].value["u"][-1]
                v__[1] = args[0].cells[i].value["v"][-1]
                v__[2] = args[0].cells[i].value["w"][-1]
                v_val__ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v__)])))
                Sf_wall__ = -args[0].geoms("Sf", i, args[0].cells[i].conj_id, False)
                v_parallel_ = np.cross(np.cross(v__, Sf_wall__), Sf_wall__)
                v_parallel_val_ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v_parallel_)])))
                check = np.dot(v_val__, v_parallel_)
                if check >= 0:
                    theta = math.acos(np.dot(v__, v_parallel_) / (v_val__ * v_parallel_val_))
                else:
                    theta = math.acos(np.dot(v__, -v_parallel_) / (v_val__ * v_parallel_val_))
                v_val__ = v_val__ * math.sin(theta)
                Ret__ = args[0].cells[i].prop.rhs[-1] * pow(args[0].cells[i].value["k"][-1], 2) / \
                        (args[0].cells[i].prop.miu[-1] * args[0].cells[i].value["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                gradCfluid__ = args[0].cells[i].grad[coor_dict[self.axis]][-1]
                dperp__ = (np.sqrt(2 * args[0].cells[i].value[coor_dict[self.axis]][-1]) - 1) * \
                            np.sqrt(gradCfluid__[0]**2 + gradCfluid__[1]**2 + gradCfluid__[2]**2)
                dCplus__ = dperp__ * pow(cmiu__, 0.25) * np.sqrt(args[0].cells[i].value["k"][-1]) * \
                            args[0].cells[i].prop.rhs[-1] / args[0].cells[i].prop.miu[-1]
                dCplus__ = np.max(np.array([dCplus__, 11.06]))
                miutau__ = v_val__ * 0.41 / (np.log(dCplus__) + 5.25)
                dplusv__ = dperp__ * miutau__ * args[0].cells[i].prop.rhs[-1] / \
                            args[0].cells[i].prop.miu[-1]
                vplus__ = np.log(dplusv__) / 0.41 + 5.25
                args[0].cells[i].value[coor_dict[self.axis]][-1] = vplus__
        return
    def itersolve(self, *args):
        # GMRES
        # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user, current_time : int
        # args mesh : mesh, user : user, what : str, time_step : int/double
        coor_dict = dict({0: "u", 1: "v", 2: "w"})
        what = coor_dict[self.axis]
        self.calccoef(args[0], args[5], what, args[4])
        lhs_transient, rhs_transient = super().calctransient(args[0], what, args[4], args[6])
        for i in range(0, lhs_transient.shape[0]):
            lhs_transient[i][i] = lhs_transient[i][i] / args[1]
        A = lambda x: sparse.linalg.spsolve(lhs_transient, x)
        under_relax_b = np.transpose(np.array([[lhs_transient[i][i] * args[0].cells[i].value[what][-1] * (1 - args[1]) \
                        for i in range(0, lhs_transient.shape[0])]]))
        b = self.rhs + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = args[2], maxiter = args[3])
        prev_x = np.array([args[0].cells[i].value[what]][-1] for i in args[0].cells.keys())
        rmsr__ = super().calcrmsr(np.flatten(x), prev_x)
        print("{} [{}]; status: {}, prev. value RMSR = {}".format(what, self.current_time, exitCode, rmsr__))
        # update value and grad
        self.updatevalue(args[0], what, np.flatten(x))
        self.calcwall(args[0])
        return rmsr__

class turb_k(linear):
    def __init__(self, *args):
        super().__init__(args[0])
    def calccoef(self, *args):
        # args mesh : mesh, user : user, what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        neigh__ = args[0].templates.neighbor["fluid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Sf__  = args[0].geoms("Sf", i, k, False)
            dCf__ = args[0].geoms("dCf", i, k, False)
            eCF__ = args[0].geoms("eCF", i, j, False)
            dCF__ = args[0].geoms("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            Ret__ = args[0].faces[k].prop.rhs[-1] * pow(args[0].faces[k].value["k"][-1], 2) / \
                    (args[0].faces[k].prop.miu[-1] * args[0].faces[k].value["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0].faces[k].grad["u"][-1], args[0].faces[k].grad["v"][-1], \
                          args[0].faces[k].grad["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0].faces[k].value["k"][-1] / args[0].faces[k].value["e"][-1], \
                   args[0].faces[k].prop.alpha[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0].faces[k].prop.rhs[-1] * cmiu__ * args[0].faces[k].value["k"][-1] * ts__
            graditr__  = super().linearitr(args[0], ["grad", "k"], i, j, k)
            if prev_row == i:
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["k"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0].faces[k].prop.miu[-1] \
                        + miut__)
                self.lhs[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                prev_row = i
            else:
                gradC_u__ = args[0].cells[prev_row].grad["u"][-1]
                gradC_v__ = args[0].cells[prev_row].grad["v"][-1]
                gradC_w__ = args[0].cells[prev_row].grad["w"][-1]
                phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                          + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
                Ret_C_ = args[0].cells[prev_row].prop.rhs[-1] * pow(args[0].cells[prev_row].value["k"][-1], 2) / \
                        (args[0].cells[prev_row].prop.miu[-1] * args[0].cells[prev_row].value["e"][-1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([args[0].cells[prev_row].grad["u"][-1], args[0].cells[prev_row].grad["v"][-1], \
                            args[0].cells[prev_row].grad["w"][-1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
                ts_C_ = np.min(np.array([args[0].cells[prev_row].value["k"][-1] / args[0].cells[prev_row].value["e"][-1], \
                    args[0].cells[prev_row].prop.alpha[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = args[0].cells[prev_row].prop.rhs[-1] * cmiu_C_ * args[0].cells[prev_row].value["k"][-1] * ts_C_
                bC__ += (miut_C_ * phi_v__ - args[0].cells[prev_row].prop.rhs[-1] * args[0].cells[prev_row].value["e"][-1]) \
                        * args[0].cells[prev_row].volume
                self.lhs[prev_row][prev_row] = aC__
                self.rhs[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["k"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0].faces[k].prop.miu[-1] \
                        + miut__)
                self.lhs[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__)
                prev_row = i
        gradC_u__ = args[0].cells[prev_row].grad["u"][-1]
        gradC_v__ = args[0].cells[prev_row].grad["v"][-1]
        gradC_w__ = args[0].cells[prev_row].grad["w"][-1]
        phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                    + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
        Ret_C_ = args[0].cells[prev_row].prop.rhs[-1] * pow(args[0].cells[prev_row].value["k"][-1], 2) / \
                (args[0].cells[prev_row].prop.miu[-1] * args[0].cells[prev_row].value["e"][-1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([args[0].cells[prev_row].grad["u"][-1], args[0].cells[prev_row].grad["v"][-1], \
                    args[0].cells[prev_row].grad["w"][-1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
        ts_C_ = np.min(np.array([args[0].cells[prev_row].value["k"][-1] / args[0].cells[prev_row].value["e"][-1], \
            args[0].cells[prev_row].prop.alpha[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
        miut_C_ = args[0].cells[prev_row].prop.rhs[-1] * cmiu_C_ * args[0].cells[prev_row].value["k"][-1] * ts_C_
        bC__ += (miut_C_ * phi_v__ - args[0].cells[prev_row].prop.rhs[-1] * args[0].cells[prev_row].value["e"][-1]) \
                * args[0].cells[prev_row].volume
        self.lhs[prev_row][prev_row] = aC__
        self.rhs[prev_row][0] = bC__
        bound__ = args[0].templates.boundary["fluid"].tocoo()
        for i, j, k in zip(bound__.row, bound__.col, bound__.data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            for l in args[0].faces[j].bound:
                self.calcbound(args[0], l, i, j, v__, args[1])
        return
    def calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        if "inlet" in args[1]:
            # specified value; zero gradient at inlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0].cells[args[2]].grad["u"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["u"][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0].cells[args[2]].grad["v"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["v"][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0].cells[args[2]].grad["w"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["w"][-1], eCf__) * eCf__))
            vin_v0_ = args[0].cells[args[2]].value["u"][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0].cells[args[2]].value["v"][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0].cells[args[2]].value["w"][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([vin_v0_, vin_v1_, vin_v2_], dtype = float)
            grad_kin_ = np.dot(args[0].cells[args[2]].grad["k"][-1] - \
                         (np.dot(args[0].cells[args[2]].grad["k"][-1], eCf__) * eCf__))
            kin_ = 0.5 * np.dot(vin__, vin__) * 0.01**2
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * kin_
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * np.dot(grad_kin_, dCf__)
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0].cells[args[2]].grad["u"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["u"][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0].cells[args[2]].grad["v"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["v"][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0].cells[args[2]].grad["w"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["w"][-1], eCf__) * eCf__))
            vout_v0_ = args[0].cells[args[2]].value["u"][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0].cells[args[2]].value["v"][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0].cells[args[2]].value["w"][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([vout_v0_, vout_v1_, vout_v2_], dtype = float)
            grad_kout_ = np.dot(args[0].cells[args[2]].grad["k"][-1] - \
                         (np.dot(args[0].cells[args[2]].grad["k"][-1], eCf__) * eCf__))
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__) * np.dot(grad_kout_, dCf__)
        else:
            pass
        return
    def calcwall(self, *args):
        # args mesh : mesh
        for i in args[0].cells.keys():
            if args[0].cells[i].conj_id >= 0:
                Ret__ = args[0].cells[i].prop.rhs[-1] * pow(args[0].cells[i].value["k"][-1], 2) / \
                        (args[0].cells[i].prop.miu[-1] * args[0].cells[i].value["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                args[0].cells[i].value["k"][-1] = 1 / np.sqrt(cmiu__)
        return
    def itersolve(self, *args):
        # GMRES
        # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user, current_time : int
        # args mesh : mesh, user : user, what : str, time_step : int/double
        self.calccoef(args[0], args[5], "k", args[4])
        lhs_transient, rhs_transient = super().calctransient(args[0], "k", args[4], args[6])
        for i in range(0, lhs_transient.shape[0]):
            lhs_transient[i][i] = lhs_transient[i][i] / args[1]
        A = lambda x: sparse.linalg.spsolve(lhs_transient, x)
        under_relax_b = np.transpose(np.array([[lhs_transient[i][i] * args[0].cells[i].value["k"][-1] * (1 - args[1]) \
                        for i in range(0, lhs_transient.shape[0])]]))
        b = self.rhs + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = args[2], maxiter = args[3])
        prev_x = np.array([args[0].cells[i].value["k"]][-1] for i in args[0].cells.keys())
        rmsr__ = super().calcrmsr(np.flatten(x), prev_x)
        print("{} [{}]; status: {}, prev. value RMSR = {}".format("k", self.current_time, exitCode, rmsr__))
        # update value and grad
        self.updatevalue(args[0], "k", np.flatten(x))
        self.calcwall(args[0])
        return rmsr__

class turb_e(linear):
    def __init__(self, *args):
        super().__init__(args[0])
    def calccoef(self, *args):
        # args mesh : mesh, user : user, what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        neigh__ = args[0].templates.neighbor["fluid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Sf__  = args[0].geoms("Sf", i, k, False)
            dCf__ = args[0].geoms("dCf", i, k, False)
            eCF__ = args[0].geoms("eCF", i, j, False)
            dCF__ = args[0].geoms("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            Ret__ = args[0].faces[k].prop.rhs[-1] * pow(args[0].faces[k].value["k"][-1], 2) / \
                    (args[0].faces[k].prop.miu[-1] * args[0].faces[k].value["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0].faces[k].grad["u"][-1], args[0].faces[k].grad["v"][-1], \
                          args[0].faces[k].grad["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0].faces[k].value["k"][-1] / args[0].faces[k].value["e"][-1], \
                   args[0].faces[k].prop.alpha[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0].faces[k].prop.rhs[-1] * cmiu__ * args[0].faces[k].value["k"][-1] * ts__
            graditr__  = super().linearitr(args[0], ["grad", "e"], i, j, k)
            if prev_row == i:
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__ / 1.3)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["e"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0].faces[k].prop.miu[-1] \
                        + miut__ / 1.3)
                self.lhs[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__ / 1.3)
                prev_row = i
            else:
                gradC_u__ = args[0].cells[prev_row].grad["u"][-1]
                gradC_v__ = args[0].cells[prev_row].grad["v"][-1]
                gradC_w__ = args[0].cells[prev_row].grad["w"][-1]
                phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                          + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
                Ret_C_ = args[0].cells[prev_row].prop.rhs[-1] * pow(args[0].cells[prev_row].value["k"][-1], 2) / \
                        (args[0].cells[prev_row].prop.miu[-1] * args[0].cells[prev_row].value["e"][-1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([args[0].cells[prev_row].grad["u"][-1], args[0].cells[prev_row].grad["v"][-1], \
                            args[0].cells[prev_row].grad["w"][-1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
                ts_C_ = np.min(np.array([args[0].cells[prev_row].value["k"][-1] / args[0].cells[prev_row].value["e"][-1], \
                    args[0].cells[prev_row].prop.alpha[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = args[0].cells[prev_row].prop.rhs[-1] * cmiu_C_ * args[0].cells[prev_row].value["k"][-1] * ts_C_
                ceps2 = 1.92 * (1 - 0.3 * math.exp(-1 * Ret_C_**2)) 
                bC__ += (1.44 * miut_C_ * phi_v__ / ts_C_ - ceps2 * args[0].cells[prev_row].prop.rhs[-1] * args[0].cells[prev_row].value["e"][-1] / ts_C_) \
                        * args[0].cells[prev_row].volume
                self.lhs[prev_row][prev_row] = aC__
                self.rhs[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__ / 1.3)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["e"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0].faces[k].prop.miu[-1] \
                        + miut__ / 1.3)
                self.lhs[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.miu[-1] + miut__ / 1.3)
                prev_row = i
        gradC_u__ = args[0].cells[prev_row].grad["u"][-1]
        gradC_v__ = args[0].cells[prev_row].grad["v"][-1]
        gradC_w__ = args[0].cells[prev_row].grad["w"][-1]
        phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                    + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
        Ret_C_ = args[0].cells[prev_row].prop.rhs[-1] * pow(args[0].cells[prev_row].value["k"][-1], 2) / \
                (args[0].cells[prev_row].prop.miu[-1] * args[0].cells[prev_row].value["e"][-1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([args[0].cells[prev_row].grad["u"][-1], args[0].cells[prev_row].grad["v"][-1], \
                    args[0].cells[prev_row].grad["w"][-1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
        ts_C_ = np.min(np.array([args[0].cells[prev_row].value["k"][-1] / args[0].cells[prev_row].value["e"][-1], \
            args[0].cells[prev_row].prop.alpha[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
        miut_C_ = args[0].cells[prev_row].prop.rhs[-1] * cmiu_C_ * args[0].cells[prev_row].value["k"][-1] * ts_C_
        ceps2 = 1.92 * (1 - 0.3 * math.exp(-1 * Ret_C_**2)) 
        bC__ += (1.44 * miut_C_ * phi_v__ / ts_C_ - ceps2 * args[0].cells[prev_row].prop.rhs[-1] * args[0].cells[prev_row].value["e"][-1] / ts_C_) \
                * args[0].cells[prev_row].volume
        self.lhs[prev_row][prev_row] = aC__
        self.rhs[prev_row][0] = bC__
        bound__ = args[0].templates.boundary["fluid"].tocoo()
        for i, j, k in zip(bound__.row, bound__.col, bound__.data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            for l in args[0].faces[j].bound:
                self.calcbound(args[0], l, i, j, v__, args[1])
        return
    def calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        if "inlet" in args[1]:
            # specified value; zero gradient at inlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0].cells[args[2]].grad["u"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["u"][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0].cells[args[2]].grad["v"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["v"][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0].cells[args[2]].grad["w"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["w"][-1], eCf__) * eCf__))
            vin_v0_ = args[0].cells[args[2]].value["u"][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0].cells[args[2]].value["v"][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0].cells[args[2]].value["w"][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([vin_v0_, vin_v1_, vin_v2_], dtype = float)
            grad_ein_ = np.dot(args[0].cells[args[2]].grad["e"][-1] - \
                         (np.dot(args[0].cells[args[2]].grad["e"][-1], eCf__) * eCf__))
            ein_ = pow(0.5 * np.dot(vin__, vin__) * 0.01**2, 1.5) * 0.09 / (0.1 * args[0].cells[args[2]].volume)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * ein_
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * np.dot(grad_ein_, dCf__)
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0].cells[args[2]].grad["u"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["u"][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0].cells[args[2]].grad["v"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["v"][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0].cells[args[2]].grad["w"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["w"][-1], eCf__) * eCf__))
            vout_v0_ = args[0].cells[args[2]].value["u"][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0].cells[args[2]].value["v"][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0].cells[args[2]].value["w"][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([vout_v0_, vout_v1_, vout_v2_], dtype = float)
            grad_eout_ = np.dot(args[0].cells[args[2]].grad["e"][-1] - \
                         (np.dot(args[0].cells[args[2]].grad["e"][-1], eCf__) * eCf__))
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__) * np.dot(grad_eout_, dCf__)
        else:
            pass
        return
    def calcwall(self, *args):
        # args mesh : mesh
        for i in args[0].cells.keys():
            if args[0].cells[i].conj_id >= 0:
                v__ = np.array([0.00, 0.00, 0.00], dtype = float)
                v__[0] = args[0].cells[i].value["u"][-1]
                v__[1] = args[0].cells[i].value["v"][-1]
                v__[2] = args[0].cells[i].value["w"][-1]
                v_val_ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v__)])))
                Ret__ = args[0].cells[i].prop.rhs[-1] * pow(args[0].cells[i].value["k"][-1], 2) / \
                        (args[0].cells[i].prop.miu[-1] * args[0].cells[i].value["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                gradCfluid__ = args[0].cells[i].grad["T"][-1]
                dperp__ = (np.sqrt(2 * args[0].cells[i].value["e"][-1]) - 1) * \
                            np.sqrt(gradCfluid__[0]**2 + gradCfluid__[1]**2 + gradCfluid__[2]**2)
                dCplus__ = dperp__ * pow(cmiu__, 0.25) * np.sqrt(args[0].cells[i].value["k"][-1]) * \
                            args[0].cells[i].prop.rhs[-1] / args[0].cells[i].prop.miu[-1]
                dCplus__ = np.max(np.array([dCplus__, 11.06]))
                miutau__ = v_val_ * 0.41 / (np.log(dCplus__) + 5.25)
                eplus__ =  args[0].cells[i].prop.miu[-1] / (args[0].cells[i].prop.rhs[-1] * miutau__ * 0.41 * dperp__)
                args[0].cells[i].value["e"][-1] = eplus__
        return
    def itersolve(self, *args):
        # GMRES
        # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user, current_time : int
        # args mesh : mesh, user : user, what : str, time_step : int/double
        self.calccoef(args[0], args[5], "e", args[4])
        lhs_transient, rhs_transient = super().calctransient(args[0], "e", args[4], args[6])
        for i in range(0, lhs_transient.shape[0]):
            lhs_transient[i][i] = lhs_transient[i][i] / args[1]
        A = lambda x: sparse.linalg.spsolve(lhs_transient, x)
        under_relax_b = np.transpose(np.array([[lhs_transient[i][i] * args[0].cells[i].value["e"][-1] * (1 - args[1]) \
                        for i in range(0, lhs_transient.shape[0])]]))
        b = self.rhs + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = args[2], maxiter = args[3])
        prev_x = np.array([args[0].cells[i].value["e"]][-1] for i in args[0].cells.keys())
        rmsr__ = super().calcrmsr(np.flatten(x), prev_x)
        print("{} [{}]; status: {}, prev. value RMSR = {}".format("e", self.current_time, exitCode, rmsr__))
        # update value and grad
        self.updatevalue(args[0], "k", np.flatten(x))
        self.calcwall(args[0])
        return rmsr__

class energy(linear):
    def __init__(self, *args):
        super().__init__(args[0])
    def calccoef(self, *args):
        # args mesh : mesh, user : user, what : str, time_step : int/double
        # fluid
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        neigh__ = args[0].templates.neighbor["fluid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Sf__  = args[0].geoms("Sf", i, k, False)
            dCf__ = args[0].geoms("dCf", i, k, False)
            eCF__ = args[0].geoms("eCF", i, j, False)
            dCF__ = args[0].geoms("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            Ret__ = args[0].faces[k].prop.rhs[-1] * pow(args[0].faces[k].value["k"][-1], 2) / \
                    (args[0].faces[k].prop.miu[-1] * args[0].faces[k].value["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0].faces[k].grad["u"][-1], args[0].faces[k].grad["v"][-1], \
                          args[0].faces[k].grad["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0].faces[k].value["k"][-1] / args[0].faces[k].value["e"][-1], \
                   args[0].faces[k].prop.alpha[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0].faces[k].prop.rhs[-1] * cmiu__ * args[0].faces[k].value["k"][-1] * ts__
            graditr__  = super().linearitr(args[0], ["grad", "T"], i, j, k)
            Pr__ = args[0].faces[k].prop.miu[-1] / (args[0].faces[k].prop.rhs[-1] * args[0].faces[k].prop.alpha[-1])
            if prev_row == i:
                aC__ += args[0].faces[k].prop.cp[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0].faces[k].prop.cp[-1] * (args[0].faces[k].prop.miu[-1] / Pr__ + miut__ / 0.9)
                bC__ += args[0].faces[k].prop.cp[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) *  args[0].faces[k].prop.cp[-1] * \
                        (args[0].faces[k].prop.miu[-1] / Pr__ + miut__ / 0.9)
                self.lhs[i][j] = args[0].faces[k].prop.cp[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * \
                                    args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * args[0].faces[k].prop.cp[-1] * (args[0].faces[k].prop.miu[-1] \
                                      / Pr__ + miut__ / 0.9)
                prev_row = i
            else:
                gradC_u__ = args[0].cells[prev_row].grad["u"][-1]
                gradC_v__ = args[0].cells[prev_row].grad["v"][-1]
                gradC_w__ = args[0].cells[prev_row].grad["w"][-1]
                phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                          + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
                Ret_C_ = args[0].cells[prev_row].prop.rhs[-1] * pow(args[0].cells[prev_row].value["k"][-1], 2) / \
                        (args[0].cells[prev_row].prop.miu[-1] * args[0].cells[prev_row].value["e"][-1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([args[0].cells[prev_row].grad["u"][-1], args[0].cells[prev_row].grad["v"][-1], \
                            args[0].cells[prev_row].grad["w"][-1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
                ts_C_ = np.min(np.array([args[0].cells[prev_row].value["k"][-1] / args[0].cells[prev_row].value["e"][-1], \
                    args[0].cells[prev_row].prop.alpha[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = args[0].cells[prev_row].prop.rhs[-1] * cmiu_C_ * args[0].cells[prev_row].value["k"][-1] * ts_C_
                bC__ += (args[0].cells[prev_row].prop.miu[-1] + miut_C_) * phi_v__ * args[0].cells[prev_row].volume
                self.lhs[prev_row][prev_row] = aC__
                self.rhs[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += args[0].faces[k].prop.cp[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0].faces[k].prop.cp[-1] * (args[0].faces[k].prop.miu[-1] / Pr__ + miut__ / 0.9)
                bC__ += args[0].faces[k].prop.cp[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) *  args[0].faces[k].prop.cp[-1] * \
                        (args[0].faces[k].prop.miu[-1] / Pr__ + miut__ / 0.9)
                self.lhs[i][j] = args[0].faces[k].prop.cp[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * \
                                    args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * args[0].faces[k].prop.cp[-1] * (args[0].faces[k].prop.miu[-1] \
                                      / Pr__ + miut__ / 0.9)
                prev_row = i
        gradC_u__ = args[0].cells[prev_row].grad["u"][-1]
        gradC_v__ = args[0].cells[prev_row].grad["v"][-1]
        gradC_w__ = args[0].cells[prev_row].grad["w"][-1]
        phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                    + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
        Ret_C_ = args[0].cells[prev_row].prop.rhs[-1] * pow(args[0].cells[prev_row].value["k"][-1], 2) / \
                (args[0].cells[prev_row].prop.miu[-1] * args[0].cells[prev_row].value["e"][-1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([args[0].cells[prev_row].grad["u"][-1], args[0].cells[prev_row].grad["v"][-1], \
                    args[0].cells[prev_row].grad["w"][-1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
        ts_C_ = np.min(np.array([args[0].cells[prev_row].value["k"][-1] / args[0].cells[prev_row].value["e"][-1], \
            args[0].cells[prev_row].prop.alpha[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
        miut_C_ = args[0].cells[prev_row].prop.rhs[-1] * cmiu_C_ * args[0].cells[prev_row].value["k"][-1] * ts_C_
        bC__ += (args[0].cells[prev_row].prop.miu[-1] + miut_C_) * phi_v__ * args[0].cells[prev_row].volume
        self.lhs[prev_row][prev_row] = aC__
        self.rhs[prev_row][0] = bC__
        bound__ = args[0].templates.boundary["fluid"].tocoo()
        for i, j, k in zip(bound__.row, bound__.col, bound__.data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            for l in args[0].faces[j].bound:
                self.calcbound(args[0], l, i, j, v__, args[1])
        # solid
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        neigh__ = args[0].templates.neighbor["solid"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            Sf__  = args[0].geoms("Sf", i, k, False)
            dCf__ = args[0].geoms("dCf", i, k, False)
            eCF__ = args[0].geoms("eCF", i, j, False)
            dCF__ = args[0].geoms("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            Ret__ = args[0].faces[k].prop.rhs[-1] * pow(args[0].faces[k].value["k"][-1], 2) / \
                    (args[0].faces[k].prop.miu[-1] * args[0].faces[k].value["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0].faces[k].grad["u"][-1], args[0].faces[k].grad["v"][-1], \
                          args[0].faces[k].grad["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0].faces[k].value["k"][-1] / args[0].faces[k].value["e"][-1], \
                   args[0].faces[k].prop.alpha[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0].faces[k].prop.rhs[-1] * cmiu__ * args[0].faces[k].value["k"][-1] * ts__
            graditr__  = super().linearitr(args[0], ["grad", "T"], i, j, k)
            if prev_row == i:
                aC__ += args[0].faces[k].prop.cp[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0].faces[k].prop.k[-1]
                bC__ += args[0].faces[k].prop.cp[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * args[0].faces[k].prop.k[-1]
                self.lhs[i][j] = args[0].faces[k].prop.cp[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.k[-1])
                prev_row = i
            else:
                for l in args[0].cells[i].domain:
                    if l in args[1].constants.columns:
                        bC__ += args[i].constants.loc[0, l] * args[0].geoms("Sf", prev_row, k, True)
                self.lhs[prev_row][prev_row] = aC__
                self.rhs[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += args[0].faces[k].prop.cp[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0].faces[k].prop.k[-1]
                bC__ += args[0].faces[k].prop.cp[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0].cells[i].grad["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0].faces[k].prop.rhs[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * args[0].faces[k].prop.k[-1]
                self.lhs[i][j] = args[0].faces[k].prop.cp[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0].faces[k].prop.rhs[-1] * \
                                     np.dot(v__, Sf__)
                self.lhs[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0].faces[k].prop.k[-1])
                prev_row = i
        for l in args[0].cells[i].domain:
            if l in args[1].constants.columns:
                bC__ += args[i].constants.loc[0, l] * args[0].geoms("Sf", prev_row, k, True)
        self.lhs[prev_row][prev_row] = aC__
        self.rhs[prev_row][0] = bC__
        bound__ = args[0].templates.boundary["solid"].tocoo()
        for i, j, k in zip(bound__.row, bound__.col, bound__.data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            for l in args[0].faces[j].bound:
                self.calcbound(args[0], l, i, j, v__, args[1])
        # conj
        neigh__ = args[0].templates.neighbor["conj"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            if "fluid" in args[0].cells[i].domain:
                fluid_id = i
                solid_id = j
            else:
                fluid_id = j
                solid_id = j
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0].faces[k].value["u"][-1]
            v__[1] = args[0].faces[k].value["v"][-1]
            v__[2] = args[0].faces[k].value["w"][-1]
            Ret__ = args[0].faces[k].prop.rhs[-1] * pow(args[0].faces[k].value["k"][-1], 2) / \
                    (args[0].faces[k].prop.miu[-1] * args[0].faces[k].value["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            gradCfluid__ = args[0].cells[fluid_id].grad["T"][-1]
            hb__ = args[0].cells[fluid_id].prop.rhs[-1] * args[0].cells[fluid_id].prop.cp[-1] * \
                   pow(cmiu__, 0.25) * np.sqrt(args[0].cells[fluid_id].value["k"][-1]) / args[0].cells[args[2]].value["T"][-1]
            graditr__  = super().linearitr(args[0], ["grad", "T"], i, j, k)
            self.lhs[fluid_id][fluid_id] += hb__ * np.dot(args[0].geoms("eCF", fluid_id, solid_id, False), \
                                               args[0].geoms("dCf", fluid_id, k, False)) * \
                                               args[0].geoms("Sf", fluid_id, k, True) / args[0].geoms("dCF", \
                                               fluid_id, solid_id, True)
            self.lhs[solid_id][solid_id] += hb__ * np.dot(args[0].geoms("eCF", solid_id, fluid_id, False), \
                                               args[0].geoms("dCf", solid_id, k, False)) * \
                                               args[0].geoms("Sf", solid_id, k, True) / args[0].geoms("dCF", \
                                               solid_id, fluid_id, True)
            self.lhs[fluid_id][solid_id] += -hb__ * np.dot(args[0].geoms("eCF", fluid_id, solid_id, False), \
                                               args[0].geoms("dCf", fluid_id, k, False)) * \
                                               args[0].geoms("Sf", fluid_id, k, True) / args[0].geoms("dCF", \
                                               fluid_id, solid_id, True)
            self.lhs[solid_id][fluid_id] += -hb__ * np.dot(args[0].geoms("eCF", solid_id, fluid_id, False), \
                                               args[0].geoms("dCf", solid_id, k, False)) * \
                                               args[0].geoms("Sf", solid_id, k, True) / args[0].geoms("dCF", \
                                               solid_id, fluid_id, True)
            self.rhs[fluid_id][0] += hb__ * np.dot((graditr__ - np.dot(graditr__, args[0].geoms("eCF", \
                                        fluid_id, solid_id, False)) * args[0].geoms("eCF", fluid_id, solid_id, False)), \
                                        args[0].geoms("dCf", fluid_id, k, False))
            self.rhs[solid_id][0] += -hb__ * np.dot((graditr__ - np.dot(graditr__, args[0].geoms("eCF", \
                                        solid_id, fluid_id, False)) * args[0].geoms("eCF", solid_id, fluid_id, False)), \
                                        args[0].geoms("dCf", solid_id, k, False))
        return
    def calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        if "hamb" in args[1]:
            Tamb__ = args[5].constants[0, "Tamb"]
            Tsky__ = 0.0552 * pow(Tamb__, 1.5)
            hsky = 5.67 * pow(10, -8) * args[0].faces[args[3]].prop.eps[-1] * \
                   (args[0].faces[args[3]].value["T"][-1] + Tsky__) * \
                   (args[0].faces[args[3]].value["T"][-1]**2 + Tsky__**2) * \
                   (args[0].faces[args[3]].value["T"][-1] - Tsky__) / \
                   (args[0].faces[args[3]].value["T"][-1] - Tamb__) 
            Tfilm__ = (args[0].faces[args[3]].value["T"][-1] + Tamb__) / 2
            rho_film__ = 1 / HAPropsSI("Vha", "P", args[5].constants.loc[0, "Pamb"], "T", Tfilm__, "W", \
                         args[5].constants.loc[0, "Wamb"])
            miu_film__ = HAPropsSI("mu", "P", args[5].constants.loc[0, "Pamb"], "T", Tfilm__, "W", \
                       args[5].constants.loc[0, "Wamb"])
            alpha_film__ = HAPropsSI("alpha", "P", args[5].constants.loc[0, "Pamb"], "T", Tfilm__, "W", \
                           args[5].constants.loc[0, "Wamb"])
            RaL__ = 9.81 * (args[0].faces[args[3]].value["T"][-1] - Tamb__) * \
                    np.sqrt(args[0].faces[args[3]].area)**3 * rho_film__ / \
                    (Tfilm__ * miu_film__ * alpha_film__)   
            if RaL__ <= pow(10, 7):
                Nu_N__ = 0.54 * pow(RaL__, 0.25)
            else:
                Nu_N__ = 0.15 * pow(RaL__, 1/3)
            hconv = Nu_N__ * args[0].faces[args[3]].prop.k[-1] / np.sqrt(args[0].faces[args[3]].area)         
            self.rhs[args[2]][0] += -(hsky + hconv) * (args[0].faces[args[3]].value["T"][-1] - Tamb__) \
                                             * args[0].geoms("Sf", args[2], args[3], True)
        elif "s2s" in args[1]:
            # von Neumann
            self.rhs[args[2]][0] += -args[0].clusts[int(args[1][-1])].value["q"][-1] * \
                                       args[0].geoms("Sf", args[2], args[3], True)
        elif "inlet" in args[1]:
            # specified value; zero gradient at inlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0].cells[args[2]].grad["u"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["u"][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0].cells[args[2]].grad["v"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["v"][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0].cells[args[2]].grad["w"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["w"][-1], eCf__) * eCf__))
            vin_v0_ = args[0].cells[args[2]].value["u"][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0].cells[args[2]].value["v"][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0].cells[args[2]].value["w"][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([vin_v0_, vin_v1_, vin_v2_], dtype = float)
            grad_Tin_ = np.dot(args[0].cells[args[2]].grad["T"][-1] - \
                         (np.dot(args[0].cells[args[2]].grad["T"][-1], eCf__) * eCf__))
            Tin_ = pow(0.5 * np.dot(vin__, vin__) * 0.01**2, 1.5) * 0.09 / (0.1 * args[0].cells[args[2]].volume)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * Tin_
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vin__, Sf__) * np.dot(grad_Tin_, dCf__)
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0].geoms("Sf", args[2], args[3], False)
            eCf__ = args[0].geoms("eCf", args[2], args[3], False)
            dCf__ = args[0].geoms("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0].cells[args[2]].grad["u"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["u"][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0].cells[args[2]].grad["v"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["v"][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0].cells[args[2]].grad["w"][-1] - \
                          (np.dot(args[0].cells[args[2]].grad["w"][-1], eCf__) * eCf__))
            vout_v0_ = args[0].cells[args[2]].value["u"][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0].cells[args[2]].value["v"][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0].cells[args[2]].value["w"][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([vout_v0_, vout_v1_, vout_v2_], dtype = float)
            grad_Tout_ = np.dot(args[0].cells[args[2]].grad["e"][-1] - \
                         (np.dot(args[0].cells[args[2]].grad["e"][-1], eCf__) * eCf__))
            self.lhs[args[2]][args[2]] += args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__)
            self.rhs[args[2]][0] += -args[0].faces[args[3]].prop.rhs[-1] * np.dot(vout__, Sf__) * np.dot(grad_Tout_, dCf__)
        else:
            pass
        return
    def calcwall(self, *args):
        # args mesh : mesh
        for i in args[0].cells.keys():
            if args[0].cells[i].conj_id >= 0:
                v__ = np.array([0.00, 0.00, 0.00], dtype = float)
                v__[0] = args[0].cells[i].value["u"][-1]
                v__[1] = args[0].cells[i].value["v"][-1]
                v__[2] = args[0].cells[i].value["w"][-1]
                v_val_ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v__)])))
                Ret__ = args[0].cells[i].prop.rhs[-1] * pow(args[0].cells[i].value["k"][-1], 2) / \
                        (args[0].cells[i].prop.miu[-1] * args[0].cells[i].value["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                gradCfluid__ = args[0].cells[i].grad["T"][-1]
                dperp__ = (np.sqrt(2 * args[0].cells[i].value["T"][-1]) - 1) * \
                            np.sqrt(gradCfluid__[0]**2 + gradCfluid__[1]**2 + gradCfluid__[2]**2)
                dCplus__ = dperp__ * pow(cmiu__, 0.25) * np.sqrt(args[0].cells[i].value["k"][-1]) * \
                            args[0].cells[i].prop.rhs[-1] / args[0].cells[i].prop.miu[-1]
                dCplus__ = np.max(np.array([dCplus__, 11.06]))
                miutau__ = v_val_ * 0.41 / (np.log(dCplus__) + 5.25)
                dplusT__ = dperp__ * miutau__ * args[0].cells[i].prop.rhs[-1] / \
                            args[0].cells[i].prop.miu[-1]
                Pr__ = args[0].cells[i].prop.miu[-1] / (args[0].cells[i].prop.rhs[-1] \
                        * args[0].cells[i].prop.alpha[-1])
                beta__ = pow(3.85 * pow(Pr__, 1/3) - 1.3, 2) + 2.12 * np.log(Pr__)
                Tplus__ = 2.12 * np.log(dplusT__) + beta__ * Pr__
                args[0].cells[i].value["T"][-1] = Tplus__
        return
    def itersolve(self, *args):
        # GMRES
        # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user, current_time : int
        # args mesh : mesh, user : user, what : str, time_step : int/double
        self.calccoef(args[0], args[5], "T", args[4])
        lhs_transient, rhs_transient = super().calctransient(args[0], "T", args[4], args[6])
        for i in range(0, lhs_transient.shape[0]):
            lhs_transient[i][i] = lhs_transient[i][i] / args[1]
        A = lambda x: sparse.linalg.spsolve(lhs_transient, x)
        under_relax_b = np.transpose(np.array([[lhs_transient[i][i] * args[0].cells[i].value["T"][-1] * (1 - args[1]) \
                        for i in range(0, lhs_transient.shape[0])]]))
        b = self.rhs + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = args[2], maxiter = args[3])
        prev_x = np.array([args[0].cells[i].value["T"]][-1] for i in args[0].cells.keys())
        rmsr__ = super().calcrmsr(np.flatten(x), prev_x)
        print("{} [{}]; status: {}, prev. value RMSR = {}".format("T", self.current_time, exitCode, rmsr__))
        # update value and grad
        self.updatevalue(args[0], "T", np.flatten(x))
        self.calcwall(args[0])
        for i in args[0].cells.keys():
            if "rho" in dir(args[0].cells[i].prop):
                args[0].cells[i].prop.updateprop(args[0].cells[i].value["P"][-1], args[0].cells[i].value["T"][-1], args[5].constants.loc[0, "Wamb"])
        for i in args[0].faces.keys():
            if "rho" in dir(args[0].faces[i].prop):
                args[0].faces[i].prop.updateprop(args[0].faces[i].value["P"][-1], args[0].faces[i].value["T"][-1], args[5].constants.loc[0, "Wamb"])
        return rmsr__

class s2s(linear):
    def __init__(self, *args):
        super().__init__(args[0], what = "s2s")
    def calccoef(self, *args):
        # args mesh : mesh, time_step : int/double
        # s2s only
        prev_row = 0
        neigh__ = args[0].templates.neighbor["s2s"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
            if prev_row == i:
                Tclust_C__ = 0.00
                rho_clust_F__ = 0.00
                area_clust_C__ = 0.00
                area_clust_F__ = 0.00
                eps_clust_C__ = 0.0
                for l in args[0].clusts[i].member:
                    Tclust_C__ += args[0].faces[l].value["T"][-1] * args[0].faces[l].area
                    area_clust_C__ +=  args[0].faces[l].area
                    eps_clust_C__ = args[0].faces[l].prop.eps[-1]
                for l in args[0].clusts[j].member:
                    rho_clust_F__ += args[0].faces[l].prop.rhs[-1] * args[0].faces[l].area
                    area_clust_F__ += args[0].faces[l].area
                Tclust_C__ = Tclust_C__ / area_clust_C__
                rho_clust_F__ = rho_clust_F__ / area_clust_F__
                self.lhs[i][j] = rho_clust_F__ * args[0].geoms("view", i, j)
                prev_row = i
            else:
                self.lhs[prev_row][prev_row] = 1
                self.rhs[prev_row][0] = eps_clust_C__ * 5.67 * pow(10, -8) * pow(Tclust_C__, 4)
                Tclust_C__ = 0.00
                rho_clust_F__ = 0.00
                area_clust_C__ = 0.00
                area_clust_F__ = 0.00
                eps_clust_C__ = 0.0
                for l in args[0].clusts[i].member:
                    Tclust_C__ += args[0].faces[l].value["T"][-1] * args[0].faces[l].area
                    area_clust_C__ +=  args[0].faces[l].area
                    eps_clust_C__ = args[0].faces[l].prop.eps[-1]
                for l in args[0].clusts[j].member:
                    rho_clust_F__ += args[0].faces[l].prop.rhs[-1] * args[0].faces[l].area
                    area_clust_F__ += args[0].faces[l].area
                Tclust_C__ = Tclust_C__ / area_clust_C__
                rho_clust_F__ = rho_clust_F__ / area_clust_F__
                self.lhs[i][j] = rho_clust_F__ * args[0].geoms("view", i, j)
                prev_row = i
        self.lhs[prev_row][prev_row] = 1
        self.rhs[prev_row][0] = eps_clust_C__ * 5.67 * pow(10, -8) * pow(Tclust_C__, 4)
        return
    def itersolve(self, *args):
        # GMRES
        # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user
        # args mesh : mesh, user : user, what : str, time_step : int/double
        self.calccoef(args[0], args[5], "q", args[4])
        lhs__ = self.lhs; rhs__ = self.rhs
        for i in range(0, lhs__.shape[0]):
            self.lhs__[i][i] = lhs__[i][i] / args[1]
        A = lambda x: sparse.linalg.spsolve(lhs__, x)
        under_relax_b = np.transpose(np.array([[lhs__[i][i] * args[0].cells[i].value["q"][-1] * (1 - args[1]) \
                        for i in range(0, lhs__.shape[0])]]))
        b = rhs__ + under_relax_b
        x, exitCode = sparse.linalg.gmres(A, b, tol = args[2], maxiter = args[3])
        prev_x = np.array([args[0].cells[i].value["q"]][-1] for i in args[0].cells.keys())
        rmsr__ = super().calcrmsr(np.flatten(x), prev_x)
        print("{} [{}]; status: {}, prev. value RMSR = {}".format("q", self.current_time, exitCode, rmsr__))
        # update value and grad
        self.updatevalue(args[0], "q", np.flatten(x))
        return rmsr__