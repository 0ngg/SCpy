import numpy as np; import pandas as pd
import math; import os
import meshio
from CoolProp.HumidAirProp import HAPropsSI
import scipy.sparse as sparse
from decimal import *
import itertools
from copy import deepcopy

# cfd scheme
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
class user:
    def __init__(self, *args):
        # args init_value : str, solid_props : str, const_value : str
        dirname = os.getcwd()
        self.inits = pd.read_csv(dirname + "\\case\\test\\csv\\" + args[0])
        self.solid_props = pd.read_csv(dirname + "\\case\\test\\csv" + args[1], index_col = 0)
        self.constants = pd.read_csv(dirname + "\\problem\\test\\csv" + args[2])
class face:
    def __init__(self, points_dict : dict, *args):
        # args new_nodes : list, new_bound : list
        self.__nodes = args[0]
        self.__centroid, self.__area = self.get_info(points_dict)
        self.__boundary = args[1]
    @property
    def nodes(self):
        pass
    @nodes.getter
    def nodes(self):
        return self.__nodes
    @nodes.deleter
    def nodes(self):
        del self.__nodes
    @property
    def area(self):
        pass
    @area.getter
    def area(self):
        return self.__area
    @area.deleter
    def area(self):
        del self.__area
    @property
    def centroid(self):
        pass
    @centroid.getter
    def centroid(self):
        return self.__centroid
    @centroid.deleter
    def centroid(self):
        del self.__centroid
    @property
    def boundary(self):
        pass
    @boundary.getter
    def boundary(self):
        return self.__boundary
    @boundary.deleter
    def boundary(self):
        del self.__boundary
    
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
        pass
    @nodes.getter
    def nodes(self):
        return self.__nodes
    @nodes.deleter
    def nodes(self):
        del self.__nodes
    @property
    def faces(self):
        pass
    @faces.getter
    def faces(self):
        return self.__faces
    @faces.deleter
    def faces(self):
        del self.__faces
    @property
    def volume(self):
        pass
    @volume.getter
    def volume(self):
        return self.__volume
    @volume.deleter
    def volume(self):
        del self.__volume
    @property
    def centroid(self):
        pass
    @centroid.getter
    def centroid(self):
        return self.__centroid
    @centroid.deleter
    def centroid(self):
        del self.__centroid
    @property
    def domain(self):
        pass
    @domain.getter
    def domain(self):
        return self.__domain
    @domain.deleter
    def domain(self):
        del self.__domain

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
        pass
    @faces.getter
    def faces(self):
        return self.__faces
    @faces.deleter
    def faces(self):
        del self.__faces
    @property
    def area(self):
        pass
    @area.getter
    def area(self):
        return self.__area
    @area.deleter
    def area(self):
        del self.__area
    @property
    def centroid(self):
        pass
    @centroid.getter
    def centroid(self):
        return self.__centroid
    @centroid.deleter
    def centroid(self):
        del self.__centroid
        
    def get_info(self, face_dict : dict):
        # centroid calc
        centroid = np.array([Decimal(0), Decimal(0), Decimal(0)])
        for it1 in self.faces:
            centroid += face_dict[it1].centroid * face_dict[it1].area
        centroid = centroid / self.__area
        centroid = np.array([round(it1, 3) for it1 in centroid])
        return centroid
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
    def print_elements(self):
        print("faces: {}\narea: {}\ncentroid: {}".format(self.faces, self.area, self.centroid))
        return
class connect:
    def __init__(self, cc_args : dict, fc_args : dict):
        self.__cc = cc_args
        self.__fc = fc_args
    @classmethod
    def copy(self, other):
        self.__cc = other.__cc
        self.__fc = other.__fc
        return self
    @property
    def cc(self):
        pass
    @cc.getter
    def cc(self):
        return deepcopy(self.__cc)
    @cc.deleter
    def cc(self):
        del self.__cc
    @property
    def fc(self):
        pass
    @fc.getter
    def fc(self):
        return deepcopy(self.__fc)
    @fc.deleter
    def fc(self):
        del self.__fc

    def get_coo(self, var : str, which : str):
        check = str("_connect__" + var)
        if check in dir(self):
            tosparse = np.transpose(np.array(self.__dict__[check][which]))
            return sparse.coo_matrix((tosparse[2], (tosparse[0], tosparse[1])))
        else:
            print("Variable not found")
        return
    def get_csr(self, var : str, which : str):
        check = str("_connect__" + var)
        if check in dir(self):
            tosparse = np.transpose(np.array(self.__dict__[check][which]))
            return sparse.csr_matrix((tosparse[2], (tosparse[0], tosparse[1])))
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
                Sf_val = np.sqrt(np.sum(np.array([it4**2 for it4 in Sf])))
                Ef = Sf / Sf_val; Tf = Sf - Ef
                dCf = np.array([round(Decimal(it4), 3) for it4 in face_dict[it3].centroid - it2.centroid])
                dCf_val = np.sqrt(np.sum(np.array([it4**2 for it4 in dCf])))
                eCf = dCf / dCf_val
                Sf_args[0].append([it1, it3, Sf[0]]); Sf_args[1].append([it1, it3, Sf[1]]); Sf_args[2].append([it1, it3, Sf[2]])
                Ef_args[0].append([it1, it3, Ef[0]]); Ef_args[1].append([it1, it3, Ef[1]]); Ef_args[2].append([it1, it3, Ef[2]])
                Tf_args[0].append([it1, it3, Tf[0]]); Tf_args[1].append([it1, it3, Tf[1]]); Tf_args[2].append([it1, it3, Tf[2]])
                dCf_args[0].append([it1, it3, dCf[0]]); dCf_args[1].append([it1, it3, dCf[1]]); dCf_args[2].append([it1, it3, dCf[2]])
                eCf_args[0].append([it1, it3, eCf[0]]); eCf_args[1].append([it1, it3, eCf[1]]); eCf_args[2].append([it1, it3, eCf[2]])
        for it1 in neigh_args:
            dCF = np.array([round(Decimal(it2), 3) for it2 in cell_dict[it1[1]].centroid - cell_dict[it1[0]].centroid])
            dCF_val = np.sqrt(np.sum(np.array([it4**2 for it4 in dCf])))
            eCF = dCF / dCF_val
            dCF_args[0].append([it1[0], it1[1], dCF[0]]); dCF_args[1].append([it1[0], it1[1], dCF[1]]); dCF_args[2].append([it1[0], it1[1], dCF[2]])
            eCF_args[0].append([it1[0], it1[1], eCF[0]]); eCF_args[1].append([it1[0], it1[1], eCF[1]]); eCF_args[2].append([it1[0], it1[1], eCF[2]])
        self.__Sf = np.array(Sf_args); self.__Ef = np.array(Ef_args); self.__Tf = np.array(Tf_args)
        self.__dCf = np.array(dCf_args); self.__eCf = np.array(eCf_args); self.__dCF = np.array(dCF_args); self.__eCF = np.array(eCF_args)
    @property
    def Sf(self):
        pass
    @Sf.getter
    def Sf(self):
        return self.__Sf
    @Sf.deleter
    def Sf(self):
        del self.__Sf
    @property
    def Ef(self):
        pass
    @Ef.getter
    def Ef(self):
        return self.__Ef
    @Ef.deleter
    def Ef(self):
        del self.__Ef
    @property
    def Tf(self):
        pass
    @Tf.getter
    def Tf(self):
        return self.__Tf
    @Tf.deleter
    def Tf(self):
        del self.__Tf
    @property
    def dCf(self):
        pass
    @dCf.getter
    def dCf(self):
        return self.__dCf
    @dCf.deleter
    def dCf(self):
        del self.__dCf
    @property
    def eCf(self):
        pass
    @eCf.getter
    def eCf(self):
        return self.__eCf
    @eCf.deleter
    def eCf(self):
        del self.__eCf
    @property
    def dCF(self):
        pass
    @dCF.getter
    def dCF(self):
        return self.__dCF
    @dCF.deleter
    def dCF(self):
        del self.__dCF
    @property
    def eCF(self):
        pass
    @eCF.getter
    def eCF(self):
        return self.__eCF
    @eCF.deleter
    def eCF(self):
        del self.__eCF

    def get(self, row : int, col : int, is_unit : bool):
        coor = [Decimal(0), Decimal(0), Decimal(0)]
        for it1 in range(0, self._geom__Sf[0].shape[0]):
            if all([self._geom__Sf[0][it1][0] == row, self._geom__Sf[0][it1][1] == col]) is True:
                coor = np.array([self._geom__Sf[it2][it1][2] for it2 in [0,1,2]])
        if is_unit is True:
            return round(Decimal(np.sqrt(np.sum(np.array([it1**2 for it1 in coor])))), 3)
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
        for it1 in bound_args:
            which_fc = [it2 for it2 in ["fluid", "solid"] if it2 in cell_dict[it1[0]].domain[0]]
            which_fc = which_fc[0]
            if which_fc not in list(fc_dict.keys()):
                fc_dict[which_fc] = [it1]
            else:
                fc_dict[which_fc].append(it1)
        print(cc_dict.keys())
        if len(list(clust_args.keys())) > 0:
            cc_dict["s2s"] = []
            for it1 in range(0, len(list(clust_dict.keys())) - 1):
                for it2 in range(it1 + 1, len(list(clust_dict.keys()))):
                    view1, view2 = clust.match_view(points_dict, face_dict, clust_dict[list(clust_dict.keys())[it1]], clust_dict[list(clust_dict.keys())[it2]])
                    cc_dict["s2s"].append([it1, it2, view1])
                    cc_dict["s2s"].append([it2, it1, view2])
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
        pass
    @nodes.getter
    def nodes(self):
        return self.__nodes
    @nodes.deleter
    def nodes(self):
        del self.__nodes
    @property
    def faces(self):
        pass
    @faces.getter
    def faces(self):
        return self.__faces
    @faces.deleter
    def faces(self):
        del self.__faces
    @property
    def cells(self):
        pass
    @cells.getter
    def cells(self):
        return self.__cells
    @cells.deleter
    def cells(self):
        del self.__cells
    @property
    def clusts(self):
        pass
    @clusts.getter
    def clusts(self):
        return self.__clusts
    @clusts.deleter
    def clusts(self):
        del self.__clusts
    @property
    def templates(self):
        pass
    @templates.getter
    def templates(self):
        return self.__templates
    @templates.deleter
    def templates(self):
        del self.__templates
    @property
    def geoms(self):
        pass
    @geoms.getter
    def geoms(self):
        return self.__geoms
    @geoms.deleter
    def geoms(self):
        del self.__geoms
    

# test element
a = mesh("sc.msh")

for i, j in a.items():
    print("face {}".format(i)); j.print_elements(); print("\n")
for i, j in b.items():
    print("cell {}".format(i)); j.print_elements(); print("\n")
for i, j in c.items():
    print("clust {}".format(i)); j.print_elements(); print("\n")
print("connect \n"); d.print_elements(); print("\n")
print("geom \n"); e.print_elements(); print("\n")

