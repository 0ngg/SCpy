"""
import compileall; import os
compileall.compile_dir('C:\\Users\\SolVer\\Documents\\TA\\Program\\SCpy\\lib\\cfd')
"""

import numpy as np
import pandas as pd
import math
from parse_gmsh import parse
import scipy.sparse as sparse
import scipy.spatial as sp
import scipy.linalg as linalg
from copy import deepcopy
from CoolProp.HumidAirProp import HAPropsSI
from collections import Counter
from pyamg import smoothed_aggregation_solver
import pickle
from time import process_time
import os
np.seterr(all='raise')

pcor_relax = 3

"""
----
INFO
"""

class Holder:
    def __init__(self, tag_: int, type_: int = None, loc_: np.ndarray = None,
                 size_: float = None, lnode_: list = None, lcurve_: list = None, lface_: list = None, layer_id: int = -1, depth_id: int = -1):
        self.__tag = tag_
        self.__type = type_
        self.__group = []
        self.__loc = loc_
        self.__size = size_
        self.__wdist = 0
        self.__lnode = lnode_
        self.__lcurve = lcurve_
        self.__lface = lface_
        self.__connect = {}
        self.__bound = []
        self.__conj = {} # only append when connect is conj
        self.__layer = layer_id
        self.__depth = depth_id

    @property
    def tag(self):
        return self.__tag
    @property
    def type(self):
        return self.__type
    @property
    def group(self):
        return self.__group
    @group.setter
    def group(self, group_):
        self.__group = group_
    @property
    def loc(self):
        return self.__loc
    @loc.setter
    def loc(self, loc_: np.ndarray):
        self.__loc = loc_
    @property
    def size(self):
        return self.__size
    @size.setter
    def size(self, size_: float):
        self.__size = size_
    @property
    def wdist(self):
        return self.__wdist
    @wdist.setter
    def wdist(self, wdist_: float):
        self.__wdist = wdist_
    @property
    def lnode(self):
        return self.__lnode
    @lnode.setter
    def lnode(self, lnode_: list):
        self.__lnode = lnode_
    @property
    def lcurve(self):
        return self.__lcurve
    @lcurve.setter
    def lcurve(self, lcurve_: list):
        self.__lcurve = lcurve_
    @property
    def lface(self):
        return self.__lface
    @lface.setter
    def lface(self, lface_: list):
        self.__lface = lface_
    @property
    def connect(self):
        return self.__connect
    @connect.setter
    def connect(self, connect_: dict):
        self.__connect = connect_
    @property
    def bound(self):
        return self.__bound
    @bound.setter
    def bound(self, bound_: list):
        self.__bound = bound_
    @property
    def conj(self):
        return self.__conj
    @conj.setter
    def conj(self, conj_: dict):
        self.__conj = conj_
    @property
    def layer(self):
        return self.__layer
    @layer.setter
    def layer(self, newlayer_: int):
        self.__layer = newlayer_
    @property
    def depth(self):
        return self.__depth
    @depth.setter
    def depth(self, newdepth_: int):
        self.__depth = newdepth_

class Layer:
    def __init__(self, nodes_: dict, curves_: dict, faces_: dict):
        self.__nodes = nodes_
        self.__curves = curves_
        self.__faces = faces_
    
    @classmethod
    def extrude(cls, other_, offset_):
        offset_nodes = deepcopy(other_.nodes)
        for it1, it2 in offset_nodes.items():
            it2.loc += offset_
        return cls(offset_nodes, deepcopy(other_.curves), deepcopy(other_.faces))

    @property
    def nodes(self):
        return self.__nodes
    @property
    def curves(self):
        return self.__curves
    @property
    def faces(self):
        return self.__faces

class Info:
    def impute_tri_curve(id_: int, face_: Holder, lcurve_: list, nodes_: dict, curves_: dict):
        curve_lnode = [[face_.lnode[0], face_.lnode[1]], [face_.lnode[0], face_.lnode[2]], [face_.lnode[1], face_.lnode[2]]]
        for it1 in range(0, len(curve_lnode)):
            curve_lnode[it1].sort(reverse = False)
        [curve_lnode.remove(it1) for it1 in curve_lnode if any([all([it2 in curves_[it3].lnode for it2 in it1]) for it3 in lcurve_])]
        for it1, it2 in curves_.items():
            if any([all([it3 in it2.lnode for it3 in it4]) for it4 in curve_lnode]):
                lcurve_.append(it1); curve_lnode.remove(curves_[it1].lnode)
            elif len(curve_lnode) == 0:
                break
        if len(curve_lnode) > 0:
            for it1 in curve_lnode:
                curves_[len(list(curves_.keys()))] = Holder(len(list(curves_.keys())), lnode_ = it1)
            lcurve_.append(list(curves_.keys())[-1])
        lcurve_ = list(np.unique(lcurve_)); lcurve_.sort(reverse = False)
        face_.lcurve = lcurve_

    def impute_quad_curve(id_: int, face_: Holder, lcurve_: list, nodes_: dict, curves_: dict):
        diag1 = []
        dist = 0.0
        fnode = deepcopy(face_.lnode); fnode.sort(reverse = False)
        for it1 in range(0, len(fnode)):
            for it2 in range(it1, len(fnode)):
                check = np.sqrt(np.sum([it3**2 for it3 in nodes_[fnode[it1]].loc - nodes_[fnode[it2]].loc]))
                if check > dist:
                    dist = check; diag1 = [fnode[it1], fnode[it2]]
        diag2 = [it1 for it1 in fnode if it1 not in diag1]
        curve_lnode = []; [curve_lnode.append([diag1[it1], diag2[it2]]) for it1, it2 in zip([0,0,1,1], [0,1,0,1])]
        [curve_lnode[it1].sort(reverse = False) for it1 in range(0, len(curve_lnode))]

        curve_lnode_fucku = deepcopy(curve_lnode)

        for it1 in lcurve_:
            check = []
            for it2 in curve_lnode:
                check.append(all([it3 in it2 for it3 in curves_[it1].lnode]))
            if any(check) is False:
                lcurve_.remove(it1)

        for it1 in lcurve_:
            check = []
            for it2 in curve_lnode:
                check.append(all([it3 in it2 for it3 in curves_[it1].lnode]))
            if all(check) is True:
                curve_lnode.remove(curves_[it1].lnode)

        for it1, it2 in curves_.items():
            if any([all([it3 in it2.lnode for it3 in it4]) for it4 in curve_lnode]):
                lcurve_.append(it1); curve_lnode.remove(curves_[it1].lnode)
            elif len(curve_lnode) == 0:
                break

        if len(curve_lnode) > 0:
            for it1 in curve_lnode:
                curves_[len(list(curves_.keys()))] = Holder(len(list(curves_.keys())), lnode_ = it1)
            lcurve_.append(list(curves_.keys())[-1])
        lcurve_ = list(np.unique(lcurve_)); lcurve_.sort(reverse = False)

        for it1 in lcurve_:
            check = []
            for it2 in curve_lnode_fucku:
                check.append(all([it3 in it2 for it3 in curves_[it1].lnode]))
            if any(check) is False:
                lcurve_.remove(it1)

        face_.lcurve = lcurve_; 

    def impute_layer(fileloc: str, extrude_len: float, extrude_part: int):
        """
        --------------
        2D layer building
        """
        mesh = parse(fileloc)
        nodes_ = {}; curves_ = {}; faces_ = {}
        # node
        ctd_node = 0
        for entity_ in mesh.get_node_entities():
            for node_ in entity_.get_nodes():
                nodes_[ctd_node] = Holder(node_.get_tag(), loc_ = np.array([x for x in node_.get_coordinates()]))
                ctd_node += 1
        # curve and face
        ctd_curve = 0; ctd_face = 0
        for entity_ in mesh.get_element_entities():
            group_ = entity_.get_group(); group_ = [it1.split('"')[1] for it1 in group_]
            #type_ = entity_.get_element_type()
            for element_ in entity_.get_elements():
                lnode_ = [it1 - 1 for it1 in element_.get_connectivity()]
                lnode_.sort(reverse = False)
                if entity_.get_dimension() == 1:
                    curves_[ctd_curve] = Holder(element_.get_tag(), type_ = type, lnode_ = lnode_)
                    curves_[ctd_curve].group = group_
                    ctd_curve += 1
                elif entity_.get_dimension() == 2:
                    faces_[ctd_face] = Holder(element_.get_tag(), type_ = type, lnode_ = lnode_)
                    faces_[ctd_face].group = group_
                    ctd_face += 1
        match_curve_func = {3: Info.impute_tri_curve, 4: Info.impute_quad_curve}
        rep_match_curve = 0        
        while rep_match_curve < 2:
            for it1, it2 in faces_.items():
                lcurve_ = [it3 for it3, it4 in curves_.items() if all([it5 in it2.lnode for it5 in it4.lnode])]
                match_curve_func[len(it2.lnode)](it1, it2, lcurve_, nodes_, curves_)
            rep_match_curve += 1
        
        """
        -------------
        Extrude layer
        """
        len_ = lambda x: np.sqrt(np.sum([it1**2 for it1 in x]))
        L0 = Layer(nodes_, curves_, faces_)
        guide_ = np.array([0,1,0], dtype = float)
        # v1_ = np.array([round(x, 3) for x in nodes_[faces_[0].lnode[1]].loc - nodes_[faces_[0].lnode[0]].loc])
        # v2_ = np.array([round(x, 3) for x in nodes_[faces_[0].lnode[2]].loc - nodes_[faces_[0].lnode[0]].loc])
        # Sf_ = np.array([round(x, 3) for x in np.cross(v1_, v2_)]); Sf_val = round(len_(Sf_), 3)
        # n_ = np.array([round(x, 3) for x in Sf_ / Sf_val])
        # if np.dot(n_, guide_) < 0:
        #     n_ = -1 * n_
        """
        avg_len_ = [len_(nodes_[it2.lnode[0]].loc - nodes_[it2.lnode[1]].loc) for it1, it2 in curves_.items()]
        avg_len_ = np.sum(avg_len_) / len(avg_len_)

        guide_part = int(np.floor(extrude_len / avg_len_))
        """
        guide_part = extrude_part + 1; avg_len_ = extrude_len / extrude_part
        layers_ = [L0]; current_length = 1
        for it1 in list(range(1, guide_part)):
            current_length += 1
            layer_tmp = Layer.extrude(L0, guide_ * it1 * avg_len_)
            layers_.append(layer_tmp)
        for it1, it2 in layers_[0].faces.items():
            it2.group.append('inlet')
        for it1, it2 in layers_[-1].faces.items():
            it2.group.append('outlet')

        return layers_, guide_part

    def impute_missing(L0: Layer, L1: Layer):
        len_face = len(list(L0.faces.keys())); len_node = len(list(L0.nodes.keys()))
        face_holder = list(L0.faces.values()); face_holder.extend(list(L1.faces.values()))
        faces__ = dict(zip(range(0, 2 * len(list(L0.faces.keys()))), face_holder))

        # faces to new faces and cells
        new_faces = {}; new_cells = {}
        for it1, it2 in L0.faces.items():
            lnode_ = []; lface_ = [it1, it1 + len_face]
            for it3, it4 in zip(it2.lcurve, [L0.curves[it5] for it5 in it2.lcurve]):
                lnode_tmp = list(np.array([[it5, it5 + len_node] for it5 in L1.curves[it3].lnode]).ravel())
                lnode_tmp.sort(reverse = False)
                lnode_.extend([it5 for it5 in lnode_tmp if it5 not in lnode_])
                tag_ = list(faces__.keys())[-1]
                group_ = it4.group
                if (it3 + 2 * len_face) not in list(new_faces.keys()):
                    new_faces[it3 + 2 * len_face] = Holder(tag_ = tag_, lnode_ = lnode_tmp); new_faces[it3 + 2 * len_face].group = group_
                lface_.append(it3 + 2 * len_face)
            group_ = [it3 for it3 in it2.group if it3 in ['fluid', 'glass', 'abs', 'insul', 'left', 'main', 'right']]
            new_cells[it1] = Holder(tag_ = it1, lnode_ = lnode_, lface_ = lface_); new_cells[it1].group = group_
            
        return new_faces, new_cells

    def impute_part(layers_: list, guide_part: int, nhorizontal: int):
        """
        ------------------------
        Configure extruded parts
        """
        new_faces, new_cells = Info.impute_missing(layers_[0], layers_[1])
        node_val = []; face_val = []; cell_val = []
        
        noted_face_len = len(list(layers_[0].faces.keys())) * len(layers_)

        old_new_face_keys = list(new_faces.keys())
        new_new_face_keys = [it1 - (2 * len(list(layers_[0].faces.keys()))) + noted_face_len for it1 in old_new_face_keys]
        alias_new_face = dict(zip(old_new_face_keys, new_new_face_keys))
        match_alias_new_face = lambda x: alias_new_face[x] if x in list(alias_new_face.keys()) else x
        match_offset_face = lambda x, y: x + y * len(list(layers_[0].faces.keys())) if x < 2 * len(list(layers_[0].faces.keys())) \
                            else x + y * len(list(new_faces.keys()))

        new_faces = dict(zip(new_new_face_keys, list(new_faces.values())))
        for it1, it2 in new_cells.items():
            it2.lface = [match_alias_new_face(it3) for it3 in it2.lface]
        for it1 in range(0, len(layers_)):
            node_val.extend(list(layers_[it1].nodes.values()))
            face_val_layer = list(layers_[it1].faces.values())
            for it2 in face_val_layer:
                it2.lnode = [it3 + len(list(layers_[0].nodes.keys())) * it1 for it3 in it2.lnode]
            face_val.extend(face_val_layer)
        for it1 in range(0, guide_part - 1):
            cell_val_layer = [deepcopy(it2) for it2 in list(new_cells.values())]
            horizontal_tag = 0; ctd_horizontal = 0
            for it2 in cell_val_layer:
                it2.layer = it1
                it2.depth = horizontal_tag
                it2.lnode = [it3 + len(list(layers_[0].nodes.keys())) * it1 for it3 in it2.lnode]
                it2.lface = [match_offset_face(it3, it1) for it3 in it2.lface]
                ctd_horizontal += 1
                if ctd_horizontal % nhorizontal == 0:
                    horizontal_tag += 1
                    ctd_horizontal = 0
            cell_val.extend(cell_val_layer)
        for it1 in range(0, guide_part - 1):
            face_val_layer = [deepcopy(it2) for it2 in list(new_faces.values())]
            for it2 in face_val_layer:
                it2.lnode = [it3 + len(list(layers_[0].nodes.keys())) * it1 for it3 in it2.lnode]
            face_val.extend(face_val_layer)
        nodes_ = dict(zip(list(range(0, len(node_val))), node_val))
        faces_ = dict(zip(list(range(0, len(face_val))), face_val))
        cells_ = dict(zip(list(range(0, len(cell_val))), cell_val))

        # layer neighbor connect
        offset_new_cell = len(list(new_cells.keys())); offset_noted_face = len(list(layers_[0].faces.keys()))
        for it1 in range(0, offset_new_cell):
            for it2 in range(1, guide_part-1):
                cells_[it1 + ((it2-1) * offset_new_cell)].connect[it1 + (it2 * offset_new_cell)] = it1 + (it2 * offset_noted_face)
                cells_[it1 + (it2 * offset_new_cell)].connect[it1 + ((it2 - 1) * offset_new_cell)] = it1 + (it2 * offset_noted_face)

        # lateral neighbor connect
        offset_new_face = len(old_new_face_keys)
        for it1 in range(0, offset_new_cell-1):
            connect_ctd = len(cells_[it1].lface) - len(list(cells_[it1].connect.keys()))
            if len(list(cells_[it1].conj.keys())) > 0:
                connect_ctd = connect_ctd - len(list(cells_[it1].conj.keys()))
            ctrl = 0
            while any([connect_ctd == 0, ctrl == 1]) is False:
                for it2 in range(it1+1, offset_new_cell):
                    check_ = [it3 for it3 in cells_[it2].lface if it3 in cells_[it1].lface]
                    if all([it1 != it2, len(check_) > 0]):
                        groups_ = deepcopy(cells_[it1].group); groups_.extend(cells_[it2].group)
                        conj_cond = all([np.unique(groups_).shape[0] > np.max([len(cells_[it1].group), len(cells_[it2].group)]),
                                    check_[0] not in list(cells_[it1].conj.values())])
                        conj_cond_2 = any(['insul' in cells_[it1].group, 'insul' in cells_[it2].group])
                        check = all([conj_cond is True, conj_cond_2 is False])
                        for it3 in range(0, guide_part-1):
                            if check is True:
                                # single cell can only have one conj. Revise .geo mesh script
                                cells_[it1 + offset_new_cell * it3].conj[it2 + (offset_new_cell * it3)] = check_[0] + it3 * offset_new_face
                                cells_[it2 + offset_new_cell * it3].conj[it1 + (offset_new_cell * it3)] = check_[0] + it3 * offset_new_face
                            else:
                                cells_[it1 + (offset_new_cell * it3)].connect[it2 + (offset_new_cell * it3)] = check_[0] + it3 * offset_new_face
                                cells_[it2 + (offset_new_cell * it3)].connect[it1 + (offset_new_cell * it3)] = check_[0] + it3 * offset_new_face
                            connect_ctd = connect_ctd - 1
                ctrl = 1

        # stupid cleanup connect and bound
        for it1, it2 in cells_.items():
            for it3 in list(it2.connect.keys()):
                if it2.connect[it3] not in it2.lface:
                    del it2.connect[it3]
            bound_ = list(it2.connect.values())
            if len(list(it2.conj.keys())) > 0:
                bound_.extend(list(it2.conj.values()))
            tobound = [it3 for it3 in it2.lface if it3 not in bound_]; it2.bound = tobound
        
        # stupid cleanup groups        
        for it1, it2 in faces_.items():
            todelete = [it3 for it3 in it2.group if it3 in ['fluid', 'abs', 'glass', 'insul', 'left', 'right', 'main']]; todelete = list(np.unique(todelete))
            [it2.group.remove(it3) for it3 in todelete]
            it2.group = list(np.unique(it2.group))
        
        return nodes_, faces_, cells_

    def make_match_face_alias(faces_: dict, cells_: dict):
        match_face = {}
        for it1, it2 in faces_.items():
            check = deepcopy(it2.lnode); check.sort(reverse = False); check = tuple(check)
            if check not in list(match_face.keys()):
                match_face[check] = [it1]
            else:
                match_face[check].append(it1)
        for it1 in list(match_face.keys()):
            if len(match_face[it1]) < 2:
                del match_face[it1]
        alias_face = {}; todelete = []
        for it1, it2 in match_face.items():
            non_zero_group = [it3 for it3 in it2 if len(faces_[it3].group) > 0]
            if len(non_zero_group) > 0:
                alias_id = np.min(non_zero_group)
                alias_member = deepcopy(it2); alias_member.remove(alias_id)
                todelete.extend(alias_member)
                for it3 in it2:
                    alias_face[it3] = alias_id

        for it1, it2 in cells_.items():
            it2.lface = [alias_face[it3] for it3 in it2.lface]
        for it1 in alias_member:
            del faces_[it1]
        
    def get_face_sizeloc(nodes_: dict, faces_: dict):
        for it1, it2 in faces_.items():
            center = np.array([0, 0, 0], dtype = float)
            for it3 in it2.lnode:
                center += nodes_[it3].loc
            it2.loc = np.array([x for x in center / len(it2.lnode)])
            node_coor = [nodes_[it1].loc for it1 in it2.lnode]
            node_coor = np.transpose(np.array(node_coor))
            deltas = [np.max(node_coor[x]) - np.min(node_coor[x]) for x in [0,1,2]]
            where_min = np.where(deltas == np.min(deltas))[0][0]
            del deltas[where_min]
            size_ = deltas[0] * deltas[1]; it2.size = size_

            # if len(it2.lnode) == 3:
            #     v1 = nodes_[it2.lnode[0]].loc - nodes_[it2.lnode[1]].loc
            #     v2 = nodes_[it2.lnode[1]].loc - nodes_[it2.lnode[2]].loc
            #     it2.size = np.sqrt(np.sum(np.array([it2**2 for it2 in np.cross(v1, v2)]))) / 2
            # if len(it2.lnode) == 4:
            #     diag1 = []
            #     dist = 0.0
            #     for it3 in range(0, len(it2.lnode)):
            #         for it4 in range(it3, len(it2.lnode)):
            #             check = np.sqrt(np.sum([it5**2 for it5 in nodes_[it2.lnode[it3]].loc - nodes_[it2.lnode[it4]].loc]))
            #             if check > dist:
            #                 dist = check; diag1 = [it2.lnode[it3], it2.lnode[it4]]
            #     diag2 = [it3 for it3 in it2.lnode if it3 not in diag1]
            #     diag1.insert(0, diag1[0]); diag1.append(diag1[-1])
            #     diag2.extend(diag2)
            #     tri_sets = list(zip(diag1, diag2))
            #     tri_vec = []; tri_area = []
            #     for it3 in tri_sets:
            #         v1 = center - nodes_[it3[0]].loc
            #         v2 = center - nodes_[it3[1]].loc
            #         tri_vec.append([v1,v2])
            #     for it3 in tri_vec:
            #         area_ = np.sqrt(np.sum([it2**2 for it2 in np.cross(it3[0], it3[1])])) / 2
            #         tri_area.append(area_)
            #     it2.size = np.sum(tri_area)

    def get_cell_sizeloc(nodes_: dict, cells_: dict):
        for it1, it2 in cells_.items():
            center = np.array([0, 0, 0], dtype = float)
            for it3 in it2.lnode:
                center += nodes_[it3].loc
            it2.loc = np.array([x for x in center / len(it2.lnode)])
            node_coor = [nodes_[it1].loc for it1 in it2.lnode]
            size_ = sp.ConvexHull(node_coor).volume; it2.size = size_
    
    def get_cell_wdist(nodes_: dict, faces_: dict, cells_: dict):
        for it1, it2 in cells_.items():
            wall_face = [it3 for it3 in it2.lface if 'noslip' in faces_[it3].group]
            if len(wall_face) > 0:
                dCf_ = faces_[wall_face[0]].loc - it2.loc; dCf_val = np.sqrt(np.sum([it3**2 for it3 in dCf_]))
                eCf_ = dCf_ / dCf_val
                vf1 = nodes_[faces_[wall_face[0]].lnode[0]].loc - faces_[wall_face[0]].loc
                vf2 = nodes_[faces_[wall_face[0]].lnode[1]].loc - faces_[wall_face[0]].loc
                vS = np.cross(vf1, vf2); vS_scale = np.sqrt(np.sum(np.array([it3**2 for it3 in vS])))
                n_ = vS / vS_scale
                if np.dot(eCf_, n_) < 0:
                    n_ = -1 * n_
                cos_ = np.dot(eCf_, n_)
                wdist_ = np.abs(cos_ * dCf_val)
                it2.wdist = wdist_

    def correct_idiot_labelling_connect_and_conj(faces_: dict, cells_: dict):
        # conj is also faulty
        match_face = lambda x: [y for y, z in faces_.items() if all([p in z.lnode for p in x])]
        for it1, it2 in cells_.items():
            correct_bound_dumbass = 0
            real_lface = []
            it2.lface = []
            for it3 in list(it2.connect.keys()):
                match_node = [x for x in it2.lnode if x in cells_[it3].lnode]
                face_match = match_face(match_node)
                it2.connect[it3] = face_match[0]
                cells_[it3].connect[it1] = face_match[0]
                real_lface.append(face_match[0])
            for it3 in list(it2.conj.keys()):
                match_node = [x for x in it2.lnode if x in cells_[it3].lnode]
                face_match = match_face(match_node)
                it2.conj[it3] = face_match[0]
                cells_[it3].conj[it1] = face_match[0]
                real_lface.append(face_match[0])
            check_bound = deepcopy(it2.bound)
            check_bound_dup = deepcopy(check_bound)
            it2.bound = []
            for it3 in check_bound_dup:
                match_node = [x for x in it2.lnode if x in faces_[it3].lnode]
                if len(match_node) < 4:
                    del check_bound[np.where(check_bound == it3)[0][0]]
                    correct_bound_dumbass += 1
                if any([it3 in list(it2.connect.values()), it3 in list(it2.conj.values())]):
                    del check_bound[np.where(check_bound == it3)[0][0]]
            check_bound = list(np.unique(check_bound)); real_lface.extend(check_bound)
            while all([correct_bound_dumbass > 0, len(real_lface) < 6]):
                for it3, it4 in faces_.items():
                    match_node = [x for x in it2.lnode if x in faces_[it3].lnode]
                    if all([len(match_node) == 4, it3 not in check_bound,
                            it3 not in real_lface]):
                        real_lface.append(it3)
                        check_bound.append(it3)
                        correct_bound_dumbass -= 1
            if len(real_lface) < 6:
                count_node = []; [count_node.extend(faces_[x].lnode) for x in real_lface]
                count_node = dict(Counter(count_node))
                new_face_lnode = [list(count_node.keys())[x] for x in np.where(np.array(list(count_node.values())) == 2)[0]]
                if len(new_face_lnode) == 4:
                    new_tag = len(list(faces_.keys()))
                    new_face = Holder(new_tag, 2, lnode_ = new_face_lnode)
                    faces_[new_tag] = new_face
                    check_bound.append(new_tag)
                    real_lface.append(new_tag)

            it2.bound = deepcopy(check_bound)
            it2.lface = deepcopy(real_lface)

    def __init__(self, fileloc, extrude_len: float, extrude_part: int, nhorizontal: int):
        """
        --------------------
        Extrude and collapse
        """
        layers_, guide_part = Info.impute_layer(fileloc, extrude_len, extrude_part)
        nodes_, faces_, cells_ = Info.impute_part(layers_, guide_part, nhorizontal)
        # Info.make_match_face_alias(faces_, cells_)
        Info.correct_idiot_labelling_connect_and_conj(faces_, cells_)

        """
        -------------------------------------
        Complete mesh set        
        """
        Info.get_face_sizeloc(nodes_, faces_)
        Info.get_cell_sizeloc(nodes_, cells_)
        # Info.get_cell_wdist(nodes_, faces_, cells_)

        """
        --------
        finalize
        """
        self.__nodes = nodes_
        self.__faces = faces_
        self.__cells = cells_

    @property
    def nodes(self):
        return self.__nodes
    @property
    def faces(self):
        return self.__faces
    @property
    def cells(self):
        return self.__cells

class Axes:    
    def __init__(self, x_: sparse.csr_matrix, y_: sparse.csr_matrix, z_: sparse.csr_matrix):
        self.__x = x_; self.__y = y_; self.__z = z_
    def vec(self, row: int, col: int):
        x_ = self.__x[row, col]; y_ = self.__y[row, col]; z_ = self.__z[row, col]
        return np.array([x_, y_, z_])
    def scalar(self, row: int, col: int):
        return np.sqrt(np.sum([it1**2 for it1 in self.vec(row, col)]))
    def norm(self, row: int, col: int):
        try:
            return np.array([x for x in self.vec(row, col) / self.scalar(row, col)])
        except:
            return False

class Geom:
    def __init__(self, info: Info):
        lcell_ = len(list(info.cells.keys()))
        lface_ = len(list(info.faces.keys()))
        cf = np.zeros(shape=(lcell_, lface_), dtype = float)
        cc = np.zeros(shape=(lcell_, lcell_), dtype = float)
        # dCF
        dCF_x = deepcopy(cc); dCF_y = deepcopy(cc); dCF_z = deepcopy(cc)
        for it1, it2 in info.cells.items():
            for it3, it4 in it2.connect.items():
                dCF_ = np.array([x for x in info.cells[it3].loc - it2.loc])
                dCF_x[it1][it3] = dCF_[0]; dCF_y[it1][it3] = dCF_[1]; dCF_z[it1][it3] = dCF_[2]
        self.__dCF = Axes(sparse.lil_matrix(dCF_x).tocsr(), sparse.lil_matrix(dCF_y).tocsr(),
                          sparse.lil_matrix(dCF_z).tocsr())     
        # dCf, Sf, Ef, Tf
        dCf_x = deepcopy(cf); dCf_y = deepcopy(cf); dCf_z = deepcopy(cf)
        Sf_x = deepcopy(cf); Sf_y = deepcopy(cf); Sf_z = deepcopy(cf)
        Ef_x = deepcopy(cf); Ef_y = deepcopy(cf); Ef_z = deepcopy(cf)
        Tf_x = deepcopy(cf); Tf_y = deepcopy(cf); Tf_z = deepcopy(cf)
        for it1, it2 in info.cells.items():
            for it3 in it2.lface:
                dCf_ = np.array([x for x in info.faces[it3].loc - it2.loc]); dCf_val = np.sqrt(np.sum([it4**2 for it4 in dCf_]))
                #v1 = info.nodes[info.faces[it3].lnode[0]].loc - info.faces[it3].loc
                #v2 = info.nodes[info.faces[it3].lnode[1]].loc - info.faces[it3].loc
                #vS = np.cross(v1, v2); vS_scalar = np.sqrt(np.sum([it4**2 for it4 in vS]))
                #if np.dot(vS, dCf_) < 0:
                #    vS = -1 * vS
                Sf_ = np.array([x for x in (dCf_ / dCf_val) * info.faces[it3].size])
                Ef_ = Sf_
                Tf_ = Sf_ - Ef_

                dCf_x[it1][it3] = dCf_[0]; dCf_y[it1][it3] = dCf_[1]; dCf_z[it1][it3] = dCf_[2]
                Sf_x[it1][it3] = Sf_[0]; Sf_y[it1][it3] = Sf_[1]; Sf_z[it1][it3] = Sf_[2]
                Ef_x[it1][it3] = Ef_[0]; Ef_y[it1][it3] = Ef_[1]; Ef_z[it1][it3] = Ef_[2]
                Tf_x[it1][it3] = Tf_[0]; Tf_y[it1][it3] = Tf_[1]; Tf_z[it1][it3] = Tf_[2]
        self.__dCf = Axes(sparse.lil_matrix(dCf_x).tocsr(), sparse.lil_matrix(dCf_y).tocsr(),
                              sparse.lil_matrix(dCf_z).tocsr())
        self.__Sf = Axes(sparse.lil_matrix(Sf_x).tocsr(), sparse.lil_matrix(Sf_y).tocsr(),
                           sparse.lil_matrix(Sf_z).tocsr())
        self.__Ef = Axes(sparse.lil_matrix(Ef_x).tocsr(), sparse.lil_matrix(Ef_y).tocsr(),
                             sparse.lil_matrix(Ef_z).tocsr())
        self.__Tf = Axes(sparse.lil_matrix(Tf_x).tocsr(), sparse.lil_matrix(Tf_y).tocsr(),
                             sparse.lil_matrix(Tf_z).tocsr())

    @property
    def dCf(self):
        return self.__dCf
    @property
    def dCF(self):
        return self.__dCF
    @property
    def Sf(self):
        return self.__Sf
    @property
    def Ef(self):
        return self.__Ef
    @property
    def Tf(self):
        return self.__Tf


"""
------
UTIL
"""

class Iter:
    def __init__(self, init):
        self.__current = init
        self.__new = init
        self.__ltime =  [init]
    def update(self):
        self.__current = self.__new
    def forward(self):
        self.__ltime.append(self.__current)
        self.__new = self.__current

    @property
    def current(self):
        return self.__current
    @property
    def new(self):
        return self.__new
    @new.setter
    def new(self, val_):
        self.__new = val_
    @property
    def ltime(self):
        return self.__ltime
    @property
    def last(self):
        return self.__ltime[-1]
    
class Prop:
    @staticmethod
    def calc_rho(P_: float, T_: float, W_: float = 0.02, working_density_: float = 1.293) -> float:
        #P_ = float(np.min([101500, np.max([np.abs(P_), 101325])]))
        #T_ = float(np.min([400, np.max([np.abs(T_), 290])]))
        if working_density_ <= float(0):
            return 1.293
        else:
            return 34 * pow(10, -6) * (T_ - 300)
        #return 1 / HAPropsSI("Vha", "P", P_, "T", T_, "W", W_)
        #return 1.293
    @staticmethod
    def calc_mu(P_: float, T_: float, W_ = 0.02) -> float:
        #P_ = float(np.min([101500, np.max([np.abs(P_), 101325])]))
        #T_ = float(np.min([400, np.max([np.abs(T_), 290])]))
        #return HAPropsSI("mu", "P", P_, "T", T_, "W", W_)
        return 1.7894 * pow(10, -5)
        # return (1.846 + 0.00472 * (T_ - 300))* pow(10, -5)
    @staticmethod
    def calc_k(P_: float, T_: float, W_ = 0.02) -> float:
        #P_ = float(np.min([101500, np.max([np.abs(P_), 101325])]))
        #T_ = float(np.min([400, np.max([np.abs(T_), 290])]))
        #return HAPropsSI("k", "P", P_, "T", T_, "W", W_)
        return 0.025
        # return 0.0263 + 0.000074 * (T_ - 300)
    @staticmethod
    def calc_cp(P_: float, T_: float, W_ = 0.02) -> float:
        #P_ = float(np.min([101500, np.max([np.abs(P_), 101325])]))
        #T_ = float(np.min([400, np.max([np.abs(T_), 290])]))
        #return HAPropsSI("cp_ha", "P", P_ + 101325, "T", T_, "W", W_)
        return 1006.43
        # return (1.007 + 0.00004 * (T_ - 300)) * pow(10, 3)
    @staticmethod
    def calc_z(P_: float, T_: float, W_ = 0.02) -> float:
        P_ = float(np.min([101500, np.max([np.abs(P_), 101325])]))
        T_ = float(np.min([400, np.max([np.abs(T_), 290])]))
        return HAPropsSI("Z", "P", P_, "T", T_, "W", W_)

class Value:
    def __init__(self, name_: str, init: float, ncell: int, nface: int, info_: Info = None,
                 geom_: Geom = None, unit_only: bool = False):
        self.__name = name_
        # zero grad at init
        funit_ = {}; cunit_ = {}
        fgrad_ = {}; cgrad_ = {}
        for it1 in range(0, ncell):
            cunit_[it1] = Iter(init)
            if unit_only is False:
                cgrad_[it1] = Iter(np.array([0, 0, 0], dtype = float))
        for it1 in range(0, nface):
            funit_[it1] = Iter(init)
            if unit_only is False:
                fgrad_[it1] = Iter(np.array([0, 0, 0], dtype = float))
        self.__cunit = cunit_; self.__funit = funit_
        self.__cgrad = cgrad_; self.__fgrad = fgrad_
        
    @property
    def name(self):
        return self.__name
    @property
    def funit(self):
        return self.__funit
    @property
    def cunit(self):
        return self.__cunit
    @property
    def fgrad(self):
        return self.__fgrad
    @property
    def cgrad(self):
        return self.__cgrad

    """    
    def new(self, cunit_: list = None, funit_: list = None):
        if cunit_ is not None:
            for it1 in cunit_:
                self.cunit[it1[0], it2[1]] = it1[2]
        if funit_ is not None:
            for it1 in funit_:
                self.funit[it1[0], it2[1]] = it2[2]
    """
    def renew_gradient(self, info_, geom_: Geom):
        if len(list(self.cgrad.keys())) > 0:
            least_square_gradient(info_, geom_, self)
    def update(self, info_: Info, geom_: Geom, with_grad: bool = True):
        for it1, it2 in self.funit.items():
            it2.update()
        for it1, it2 in self.cunit.items():
            it2.update()
        if with_grad is True:
            self.renew_gradient(info_, geom_)
            for it1, it2 in self.cgrad.items():
                it2.update()
            for it1, it2 in self.fgrad.items():
                it2.update()
        elif with_grad is False:
            for it1, it2 in self.cgrad.items():
                it2.update()
            for it1, it2 in self.fgrad.items():
                it2.update()
    def forward(self):
        for it1, it2 in self.funit.items():
            it2.forward()
        for it1, it2 in self.cunit.items():
            it2.forward()
        if self.cgrad is not None:
            for it1, it2 in self.fgrad.items():
                it2.forward()
            for it1, it3 in self.cgrad.items():
                it2.forward()

class Pool:
    def __init__(self, info_: Info, geom_: Geom, P_: float = 0, v_: float = 0, w_: float = 0, T_: float = 300, mdot_: float = 0):
        ncell_ = len(list(info_.cells.keys())); nface_ = len(list(info_.faces.keys()))
        self.__u = Value("u", float(0), ncell_, nface_)
        self.__v = Value("v", v_, ncell_, nface_)
        self.__w = Value("w", w_, ncell_, nface_)
        self.__P = Value("P", P_, ncell_, nface_)
        self.__T = Value("T", T_, ncell_, nface_)
        self.__k = Value("k", float(1e-5), ncell_, nface_)
        self.__omega = Value("omega", float(0), ncell_, nface_)
        self.__cp = Value('cp', Prop.calc_cp(P_, T_), ncell_, nface_)
        self.__Pcor = Value("Pcor", float(0), ncell_, nface_)
        self.__mdot = Value("mdot", mdot_, ncell_, nface_, info_ = info_, geom_ = geom_, unit_only = True)
        self.__working_density = 1.293

    @property
    def u(self):
        return self.__u
    @property
    def v(self):
        return self.__v
    @property
    def w(self):
        return self.__w 
    @property
    def P(self):
        return self.__P
    @property
    def T(self):
        return self.__T
    @property
    def Pcor(self):
        return self.__Pcor
    @property
    def mdot(self):
        return self.__mdot
    @property
    def k(self):
        return self.__k
    @property
    def omega(self):
        return self.__omega
    @property
    def cp(self):
        return self.__cp
    @property
    def working_density(self):
        return self.__working_density
    @working_density.setter
    def working_density(self, new_working_density):
        self.__working_density = new_working_density

class User:
    def __init__(self, q_glass: float, q_abs: float):
        self.__constants = {'P': 0, 'T': 300}
        self.__solid_eps = {'glass': 0.89, 'abs': 0.95, 'insul': 0.95}
        #self.__solid_alpha = {'glass': 0.06, 'abs': 0.95} # black matte Al, abs insulated polyurethane, insul heat resistant grade polystrene
        self.__solid_alpha = {'glass': 0.9 / (840 * 2400), 'abs': 0.83 / (1200 * 413), 'insul': 0.159 / (1100 * 1050)}
        self.__solid_cp = {'glass': 840, 'abs': 1200, 'insul': 1100} # 2440
        self.__solid_k = {'glass': 0.9, 'abs': 0.83, 'insul': 0.159} # W m/K # 0.05
        self.__solid_q = {'glass': q_glass, 'abs': q_abs} # W / m2
        self.__solid_rho = {'glass': 2400, 'abs': 413, 'insul': 1050} # 1.39 g/cc
    @property
    def constants(self):
        return self.__constants
    @property
    def solid_cp(self):
        return self.__solid_cp
    @property
    def solid_eps(self):
        return self.__solid_eps
    @property
    def solid_k(self):
        return self.__solid_k
    @property
    def solid_alpha(self):
        return self.__solid_alpha
    @property
    def solid_q(self):
        return self.__solid_q
    @solid_q.setter
    def solid_q(self, new_solid_q):
        self.__solid_q = new_solid_q
    @property
    def solid_rho(self):
        return self.__solid_rho

class Export:
    def __init__(self, column_pd_list, io_layer_, wall_layer_, solid_layer_, fluid_layer_):
        self.__column_list = column_pd_list
        self.__io_layer = io_layer_
        self.__wall_layer = wall_layer_
        self.__solid_layer = solid_layer_
        self.__fluid_layer = fluid_layer_
    
    @classmethod
    def find_member_layers(cls, info_: Info):
        # length-channel gap wise -> u, v, w, P, T (avg)
        # channel gap wise inlet-outlet -> u, v, w, P, T (avg)
        # length-channel gap wise abs-fluid bound -> u, v, w, P, T (avg)
        # length-channel gap wise glass-fluid bound -> T (avg)

        io_layer_members = {'inlet': {}, 'outlet': {}} # io -> channel gap-wise only (face)
        wall_layer_members = {'abs': {}, 'glass': {}, 'insul': {}} # conj_solid -> length-wise only (face)
        solid_layer_members = {'abs': {}, 'glass': {}, 'insul': {}} # conj_solid -> length-wise only (cell)
        fluid_layer_members = {} # length-wise -> channel gap-wise (cell)
        
        for it1, it2 in info_.cells.items():
            if it2.group[0] == 'fluid':
                if it2.layer + 1 not in list(fluid_layer_members.keys()):
                    fluid_layer_members[it2.layer + 1] = {}; fluid_layer_members[it2.layer + 1][it2.depth] = [it1]
                elif it2.depth not in list(fluid_layer_members[it2.layer + 1].keys()):
                    fluid_layer_members[it2.layer + 1][it2.depth] = [it1]
                else:
                    fluid_layer_members[it2.layer + 1][it2.depth].append(it1)
                for it3 in it2.bound:
                    if 'inlet' in info_.faces[it3].group:
                        if it2.depth not in list(io_layer_members['inlet'].keys()):
                            io_layer_members['inlet'][it2.depth] = [it3]
                        else:
                            io_layer_members['inlet'][it2.depth].append(it3)
                    elif 'outlet' in info_.faces[it3].group:
                        if it2.depth not in list(io_layer_members['outlet'].keys()):
                            io_layer_members['outlet'][it2.depth] = [it3]
                        else:
                            io_layer_members['outlet'][it2.depth].append(it3)
                for it3, it4 in it2.conj.items():
                    if it2.layer not in list(wall_layer_members[info_.cells[it3].group[0]].keys()):
                        wall_layer_members[info_.cells[it3].group[0]][it2.layer] = [it4]
                        solid_layer_members[info_.cells[it3].group[0]][it2.layer] = [it3]
                    else:
                        wall_layer_members[info_.cells[it3].group[0]][it2.layer].append(it4)
                        solid_layer_members[info_.cells[it3].group[0]][it2.layer].append(it3)
            elif it2.group[0] == 'insul':
                if it2.layer not in list(wall_layer_members[info_.cells[it3].group[0]].keys()):
                    wall_layer_members[info_.cells[it3].group[0]][it2.layer] = [it4]
                    solid_layer_members[info_.cells[it3].group[0]][it2.layer] = [it3]
                else:
                    wall_layer_members[info_.cells[it3].group[0]][it2.layer].append(it4)
                    solid_layer_members[info_.cells[it3].group[0]][it2.layer].append(it3)

        io_layer_members[0] = io_layer_members.pop('inlet')
        io_layer_members[len(list(fluid_layer_members.keys())) + 1] = io_layer_members.pop('outlet')
        
        io_layer_columns = []
        wall_layer_columns = []
        solid_layer_columns = []
        fluid_layer_columns = []
        
        for it1, it2 in io_layer_members.items():
            for it3, it4 in it2.items():
                io_layer_columns.append('fluid' + '_' + str(it1) + '_' + str(it3))
        for it1, it2 in solid_layer_members.items():
            for it3, it4 in it2.items():
            #     wall_layer_columns.append('conj_' + it1 + str(it3) + '_d')
                solid_layer_columns.append(it1 + '_' + str(it3) + '_d')
        for it1, it2 in fluid_layer_members.items():
            for it3, it4 in it2.items():
                fluid_layer_columns.append('fluid_' + str(it1) + '_' + str(it3))
        
        # io_layer_columns.extend(wall_layer_columns); io_layer_columns.extend(fluid_layer_columns)
        
        column_list_pd = dict({'fluid': fluid_layer_columns, 'solid': solid_layer_columns, 'io': io_layer_columns})
        
        return cls(column_list_pd, io_layer_members, wall_layer_members, solid_layer_members, fluid_layer_members)
    
    @property
    def column_list(self):
        return self.__column_list
    @property
    def io_layer(self):
        return self.__io_layer
    @property
    def wall_layer(self):
        return self.__wall_layer
    @property
    def solid_layer(self):
        return self.__solid_layer
    @property
    def fluid_layer(self):
        return self.__fluid_layer
    
    def generate_dataset_entry(self, df_fluid, df_solid, pool_: Pool):
        for it1 in ['u', 'v', 'w', 'P', 'T', 'k', 'omega']:
            new_entry_io = np.array([])
            # new_entry_wall = np.array([])
            new_entry_fluid = np.array([])
            for it3, it4 in self.io_layer.items():
                for it5, it6 in it4.items():
                    avg_val = np.mean([pool_.__getattribute__(it1).funit[it7].current for it7 in it6])
                    new_entry_io = np.append(new_entry_io, avg_val)
            # for it3, it4 in self.wall_layer.items():
            #     for it5, it6 in it4.items():
            #         avg_val = np.mean([pool_.__getattribute__(it1).funit[it7].current for it7 in it6])
            #         new_entry_wall = np.append(new_entry_wall, avg_val)
            for it3, it4 in self.fluid_layer.items():
                for it5, it6 in it4.items():
                    avg_val = np.mean([pool_.__getattribute__(it1).cunit[it7].current for it7 in it6])
                    new_entry_fluid = np.append(new_entry_fluid, avg_val)
            new_entry_fluid_concat = np.concatenate((new_entry_fluid, new_entry_io))
            df_fluid[it1].loc[df_fluid[it1].shape[0], :] = new_entry_fluid_concat
        for it1, it2 in df_solid.items():
            new_entry_solid = []
            for it3, it4 in self.solid_layer.items():
                for it5, it6 in it4.items():
                    new_entry_solid.append(np.mean([pool_.__getattribute__(it1).cunit[it7].current for it7 in it6]))
            it2.loc[it2.shape[0], :] = new_entry_solid        

        return

    def generate_loc_size_legend(self, info_: Info):
        io_loc_legend = deepcopy(self.io_layer) # io -> channel gap-wise only (face)
        wall_loc_legend = deepcopy(self.wall_layer) # conj_solid -> length-wise only (face)
        solid_loc_legend = deepcopy(self.solid_layer) # conj_solid -> length-wise only (cell)
        fluid_loc_legend = deepcopy(self.fluid_layer) # length-wise -> channel gap-wise (cell)
        
        for it1, it2 in io_loc_legend.items():
            for it3, it4 in it2.items():
                new_loc_entry = np.array([0, 0, 0], dtype = float)
                for it5 in it4:
                    new_loc_entry += info_.faces[it5].loc
                new_loc_entry = new_loc_entry / len(it4)
                it4 = new_loc_entry
        for it1, it2 in wall_loc_legend.items():
            for it3, it4 in it2.items():
                new_loc_entry = np.array([0, 0, 0], dtype = float)
                for it5 in it4:
                    new_loc_entry += info_.faces[it5].loc
                new_loc_entry = new_loc_entry / len(it4)
                it4 = new_loc_entry
        for it1, it2 in solid_loc_legend.items():
            for it3, it4 in it2.items():
                new_loc_entry = np.array([0, 0, 0], dtype = float)
                for it5 in it4:
                    new_loc_entry += info_.cells[it5].loc
                new_loc_entry = new_loc_entry / len(it4)
                it4 = new_loc_entry
        for it1, it2 in fluid_loc_legend.items():
            for it3, it4 in it2.items():
                new_loc_entry = np.array([0, 0, 0], dtype = float)
                for it5 in it4:
                    new_loc_entry += info_.cells[it5].loc
                new_loc_entry = new_loc_entry / len(it4)
        
        return dict({'io': io_loc_legend, 'wall': wall_loc_legend, 'solid': solid_loc_legend,
                     'fluid': fluid_loc_legend})
        
class Results:
    def __init__(self, info_: Info, df_fluid_, df_solid_, export_: Export, delta_t_, lambda_, tol_, geom_external, qglass_, qabs_):
        self.__fluid = df_fluid_
        self.__solid = df_solid_
        self.__loc_legend = export_.generate_loc_size_legend(info_)
        self.__area = {'abs': geom_external[0], 'glass': geom_external[1], 'insul': geom_external[2], 'fluid': geom_external[3]}
        self.__length = geom_external[3]
        self.__delta_t = delta_t_
        self.__lambda = lambda_
        self.__tol = tol_
        self.__qglass = qglass_
        self.__qabs = qabs_
    
    @property
    def fluid(self):
        return self.__fluid
    @property
    def solid(self):
        return self.__solid
    @property
    def loc_legend(self):
        return self.__loc_legend
    @property
    def area(self):
        return self.__area
    @property
    def length(self):
        return self.__length
    @property
    def delta_t(self):
        return self.__delta_t
    @property
    def under_relax(self):
        return self.__lambda
    @property
    def tol(self):
        return self.__tol
    @property
    def qglass(self):
        return self.__qglass
    @property
    def qabs(self):
        return self.__qabs


"""
-----------
INTERPOLATE
"""

def least_square_gradient(info_: Info, geom_: Geom, value_: Value):
    for it1, it2 in info_.cells.items():
        a_neigh_coef = np.zeros(shape=(3,3), dtype = float)
        a_bound_coef = np.zeros(shape=(3,3), dtype = float)
        b_neigh_coef = np.zeros(shape=(3,1), dtype = float)
        b_bound_coef = np.zeros(shape=(3,1), dtype = float)
        for it3, it4 in it2.connect.items():
            w_ = 1 / geom_.dCF.scalar(it1, it3)
            dCF_ = geom_.dCF.vec(it1, it3)
            for it5 in range(0, 3):
                for it6 in range(0, 3):
                    a_neigh_coef[it5][it6] += w_ * dCF_[it5] * dCF_[it6]
                b_neigh_coef[it5][0] += w_ * dCF_[it5] * (value_.cunit[it3].current - value_.cunit[it1].current)
        if len(list(it2.conj.keys())) > 0:
            for it3, it4 in it2.conj.items():
                w_ = 1 / geom_.dCf.scalar(it1, it4)
                dCf_ = geom_.dCf.vec(it1, it4)
                for it5 in range(0, 3):
                    for it6 in range(0, 3):
                        a_neigh_coef[it5][it6] += w_ * dCf_[it5] * dCf_[it6]
                    b_neigh_coef[it5][0] += w_ * dCf_[it5] * (value_.funit[it4].current - value_.cunit[it1].current)
        for it3 in it2.bound:
            if any([it4 in ['noslip', 'inlet', 'outlet'] for it4 in info_.faces[it3].group]):
                w_ = 1 / geom_.dCf.scalar(it1, it3)
                dCf_ = geom_.dCf.vec(it1, it3)
                for it4 in range(0, 3):
                    for it5 in range(0, 3):
                        a_bound_coef[it4][it5] += w_ * dCf_[it4] * dCf_[it5]
                    b_bound_coef[it4][0] += w_ * dCf_[it4] * (value_.funit[it3].current - value_.cunit[it1].current)
        
        a_coef = a_neigh_coef + a_bound_coef; b_coef = b_neigh_coef + b_bound_coef

        #try:
        c, low = linalg.cho_factor(a_coef)
        grad_ = linalg.cho_solve((c, low), b_coef)
        grad_ = grad_.ravel()
        #except:
        #    #grad_ = linalg.lstsq(a_coef, b_coef); grad_ = grad_[0]
        #    grad_ = value_.cgrad[it1].current

        value_.cgrad[it1].new = grad_
    clear_of_interpolation = []
    for it1, it2 in info_.cells.items():
        for it3, it4 in it2.connect.items():
            if all([it4 in clear_of_interpolation]) is False:
                dCf_ = geom_.dCf.scalar(it1, it4); dFf_ = geom_.dCf.scalar(it3, it4)
                gC_ = dFf_ / (dCf_ + dFf_)
                dCF_ = geom_.dCF.scalar(it1, it3)
                eCF_ = geom_.dCF.norm(it1, it3)
                gradC_ = value_.cgrad[it1].current
                gradF_ = value_.cgrad[it1].current
                gradfCF_ = (gC_ * gradC_) + ((1 - gC_) * gradF_)
                gradf_ = gradfCF_ + ( ((value_.cunit[it3].current - value_.cunit[it1].current) / dCF_)
                                     - (np.dot(gradfCF_, eCF_))) * eCF_
                value_.fgrad[it4].new = gradf_
                clear_of_interpolation.append(it4)
        if len(list(it2.conj.values())) > 0:
            for it3, it4 in it2.conj.items():
                if it4 not in clear_of_interpolation:
                    dCf_ = geom_.dCf.scalar(it1, it4); dFf_ = geom_.dCf.scalar(it3, it4)
                    gC_ = dFf_ / (dCf_ + dFf_)
                    dCF_ = geom_.dCF.scalar(it1, it3)
                    eCF_ = geom_.dCF.norm(it1, it3)
                    gradC_ = value_.cgrad[it1].current
                    gradF_ = value_.cgrad[it1].current
                    gradfCF_ = (gC_ * gradC_) + ((1 - gC_) * gradF_)
                    try:
                        gradf_ = gradfCF_ + ( ((value_.cunit[it3].current - value_.cunit[it1].current) / dCF_)
                                                - (np.dot(gradfCF_, eCF_))) * eCF_
                    except:
                        eCf_ = geom_.dCf.norm(it1, it4)
                        gradf_ = gradfCF_ + ( ((value_.cunit[it3].current - value_.cunit[it1].current) / dCf_)
                                                - (np.dot(gradfCF_, eCf_))) * eCf_
                    value_.fgrad[it4].new = gradf_
                    clear_of_interpolation.append(it4)
        for it3 in it2.bound:
            if any([it4 in ['noslip', 'inlet', 'outlet'] for it4 in info_.faces[it3].group]):
                gradC_ = value_.cgrad[it1].current
                eCf_ = geom_.dCf.norm(it1, it3)
                gradf_ = gradC_ - (np.dot(gradC_, eCf_)) * eCf_
                value_.fgrad[it3].new = gradf_
                clear_of_interpolation.append(it3)                

def least_square_gradient_norm(row_: int, new_dist: np.array, info_: Info, geom_: Geom):
    a_neigh_coef = np.zeros(shape=(3,3), dtype = float)
    a_bound_coef = np.zeros(shape=(3,3), dtype = float)
    b_neigh_coef = np.zeros(shape=(3,1), dtype = float)
    b_bound_coef = np.zeros(shape=(3,1), dtype = float)
    for it3, it4 in info_.cells[row_].connect.items():
        w_ = 1 / geom_.dCF.scalar(row_, it3)
        dCF_ = geom_.dCF.vec(row_, it3)
        for it5 in range(0, 3):
            for it6 in range(0, 3):
                a_neigh_coef[it5][it6] += w_ * dCF_[it5] * dCF_[it6]
            b_neigh_coef[it5][0] += w_ * dCF_[it5] * (new_dist[it3] - new_dist[row_])
    for it3 in info_.cells[row_].bound:
        if any([it4 in ['noslip'] for it4 in info_.faces[it3].group]):
            w_ = 1 / geom_.dCf.scalar(row_, it3)
            dCf_ = geom_.dCf.vec(row_, it3)
            for it4 in range(0, 3):
                for it5 in range(0, 3):
                    a_bound_coef[it4][it5] += w_ * dCf_[it4] * dCf_[it5]
                b_bound_coef[it4][0] += 0
    
    a_coef = a_neigh_coef + a_bound_coef; b_coef = b_neigh_coef + b_bound_coef

    #try:
    c, low = linalg.cho_factor(a_coef)
    grad_ = linalg.cho_solve((c, low), b_coef)
    grad_ = grad_.ravel()
    #except:
    #    #grad_ = linalg.lstsq(a_coef, b_coef); grad_ = grad_[0]
    #    grad_ = value_.cgrad[it1].current

    return grad_

def linear_interpolate(val1_: tuple, val2_: tuple, info_: Info, geom_: Geom):
    part_face = info_.cells[val1_[0]].connect[val2_[0]]
    dCf_ = geom_.dCf.scalar(val1_[0], part_face); dFf_ = geom_.dCf.scalar(val2_[0], part_face)
    gC_ = dFf_ / (dCf_ + dFf_)
    return (gC_ * val1_[1]) + ((1 - gC_) * val2_[1])

def closure_interpolate(val1_: float, val2_: float, row_: int, pool_: Pool, info_: Info):
    F1_ = calc_F1(row_, pool_, info_)
    return F1_ * val1_ + (1 - F1_) * val2_


"""
----
WALL
"""

def calc_St(row_: int, pool_: Pool):
    st_ = [pool_.u.cgrad[row_].current[0]**2, pool_.v.cgrad[row_].current[1]**2,
           pool_.w.cgrad[row_].current[2]**2]
    return np.sqrt(np.sum(st_))

def calc_F1(row_: int, pool_: Pool, info_: Info):
    rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
    mu_ = Prop.calc_mu(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current)
    try:
        CD_ = np.max([2 * rho_ * 1.186 * np.dot(pool_.k.cgrad[row_].current,
                    pool_.omega.cgrad[row_].current) / pool_.omega.cunit[row_].current, pow(10, -10)])
        gamma111_ = np.sqrt(pool_.k.cunit[row_].current) / (0.09 * pool_.omega.cunit[row_].current * \
                                                            info_.cells[row_].wdist)
        gamma112_ = 500 * (mu_ / rho_) / (info_.cells[row_].wdist**2 * pool_.omega.cunit[row_].current)
        gamma11_ = np.max([gamma111_, gamma112_])
        gamma12_= 4 * rho_ * 1.186 * pool_.k.cunit[row_].current / (CD_ * info_.cells[row_].wdist**2)
    except:
        gamma11_ = 0; gamma12_ = 0
    return math.tanh(np.min([gamma11_, gamma12_]))

def calc_F2(row_: int, pool_: Pool, info_: Info):
    rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
    mu_ = Prop.calc_mu(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current)
    try:
        gamma21_ =  2 * np.sqrt(pool_.k.cunit[row_].current) / (0.09 * pool_.omega.cunit[row_].current * info_.cells[row_].wdist)
        gamma22_ =  500 * (mu_ / rho_) / (info_.cells[row_].wdist**2 * pool_.omega.cunit[row_].current)
    except:
        gamma21_ = 0; gamma22_ = 0
    return math.tanh(np.max([gamma21_, gamma22_])**2)

def calc_mut(row_: int, pool_: Pool, info_: Info):
    # a1 = 0.31
    rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
    denom_ = np.sqrt(2) * calc_St(row_, pool_) * calc_F2(row_, pool_, info_)
    try:
        mut = rho_ * 0.31 * pool_.k.cunit[row_].current / np.max([0.31 * pool_.omega.cunit[row_].current, denom_])
    except:
        mut = 0
    return mut

def calc_kt(row_: int, pool_: Pool, info_: Info):
    mut_ = calc_mut(row_, pool_, info_)
    return mut_ / 0.9

def calc_u_tau(row_: int, col_: int, pool_: Pool, info_: Info, mu_: float, rho_: float, parallelv_: float, wdist_: float):
    ustar_ = calc_u_star(row_, col_, pool_, mu_, rho_, parallelv_, wdist_)
    dstar_ = wdist_ * ustar_ / (mu_ / rho_)
    utau_vis = np.sqrt(mu_ * parallelv_ / (rho_ * wdist_))
    utau_log = parallelv_ / ((math.log10(dstar_) / 0.41) + 5.25)

    return pow(utau_vis**4 + utau_log**4, 0.25)

def calc_u_star(row_: int, col_: int, pool_: Pool, mu_: float, rho_: float, parallelv_: float, wdist_: float):
    ustar_vis = np.sqrt(mu_ * parallelv_/ (rho_ * wdist_))
    ustar_log = pow(0.09, 1/4) * pow(pool_.k.cunit[row_].current, 0.5)

    return pow(ustar_vis**4 + ustar_log**4, 0.25)

def calc_u_plus(row_: int, col_: int, pool_: Pool, info_: Info, geom_: Geom, mu_: float, rho_: float, parallelv_: float, wdist_: float):
    utau_ = calc_u_tau(row_, col_, pool_, info_, mu_, rho_, parallelv_, wdist_)
    v_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
    n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(row_, col_)
    parallelv_ = np.dot(v_, n_) * n_
    parallelv_ = np.linalg.norm(parallelv_)
    try:
        return parallelv_ / utau_
    except:
        return 0.00


"""
LINEAR
------
"""

class linear:
    def wdist_norm_coef(row_: int, info_: Info, geom_: Geom):
        aC_ = []; aF_ = []
        col_ = [row_]
        isfluid = bool('fluid' in info_.cells[row_].group)
        for it1, it2 in info_.cells[row_].connect.items():
            col_.append(it1)
            aF_dist = geom_.Sf.scalar(row_, it2) / geom_.dCF.scalar(row_, it1)
            if isfluid is True:
                aC_.append(aF_dist); aF_.append(-aF_dist)
            else:
                aC_.append(0); aF_.append(0)
        
        return np.sum(aC_), aF_, info_.cells[row_].size, list(np.full(shape=(1,len(col_)), fill_value = row_)[0]), col_
    
    def calc_wdist_norm(info_: Info, geom_: Geom):
        dataA_lil = []; dataB_lil = []; row = []; col = []; row_B = []
        approx = []
        wdist_old = []
        for it1, it2 in info_.cells.items():
            wdist_old.append(it2.wdist)
            row_B.append(it1)
            aC_norm, aF_norm, bC_norm, row_, col_ = linear.wdist_norm_coef(it1, info_, geom_)
            row.extend(row_); col.extend(col_)
            dataA_lil.append(aC_norm); dataA_lil.extend(aF_norm)
            dataB_lil.append(bC_norm)
        
        A_ = sparse.coo_matrix((dataA_lil, (row, col)), dtype = float).tocsr()
        B_ = sparse.coo_matrix((dataB_lil, (row_B, list(np.zeros(shape=(1, len(row_B)), dtype = int)[0]))), dtype = float).tocsr()
        B_arr = B_.toarray()
        n = A_.get_shape()[0]
        x_init = np.zeros((1, n))[0]

        try:
            ilu = sparse.linalg.spilu(A_)
            M_x = lambda x: ilu.solve(x)
            Anorm = sparse.linalg.LinearOperator((n, n), M_x)
            new_norm, exitCode_norm = sparse.linalg.gmres(A_, B_arr, x0 = x_init, M = Anorm, atol = 0.05, tol = 0.05, maxiter = 20); new_norm = np.array(new_norm).ravel()
        except:
            new_norm, exitCode_norm = sparse.linalg.gmres(A_, B_arr, x0 = x_init, atol = 0.05, tol = 0.05, maxiter = 20); new_norm = np.array(new_norm).ravel()            
        
        max_e = np.min([abs(math.ceil(math.log10(it1))) for it1 in new_norm])
        new_norm = np.array([float(str(it1)[:3]) / pow(10, 3 + abs(math.floor(math.log10(it1))) - max_e) for it1 in new_norm])

        print('wdist: %s (%s)\n' % (np.mean(new_norm), exitCode_norm))

        for it1, it2 in info_.cells.items():
            if 'fluid' in it2.group:
                grad_norm = least_square_gradient_norm(it1, new_norm, info_, geom_)
                scalar_grad_norm = np.sqrt(np.sum([x**2 for x in grad_norm]))
                it2.wdist = -scalar_grad_norm + np.sqrt(scalar_grad_norm**2 + 2 * new_norm[it1])

        return
    
    # common linear forms
    def match_SMART(arg: int):
        match arg:
            case 0:
                return [3/4,3/8]
            case 1:
                return [3/4,3/8]
            case 2:
                return [3/4,3/8]
            case _:
                return [1,0]
        
    def SMART(theta_, row_: int, col_: int, geom_: Geom, isbound: bool = False):
        check_func = [lambda x: all([x > 0, x < 1/6]), lambda x: all([x >= 1/6, x < 5/6]), lambda x: all([x >= 5/6, x < 1]), lambda x: x >= 1, lambda x: x <= 0]
        match_arg = lambda x: linear.match_SMART(np.where([it1(x) for it1 in check_func])[0][0])
        if isbound is False:
            dCF_ = geom_.dCF.vec(row_, col_)
            thetaUplus_ = theta_.cunit[col_].current - (2 * np.dot(theta_.cgrad[row_].current, dCF_))
            thetaUmin_ = theta_.cunit[row_].current - (2 * np.dot(theta_.cgrad[col_].current, -1 * dCF_))

            try:
                thetaCplus_ = (theta_.cunit[row_].current - thetaUplus_) / (theta_.cunit[col_].current - thetaUplus_)
            except:
                thetaCplus_ = 1
            try:
                thetaFmin_ = (theta_.cunit[col_].current - thetaUmin_) / (theta_.cunit[row_].current - thetaUmin_)
            except:
                thetaFmin_ = 1

        elif isbound is True:
            dCf_ = geom_.dCf.vec(row_, col_)
            thetaUplus_ = theta_.funit[col_].current - (1.5 * np.dot(theta_.cgrad[row_].current, dCf_))
            thetaUmin_ = theta_.cunit[row_].current - (1.5 * np.dot(theta_.fgrad[col_].current, -1 * dCf_))
            try:
                thetaCplus_ = (theta_.cunit[row_].current - thetaUplus_) / (theta_.funit[col_].current - thetaUplus_)
            except:
                thetaCplus_ = 1
            try:
                thetaFmin_ = (theta_.funit[col_].current - thetaUmin_) / (theta_.cunit[row_].current - thetaUmin_)
            except:
                thetaFmin_ = 1          

        lkplus_ = match_arg(thetaCplus_); lkmin_ = match_arg(thetaFmin_)
        return lkplus_, lkmin_

    def peclet_bounding(aF_conv__, mdot__, cp_):
        check = mdot__ * cp_ / aF_conv__
        if bool(check > 2):
            return 2 / check
        return 1

    def NWF(theta_, row_: int, gamma_: str, theta_turb: str, pool_: Pool, info_: Info, geom_: Geom, user_: User, cp_: float = 1, issolid: bool = False, iswall: bool = False):
        aC_ = []; aF_ = []; bC_= []
        col_ = [row_]
        theta_none = lambda x: 1
        theta_k_calc = lambda x: calc_F1(x, pool_, info_) + 1
        theta_w_calc = lambda x: (2 - 1.186) * calc_F1(x, pool_, info_) + 1.186
        turb_func = {'none': theta_none, 'tke': theta_k_calc, 'stke': theta_w_calc}
        mu_calc = lambda x, y: (Prop.calc_mu(pool_.P.cunit[x].current, pool_.T.cunit[x].current) + calc_mut(x, pool_, info_)) / turb_func[y](x)
        k_calc = lambda x, y: (Prop.calc_k(pool_.P.cunit[x].current, pool_.T.cunit[x].current) + calc_kt(x, pool_, info_)) / turb_func[y](x)
        gamma_func = {'mu': mu_calc, 'k': k_calc}
        for it1, it2 in info_.cells[row_].connect.items():
            col_.append(it1)
            lkplus_, lkmin_ = linear.SMART(theta_, row_, it1, geom_)
            vf_ = np.array([pool_.u.funit[it2].current, pool_.v.funit[it2].current, pool_.w.funit[it2].current])
            rhoC_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
            rhoF_ = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = 0)
            rho_ = (rhoC_ + rhoF_) / 2
            Sf_ = geom_.Sf.vec(row_, it2)
            mdot_ = rho_ * np.dot(vf_, Sf_)
            dCF_ = geom_.dCF.vec(row_, it1)
            dCF_val = geom_.dCF.scalar(row_, it1); dFf_val = geom_.dCf.scalar(it1, it2)
            gC_ = dFf_val / dCF_val
            gamma_f = gC_ * gamma_func[gamma_](row_, theta_turb) + (1 - gC_) * gamma_func[gamma_](it1, theta_turb)
            ECf_val = geom_.Ef.scalar(row_, it2)
            TCf_ = geom_.Tf.vec(row_, it2)
            thetaUplus_ = theta_.cunit[it1].current - (2 * np.dot(theta_.cgrad[row_].current, dCF_))
            thetaUmin_ = theta_.cunit[row_].current - (2 * np.dot(theta_.cgrad[it1].current, -1 * dCF_))
            # diffusion
            aF_diff = (gamma_f / dCF_val) * ECf_val
            bC_diff = gamma_f * np.dot(theta_.cgrad[it1].current, TCf_)
            if issolid is False:
                #bound_conv_factor = linear.peclet_bounding(aF_diff, mdot_, cp_)
                #mdot_ = mdot_ * bound_conv_factor
                # convection
                
                aC_conv = np.max([mdot_, 0]) * lkplus_[0] - np.max([-mdot_, 0]) * lkmin_[1]
                aF_conv = np.max([mdot_, 0]) * lkplus_[1] - np.max([-mdot_, 0]) * lkmin_[0]
                bC_conv = -np.max([mdot_, 0]) * (1 - lkplus_[0] - lkplus_[1]) * thetaUplus_ + \
                          np.max([-mdot_, 0]) * (1 - lkmin_[0] - lkmin_[1]) * thetaUmin_
                
                # aC_conv = mdot_ * lkplus_[0] - mdot_ * lkmin_[1]
                # aF_conv = mdot_ * lkplus_[1] - mdot_ * lkmin_[0]
                # bC_conv = -mdot_ * (1 - lkplus_[0] - lkplus_[1]) * thetaUplus_ + \
                #           mdot_ * (1 - lkmin_[0] - lkmin_[1]) * thetaUmin_

                # aC_conv = np.max([mdot_, 0]); aF_conv = np.max([-mdot_, 0]); bC_conv = 0
                # aC_conv = 0; aF_conv = 0; bC_conv = 0
                # aF_diff = 0
                if iswall is False:
                    aC_.append((cp_ * aC_conv) + aF_diff); aF_.append((cp_ * aF_conv) - aF_diff); bC_.append(cp_ * bC_conv)
                elif iswall is True:
                    aC_.append((cp_ * aC_conv)); aF_.append((cp_ * aF_conv)); bC_.append(cp_ * bC_conv)
            elif issolid is True:
                gamma1_ = user_.solid_k[info_.cells[row_].group[0]]
                gamma2_ = user_.solid_k[info_.cells[it1].group[0]]
                dCf_ = geom_.dCf.scalar(row_, it2); dFf_ = geom_.dCf.scalar(it1, it2)
                gC_ = dFf_ / (dCf_ + dFf_); gamma_f = gamma1_ * gamma2_ / (dCf_ * gamma2_ + dFf_ * gamma1_)
                aF_diff = gamma_f * ECf_val / dCF_val
                aC_.append(aF_diff); aF_.append(-aF_diff); bC_.append(0)
        if len(list(info_.cells[row_].conj.keys())) > 0:
            col_.append(list(info_.cells[row_].conj.keys())[0])
            aF_.append(0)
        return np.sum(aC_), aF_, np.sum(bC_), list(np.full(shape=(1,len(col_)), fill_value = row_)[0]), col_

    def NWF_diffusion_only(theta_, row_: int, gamma_: float, pool_: Pool, info_: Info, geom_: Geom, cp_: float = 1, issolid: bool = False):
        aC_ = []; aF_ = []; bC_= []
        col_ = [row_]
        for it1, it2 in info_.cells[row_].connect.items():
            col_.append(it1)
            lkplus_, lkmin_ = linear.SMART(theta_, row_, it1, geom_)
            vf_ = np.array([pool_.u.funit[it2].current, pool_.v.funit[it2].current, pool_.w.funit[it2].current])
            rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
            Sf_ = geom_.Sf.vec(row_, it2)
            mdot_ = rho_ * np.dot(vf_, Sf_)
            dCF_ = geom_.dCF.vec(row_, it1); dCF_val = geom_.dCF.scalar(row_, it1)
            ECf_val = geom_.Ef.scalar(row_, it2)
            TCf_ = geom_.Tf.vec(row_, it2)
            thetaUplus_ = theta_.cunit[it1].current - (2 * np.dot(theta_.cgrad[row_].current, dCF_))
            thetaUmin_ = theta_.cunit[row_].current - (2 * np.dot(theta_.cgrad[it1].current, -1 * dCF_))
            # diffusion
            #aC_diff = -(gamma_ / dCF_val) * ECf_val
            aF_diff = (gamma_ / dCF_val) * ECf_val
            bC_diff = gamma_ * np.dot(theta_.cgrad[it1].current, TCf_)
            if issolid is False:
                # convection
                aC_conv = np.max(mdot_, 0) * lkplus_[0] - np.max(-mdot_, 0) * lkmin_[1]
                aF_conv = np.max(mdot_, 0) * lkplus_[1] - np.max(-mdot_, 0) * lkmin_[0]
                bC_conv = np.max(mdot_, 0) * (1 - lkplus_[0] - lkplus_[1]) * thetaUplus_ - \
                          np.max(-mdot_, 0) * (1 - lkmin_[0] - lkmin_[1]) * thetaUmin_

                aC_.append(aF_diff); aF_.append(-aF_diff); bC_.append(0)
                #aC_.append(aF_diff); aF_.append(- aF_diff); bC_.append(bC_diff)
            elif issolid is True:
                aC_.append(aF_diff); aF_.append(-aF_diff); bC_.append(bC_diff)
        return np.sum(aC_), aF_, np.sum(bC_), list(np.full(shape=(1,len(col_)), fill_value = row_)[0]), col_

    def relax_trans(theta_, pool_: Pool, info_: Info, aC_: float, row_: int, lambda_: float, delta_t: float, cp_: float = 1, cp_last: float = 1, isinit_: bool = False):
        aC_t = ((1 - lambda_) * aC_ * cp_ / lambda_) 
        bC_t = ((1 - lambda_) * aC_ * theta_.cunit[row_].current * cp_ / lambda_)
        #bC_t = ((1 - lambda_ ) * aC_ * theta_.cunit[row_].current / lambda_)
        #aC_t = ((1 - lambda_) / lambda_) * aC_
        return aC_t, bC_t #+ pow(10, -16)
        # return  0, 0

    def temporal_conv(theta_, pool_: Pool, info_: Info, row_: int, delta_t: float, rho_, rho_last, cp_: float = 1, cp_last: float = 1, isinit_: bool = False):
        delta_2t = delta_t
        if isinit_ is False:
            delta_2t = delta_t * 2
        aC_temporal = (rho_ * info_.cells[row_].size * cp_ / delta_2t)
        bC_temporal = (rho_last * info_.cells[row_].size * theta_.cunit[row_].last * cp_last / delta_2t)
        return aC_temporal, bC_temporal #* (1 + 1e-12)
        # return 0, 0

class Boundary:
    # boundary specific coef calcs
    def noslip_momentum(theta_, row_: int, col_: int, aC_nwf_, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User):
        aC_noslip = float(0); bC_noslip = float(0)
        mu_ =  Prop.calc_mu(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current)
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        wdist_ = info_.cells[row_].wdist
        v_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
        n_ = np.array([0,1,0], dtype = float) #-1 * geom_.Sf.norm(row_, col_)
        parallelv_ = np.dot(v_, n_) * n_
        parallelv_ = np.linalg.norm(parallelv_)
        ustar_ = calc_u_star(row_, col_, pool_, mu_, rho_, parallelv_, wdist_)
        uplus_ = calc_u_plus(row_, col_, pool_, info_, geom_, mu_, rho_, parallelv_, wdist_)
        Sf_ = geom_.Sf.vec(row_, col_)
        match theta_.name:
            case 'u':
                aC_noslip = rho_ * (ustar_ / uplus_) * (1 - n_[0]**2)
                bC_noslip = rho_ * (ustar_ / uplus_) * (v_[0] * (1 - n_[0]**2) + (v_[1] * n_[1] * n_[0]) + (v_[2] * n_[2] * n_[0])) #- pool_.P.funit[col_].current * Sf_[0]
            case 'v':
                aC_noslip = rho_ * (ustar_ / uplus_) * (1 - n_[1]**2)
                bC_noslip = rho_ * (ustar_ / uplus_) * ((v_[0] * n_[0] * n_[1]) + v_[1] * (1 - n_[1]**2) + (v_[2] * n_[2] * n_[1])) #- pool_.P.funit[col_].current * Sf_[1]
            case 'w':
                aC_noslip = rho_ * (ustar_ / uplus_) * (1 - n_[2]**2)
                bC_noslip = rho_ * (ustar_ / uplus_) * ((v_[0] * n_[0] * n_[2]) + (v_[1] * n_[1] * n_[2]) + v_[2] * (1 - n_[2]**2)) #- pool_.P.funit[col_].current * Sf_[2]
        return aC_noslip, bC_noslip

    def inlet_momentum(theta_, row_: int, col_: int, aC_nwf, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User):
        match theta_.name:
            case 'u':
                which = 0
            case 'v':
                which = 1
            case 'w':
                which = 2

        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        dCf_ = geom_.dCf.vec(row_, col_); dCf_val = geom_.dCf.scalar(row_, col_)
        n_ = geom_.Sf.norm(row_, col_)
        Sf_ = geom_.Sf.vec(row_, col_)
        vf_ = np.array([pool_.u.funit[col_].current, pool_.v.funit[col_].current, pool_.w.funit[col_].current])
        mdot_ = rho_ * np.dot(vf_, Sf_)
        gradf_ = theta_.cgrad[row_].current - (np.dot(theta_.cgrad[row_].current, n_)) * n_
        right1 = np.dot(gradf_, dCf_)

        aC_conv = mdot_
        bC_conv = -mdot_ * right1 #theta_.funit[col_].current 

        return aC_conv, bC_conv #- pool_.P.funit[col_].current * Sf_[which]

    def outlet_momentum(theta_, row_: int, col_: int, aC_nwf, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User):
        match theta_.name:
            case 'u':
                which = 0
            case 'v':
                which = 1
            case 'w':
                which = 2

        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        dCf_ = geom_.dCf.vec(row_, col_); dCf_val = geom_.dCf.scalar(row_, col_)
        n_ = geom_.Sf.norm(row_, col_)
        Sf_ = geom_.Sf.vec(row_, col_)
        vf_ = np.array([pool_.u.funit[col_].current, pool_.v.funit[col_].current, pool_.w.funit[col_].current])
        mdot_ = rho_ * np.dot(vf_, Sf_)
        gradf_ = theta_.cgrad[row_].current - (np.dot(theta_.cgrad[row_].current, n_)) * n_
        right1 = np.dot(gradf_, dCf_)
        
        aC_conv = mdot_
        bC_conv = -mdot_ * right1 #theta_.funit[col_].current 

        return aC_conv, bC_conv #-pool_.P.funit[col_].current * Sf_[which]

    def inlet_outlet_energy(theta_, row_: int, col_: int, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User, Tref_: float = 300, cp_: float = 1):
        v_ = np.array([pool_.u.funit[col_].current, pool_.v.funit[col_].current, pool_.w.funit[col_].current])
        Tref_ = pool_.T.funit[col_].current + np.dot(v_, v_) / (2 * cp_)
        pool_.T.funit[col_].new = Tref_; pool_.cp.funit[col_].new = Prop.calc_cp(pool_.P.funit[col_].current, Tref_)
        aC_io, bC_io = Boundary.dirichlet_energy(theta_, row_, col_, gamma_, info_, pool_, geom_, user_, Tref_ = Tref_, cp_ = cp_)
        return aC_io, bC_io

    def symmetry_fluid(theta_, row_: int, col_: int, aC_nwf, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User, cp_: float = 1, Tref_: float = 300):
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        dCf_ = geom_.dCf.vec(row_, col_); dCf_val = geom_.dCf.scalar(row_, col_)
        n_ = geom_.Sf.norm(row_, col_)
        Sf_ = geom_.Sf.vec(row_, col_)
        vf_ = np.array([pool_.u.funit[col_].current, pool_.v.funit[col_].current, pool_.w.funit[col_].current])
        mdot_ = rho_ * np.dot(vf_, Sf_)
        gradf_ = pool_.T.cgrad[row_].current - (np.dot(pool_.T.cgrad[row_].current, n_)) * n_
        right1 = np.dot(gradf_, dCf_)

        aC_conv = mdot_ * cp_
        bC_conv = -mdot_ * cp_ * right1 #theta_.cunit[row_].current

        return aC_conv, bC_conv

    def symmetry_solid(theta_, row_: int, col_: int, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User, cp_: float = 1):
        Sf_ = geom_.Sf.vec(row_, col_)
        n_ = geom_.Sf.norm(row_, col_)
        gradf_ = pool_.T.cgrad[row_].current - (np.dot(pool_.T.cgrad[row_].current, n_)) * n_
        bC_sym = gamma_ * np.dot(gradf_, Sf_)

        return 0, bC_sym

    def dirichlet_energy(theta_, row_: int, col_: int, aC_nwf, gamma_: float, info_: Info, pool_: Pool, geom_: Geom, user_: User, cp_: float = 1, Tref_: float = 300):
        SCf_ = geom_.Sf.vec(row_, col_)
        SCf_val = geom_.Sf.scalar(row_, col_)
        dCf_val = geom_.dCf.scalar(row_, col_)
        eCf_ = geom_.dCf.norm(row_, col_)
        mdot_ = np.dot(np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current]), SCf_) * Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        aC_dir = gamma_ * (SCf_val / dCf_val)
        bC_dir = (-mdot_ * cp_ + gamma_ * SCf_val / dCf_val) * Tref_
        return aC_dir, bC_dir

    def von_neumann(mdot_: float, gamma_: float, q_: float, dCf_val: float, SCf_val: float, cp_: float = 1):
        # temp only
        aC_neu = 0
        bC_neu = q_ * (SCf_val)
        return aC_neu, bC_neu
    
    # code hamb, namespace use
    # qglass is enabled
    # conjugate, namespace used
    # # either htop or hbot
    # # qs2s is enabled
    # # if solid, qabs and qs2s is enabled
    # # if not conjugate, symmetry bound coef used 
    
    def htopconjugateFluidHTF(fluid_: int, solid_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom, channel_length: float):
        face_groups = info_.faces[face_].group; qh_ = 0; k_ = 0; h_ = 0
        # htop, hbot, s2s_glass1, s2s_abs2, s2s_abs3, qabs
        # positive to solid, none to fluid -> s2s_glass1, s2s_abs2, s2s_abs3, qabs
        # h always positive for fluid, negative for solid

        dCf_fluid = geom_.dCf.scalar(fluid_, face_)
        Tf_ = (pool_.T.cunit[fluid_].current + pool_.T.cunit[solid_].current) / 2
        v_ = np.array([pool_.u.cunit[fluid_].current, pool_.v.cunit[fluid_].current, pool_.w.cunit[fluid_].current])
        n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(solid_, face_)
        parallelv_ = np.dot(v_, n_) * n_
        parallelv_ = np.linalg.norm(parallelv_)
        k_ = Prop.calc_k(pool_.P.funit[face_].current, Tf_)
        cp_ = Prop.calc_cp(pool_.P.funit[face_].current, Tf_)
        rho_ = Prop.calc_rho(pool_.P.funit[face_]          .current, Tf_, working_density_ = 0)
        mu_ = Prop.calc_mu(pool_.P.funit[face_].current, Tf_)
        alpha_ = 0.142 / (1.293 * 1006.43)
        ReL_ = (rho_ / mu_) * np.abs(parallelv_) * math.sqrt(info_.faces[face_].size)
        n_ = geom_.Sf.norm(solid_, face_)
        cos_incl_ = np.abs(np.dot(n_, np.array([0, 1, 0])))
        Pr_ = 0.71 #mu_ / (rho_ * alpha_)

        Nu_ = float(0) # positive for fluid, negative for solid
        
        """
        if info_.cells[fluid_].loc[1] < 0.2 * channel_length:
            Nu_ = 0.1
        elif info_.cells[fluid_].loc[1] <= 0.8 * channel_length:
            Nu_ = 0.5
        else:
            Nu_ = 0.8
        """

        if all([Pr_ >= 0.6, ReL_ < 5 * pow(10, 5)]) is True:
            Nu_ += 0.332 * pow(ReL_, 0.5) * pow(Pr_, 1/3)
        elif all([all([Pr_ >= 0.6, Pr_ <= 60]), all([ReL_ >= pow(10, 5), ReL_ <= pow(10, 8)])]) is True:
            Nu_ += (0.037 * pow(ReL_, 0.8) - 871) * pow(Pr_, 1/3)

        try:
            RaL_ = 9.81 * cos_incl_ * (1 / Tf_) * (Tf_ - pool_.T.cunit[fluid_].current) * pow(info_.faces[face_].size, 3/2) / (mu_ * alpha_ / rho_)
            
            if RaL_ < pow(10, 4):
                Nu_ += 0.68 + 0.67 * pow(RaL_, 1/4) / pow(1 + pow(0.492 / Pr_, 9/16), 4/9)
            elif all([all([RaL_ >= pow(10, 4), RaL_ <= pow(10, 7)]), Pr_ >= 0.7]) is True:
                Nu_ += 0.54 * pow(RaL_, 1/4)
            elif all([RaL_ > pow(10, 7), RaL_ <= pow(10, 11)]):
                Nu_ += 0.15 * pow(RaL_, 1/3)
        except:
            pass

        #else:
        #    Nu_ = 0.1 + info_.cells[fluid_].loc[1] * (0.01)
        #if info_.cells[fluid_].loc[1] < 0.2:
        #    Nu_ = 0.8 * Nu_

        h_ = Nu_ * k_ / dCf_fluid
        #qh_ = np.max([h_ * (pool_.T.cunit[solid_].current - pool_.T.cunit[fluid_].current), 0])
        #print(face_groups, pool_.T.cunit[solid_].current - pool_.T.cunit[fluid_].current, k_, math.sqrt(info_.faces[face_].size), h_, qh_)
        return h_

    def hbotconjugateFluidHTF(fluid_: int, solid_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom, channel_length: float):
        face_groups = info_.faces[face_].group; qh_ = 0; k_ = 0; h_ = 0
        # htop, hbot, s2s_glass1, s2s_abs2, s2s_abs3, qabs
        # positive to solid, none to fluid -> s2s_glass1, s2s_abs2, s2s_abs3, qabs
        # h always positive for fluid, negative for solid
        dCf_fluid = geom_.dCf.scalar(fluid_, face_)
        Tf_ = (pool_.T.cunit[fluid_].current + pool_.T.cunit[solid_].current) / 2
        n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(solid_, face_)
        v_ = np.array([pool_.u.cunit[fluid_].current, pool_.v.cunit[fluid_].current, pool_.w.cunit[fluid_].current])
        parallelv_ = np.dot(v_, n_) * n_
        parallelv_ = np.linalg.norm(parallelv_)
        k_ = Prop.calc_k(pool_.P.funit[face_].current, Tf_)
        cp_ = Prop.calc_cp(pool_.P.funit[face_].current, Tf_)
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, Tf_, working_density_ = 0)
        mu_ = Prop.calc_mu(pool_.P.funit[face_].current, Tf_)
        alpha_ = 0.142 / (1.293 * 1006.43)
        ReL_ = (rho_ / mu_) * np.abs(parallelv_) * math.sqrt(info_.faces[face_].size)  
        cos_incl_ = np.abs(np.dot(n_, np.array([0, 1, 0])))
        Pr_ = mu_ / (rho_ * alpha_)

        Nu_ = float(0) # positive for fluid, negative for solid

        if all([Pr_ >= 0.6, ReL_ < 5 * pow(10, 5)]) is True:
            Nu_ += 0.332 * pow(ReL_, 0.5) * pow(Pr_, 1/3)
        elif all([all([Pr_ >= 0.6, Pr_ <= 60]), all([ReL_ >= 5 * pow(10, 5), ReL_ <= pow(10, 8)])]) is True:
            Nu_ += (0.037 * pow(ReL_, 0.8) - 871) * pow(Pr_, 1/3)

        try:
            RaL_ = 9.81 * cos_incl_ * (1 / Tf_) * (Tf_ - pool_.T.cunit[fluid_].current) * pow(info_.faces[face_].size, 3/2) / (mu_ * alpha_ / rho_)
            
            if RaL_ < pow(10, 4):
                Nu_ += 0.68 + 0.67 * pow(RaL_, 1/4) / pow(1 + pow(0.492 / Pr_, 9/16), 4/9)
            elif all([all([RaL_ >= pow(10, 4), RaL_ <= pow(10, 9)]), Pr_ >= 0.7]) is True:
                Nu_ += 0.52 * pow(RaL_, 1/5)
        except:
            pass

        h_ = Nu_ * k_ / dCf_fluid
        #qh_ = np.max([h_ * (pool_.T.cunit[solid_].current - pool_.T.cunit[fluid_].current), 0])
        return h_

    def conjugateFluidGlass(fluid_: int, solid_: int, face_: int, gamma_: float, pool_: Pool, info_: Info, geom_: Geom, user_: User, channel_length: float,
                            cp_: float = 1, qhwg_: float = 0):
        # temp only
        # amb += neumann qglass - mixed hamb
        # calc glaze hamb
        # eps glass -> 0.89

        # h sky loss
        Tsolid_ = pool_.T.cunit[solid_].current
        Tfilm_ = (pool_.T.cunit[solid_].current + pool_.T.cunit[fluid_].current) / 2
        Tsky_ = 0.0552 * pow(300, 1.5)
        try:
            hsky_ = 5.67 * pow(10, -8) * 0.89 * (Tsolid_ + Tsky_) * (Tsolid_**2 + Tsky_**2) * (Tsolid_ - Tsky_) / (Tsolid_ - 300)
        except:
            hsky_ = 0
        n_ = geom_.Sf.norm(solid_, face_)
        cos_incl_ = np.dot(n_, np.array([0, 1, 0]))
        k_ = Prop.calc_k(pool_.P.funit[face_].current, Tfilm_)
        cp_ = Prop.calc_cp(pool_.P.funit[face_].current, Tfilm_)
        mu_ = Prop.calc_mu(pool_.P.funit[face_].current, Tfilm_)
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, Tfilm_, working_density_ = 0)
        alpha_ = 0.142 / (1.293 * 1006.43)
        Pr_ = 0.71 #mu_ / (rho_ * alpha_)

        NuN_ = 0
        RaL_ = 9.81 * cos_incl_ * (1 / Tfilm_) * (Tfilm_ - 300) * pow(info_.faces[face_].size, 2/3) / (mu_ * alpha_ / rho_)
        if all([all([RaL_ >= pow(10, 4), RaL_ <= pow(10, 7)]), Pr_ >= 0.7]) is True:
            NuN_ = 0.54 * pow(RaL_, 1/4)
        elif all([RaL_ > pow(10, 7), RaL_ <= pow(10, 11)]) is True:
            NuN_ = 0.15 * pow(RaL_, 1/3)
        hconv_ = NuN_ * k_ / math.sqrt(info_.faces[face_].size)

        h_fluid_ = Boundary.hbotconjugateFluidHTF(fluid_, solid_, face_, pool_, info_, geom_, channel_length) - hsky_

        SCf_solid = geom_.Sf.scalar(solid_, face_)
        dCf_solid = geom_.dCf.scalar(solid_, face_)
        dCf_fluid = geom_.dCf.scalar(fluid_, face_)
        dCF_fluid = dCf_solid + dCf_fluid

        qh_solid_ = (user_.solid_q['glass'] + qhwg_) * info_.faces[face_].size
        qh_fluid_ = h_fluid_ * info_.faces[face_].size

        n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(fluid_, face_)
        v_ = np.array([pool_.u.cunit[fluid_].current, pool_.v.cunit[fluid_].current, pool_.w.cunit[fluid_].current])
        parallelv_ = np.dot(v_, n_) * n_
        parallelv_ = np.linalg.norm(parallelv_)
        wdist_ = info_.cells[fluid_].wdist
        ustar_ = calc_u_star(fluid_, face_, pool_, mu_, rho_, parallelv_, wdist_)
        dstar_ = (wdist_ * ustar_ * rho_) / mu_
        Tstar_ = (h_fluid_ * SCf_solid * (pool_.T.cunit[solid_].current - pool_.T.cunit[fluid_].current)) / (rho_ * cp_ * ustar_)
        betaPr_ = pow(3.85 * pow(Pr_, 1/3) - 1.3, 2) + 2.12 * math.log10(Pr_)
        Gamma_ = (0.01 * (Pr_ * dstar_)**4) / (1 + 5 * Pr_**3 * dstar_)
        Tplus_ = (Pr_ * dstar_ * math.exp(-Gamma_)) + (2.12 * math.log10(1 + dstar_) + betaPr_) * math.exp(-1/Gamma_)

        wall_term = rho_ * cp_ * ustar_ * info_.faces[face_].size / Tplus_
            
        aC_fluid = wall_term #+ qh_fluid_
        aF_fluid = -wall_term #- qh_fluid_
        bC_fluid = 0
        aC_solid = wall_term - hsky_ * info_.faces[face_].size #+ qh_fluid_
        aF_solid = -wall_term #- qh_fluid_
        bC_solid = qh_solid_ - hsky_ * info_.faces[face_].size * 300

        return aC_fluid, aF_fluid, bC_fluid, aC_solid, aF_solid, bC_solid
        # return 0, 0, 0, 0, 0, 0

    def conjugateFluidAbs(fluid_: int, solid_: int, face_: int, gamma_: float, pool_: Pool, info_: Info, geom_: Geom, user_: User, channel_length,
                          cp_: float = 1, qhwg_: float = 0):
        h_fluid_ = Boundary.htopconjugateFluidHTF(fluid_, solid_, face_, pool_, info_, geom_, channel_length)
        qh_solid_ = (user_.solid_q['abs'] + qhwg_) * info_.faces[face_].size
        qh_fluid_ = h_fluid_ * info_.faces[face_].size
        
        SCf_solid = geom_.Sf.scalar(solid_, face_)
        dCf_solid = geom_.dCf.scalar(solid_, face_)
        dCf_fluid = geom_.dCf.scalar(fluid_, face_)
        dCF_fluid = dCf_solid + dCf_fluid

        Tfilm_ = (pool_.T.cunit[solid_].current + pool_.T.cunit[fluid_].current) / 2
        k_ = Prop.calc_k(pool_.P.funit[face_].current, Tfilm_)
        cp_ = Prop.calc_cp(pool_.P.funit[face_].current, Tfilm_)
        mu_ = Prop.calc_mu(pool_.P.funit[face_].current, Tfilm_)  
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, Tfilm_, working_density_ = 0)
        alpha_ = k_ / (rho_ * cp_)
        Pr_ = 0.71 #mu_ / (rho_ * alpha_)

        n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(fluid_, face_)
        v_ = np.array([pool_.u.cunit[fluid_].current, pool_.v.cunit[fluid_].current, pool_.w.cunit[fluid_].current])
        parallelv_ = np.dot(v_, n_) * n_
        parallelv_ = np.linalg.norm(parallelv_)   
        wdist_ = info_.cells[fluid_].wdist
        ustar_ = calc_u_star(fluid_, face_, pool_, mu_, rho_, parallelv_, wdist_)
        dstar_ = (wdist_ * ustar_ * rho_) / mu_
        Tstar_ = (h_fluid_ * SCf_solid * (pool_.T.cunit[solid_].current - pool_.T.cunit[fluid_].current)) / (rho_ * cp_ * ustar_)
        betaPr_ = pow(3.85 * pow(Pr_, 1/3) - 1.3, 2) + 2.12 * math.log10(Pr_)
        Gamma_ = (0.01 * (Pr_ * dstar_)**4) / (1 + 5 * Pr_**3 * dstar_)
        Tplus_ = (Pr_ * dstar_ * math.exp(-Gamma_)) + (2.12 * math.log10(1 + dstar_) + betaPr_) * math.exp(-1/Gamma_)

        wall_term = rho_ * cp_ * ustar_ * info_.faces[face_].size / Tplus_
        
        aC_fluid = wall_term #+ qh_fluid_
        aF_fluid = -wall_term #- qh_fluid_
        bC_fluid = 0
        aC_solid = wall_term #+ qh_fluid_
        aF_solid = -wall_term #- qh_fluid_
        bC_solid = qh_solid_

        """
        Twall_ = np.min([pool_.T.cunit[fluid_].current + (Tstar_ * Tplus_), (pool_.T.cunit[solid_].current + pool_.T.cunit[fluid_].current)/2])
        pool_.T.funit[face_].new = Twall_

        aC_fluid, bC_fluid = Boundary.dirichlet_energy(pool_.T, fluid_, face_, gamma_, info_, pool_, geom_, user_, Tref_ = Twall_, cp_ = cp_)
        aC_solid, bC_solid = Boundary.dirichlet_energy(pool_.T, solid_, face_, user_.solid_k['abs'], info_, pool_, geom_, user_, Twall_, user_.solid_cp['abs'])
        """

        return aC_fluid, aF_fluid, bC_fluid, aC_solid, aF_solid, bC_solid
        # return 0, 0, 0, 0, 0, 0

    def noslip_pcorrect(row_: int, col_: int,
                    A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                    lambda_: float, delta_t, pool_: Pool, info_: Info, geom_: Geom, mfstar_: list, isinit: bool = False):
        rho_ = Prop.calc_rho(pool_.T.funit[col_].current, pool_.P.funit[col_].current, working_density_ = 0)
        rhoDiff_ = Prop.calc_rho(pool_.T.funit[col_].current, pool_.P.funit[col_].current, working_density_ = pool_.working_density)
        SCf_ = geom_.Sf.vec(row_, col_)
        DauC_ = Pcorrect.calc_bound_Dauf(row_, col_, pool_, info_, geom_,
                        A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        aC_noslip = rho_ * DauC_ * info_.cells[row_].size
        return rho_ * DauC_ * info_.cells[row_].size, rho_ * DauC_ * info_.cells[row_].size * rhoDiff_ * 287.053 * (pool_.T.cunit[row_].current - 300) #pool_.P.funit[col_].current

    def inlet_pcorrect(row_: int, col_: int,
                    A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                    lambda_: float, delta_t, pool_: Pool, info_: Info, geom_: Geom, mfstar_: list, isinit: bool = False):

        rho_ = Prop.calc_rho(pool_.P.funit[col_].current, pool_.T.funit[col_].current, working_density_ = 0)
        Dauf_ = Pcorrect.calc_bound_Dauf(row_, col_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        mbstar_ = Pcorrect.calc_bound_mfstar(row_, col_, rho_, Dauf_, info_, pool_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        vf_ = np.array([pool_.u.funit[col_].new, pool_.v.funit[col_].new, pool_.w.funit[col_].new])
        # mfstar_.append(mbstar_)
        # aC_io = rho_ * mbstar_ * Dauf_ * info_.cells[row_].size / (mbstar_ - Dauf_ * info_.cells[row_].size * np.dot(rho_ * vf_, rho_ * vf_))
        return 0, mbstar_
    
    def outlet_pcorrect(row_: int, col_: int,
                    A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                    lambda_: float, delta_t, pool_: Pool, info_: Info, geom_: Geom, mfstar_: float, isinit: bool = False):

        rho_ = Prop.calc_rho(pool_.P.funit[col_].current, pool_.T.funit[col_].current, working_density_ = 0)
        Dauf_ = Pcorrect.calc_bound_Dauf(row_, col_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        mbstar_ = Pcorrect.calc_bound_mfstar(row_, col_, rho_, Dauf_, info_, pool_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # mfstar_.append(mbstar_)
        return rho_ * Dauf_ * info_.cells[row_].size, mbstar_

    def dirichlet_tke(theta_, row_, face_, gamma_, info_, pool_, geom_, user_, cp_: float = 1):
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        vf_ = np.array([pool_.u.funit[face_].current, pool_.v.funit[face_].current, pool_.w.funit[face_].current])
        # kref_ = pow(0.001, 2) * np.dot(vf_, vf_) / 2
        SCf_val = geom_.Sf.scalar(row_, face_)
        dCf_val = geom_.dCf.scalar(row_, face_)
        dCf_ = geom_.dCf.vec(row_, face_)
        eCf_ = geom_.dCf.norm(row_, face_)
        gradf_ = pool_.k.cgrad[row_].current - (np.dot(pool_.k.cgrad[row_].current, eCf_)) * eCf_
        right1 = np.dot(gradf_, dCf_)
        mdot_ = rho_ * np.dot(vf_, geom_.Sf.vec(row_, face_))
        aC_dir = mdot_ + gamma_ * (SCf_val / dCf_val)
        bC_dir = (-mdot_ + gamma_ * (SCf_val / dCf_val)) * pool_.k.cunit[row_].current #+ (gamma_ * SCf_val / dCf_val) * kref_ - mdot_ * kref_
        return aC_dir, bC_dir

    def symmetry_tke(theta_, row_, face_, gamma_, info_, pool_, geom_, user_, cp_: float = 1):
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        vf_ = np.array([pool_.u.funit[face_].current, pool_.v.funit[face_].current, pool_.w.funit[face_].current])
        # SCf_val = geom_.Sf.scalar(row_, face_)
        # dCf_val = geom_.dCf.scalar(row_, face_)
        dCf_ = geom_.dCf.vec(row_, face_)
        eCf_ = geom_.dCf.norm(row_, face_)
        gradf_ = pool_.k.cgrad[row_].current - (np.dot(pool_.k.cgrad[row_].current, eCf_)) * eCf_
        right1 = np.dot(gradf_, dCf_)
        eCf_ = geom_.dCf.norm(row_, face_)
        mdot_ = rho_ * np.dot(vf_, geom_.Sf.vec(row_, face_))
        aC_dir = mdot_ #gamma_ * (SCf_val / dCf_val)
        bC_dir = -mdot_ * right1 #pool_.k.cunit[row_].current #(gamma_ * SCf_val / dCf_val) * pool_.k.cunit[row_].current - mdot_ * pool_.k.cunit[row_].current
        return aC_dir, bC_dir

    def dirichlet_stke(theta_, row_, face_, gamma_, gamma_t, info_, pool_, geom_, user_, cp_: float = 1):
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        vf_ = np.array([pool_.u.funit[face_].current, pool_.v.funit[face_].current, pool_.w.funit[face_].current])
        # try:
        #     omegaref_ = rho_ * pool_.k.funit[face_].current / gamma_t
        # except:
        #     omegaref_ = 0
        SCf_val = geom_.Sf.scalar(row_, face_)
        dCf_val = geom_.dCf.scalar(row_, face_)
        # eCf_ = geom_.dCf.norm(row_, face_)
        # mdot_ = np.dot(vf_, geom_.Sf.vec(row_, face_))
        # aC_dir = gamma_ * (SCf_val / dCf_val)
        # bC_dir = (gamma_ * SCf_val / dCf_val) * omegaref_ - mdot_ * omegaref_
        # return aC_dir, bC_dir
    
        dCf_ = geom_.dCf.vec(row_, face_)
        eCf_ = geom_.dCf.norm(row_, face_)
        gradf_ = pool_.omega.cgrad[row_].current - (np.dot(pool_.omega.cgrad[row_].current, eCf_)) * eCf_
        right1 = np.dot(gradf_, dCf_)
        eCf_ = geom_.dCf.norm(row_, face_)
        mdot_ = rho_ * np.dot(vf_, geom_.Sf.vec(row_, face_))
        aC_dir = mdot_ + gamma_ * (SCf_val / dCf_val)
        bC_dir = (-mdot_ + gamma_ * (SCf_val / dCf_val)) * pool_.omega.cunit[row_].current #(gamma_ * SCf_val / dCf_val) * pool_.k.cunit[row_].current - mdot_ * pool_.k.cunit[row_].current
        return aC_dir, bC_dir
    
    def symmetry_stke(theta_, row_, face_, gamma_, gamma_t, info_, pool_, geom_, user_, cp_: float = 1):
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        vf_ = np.array([pool_.u.funit[face_].current, pool_.v.funit[face_].current, pool_.w.funit[face_].current])
        # SCf_val = geom_.Sf.scalar(row_, face_)
        # dCf_val = geom_.dCf.scalar(row_, face_)
        dCf_ = geom_.dCf.vec(row_, face_)
        eCf_ = geom_.dCf.norm(row_, face_)
        gradf_ = pool_.omega.cgrad[row_].current - (np.dot(pool_.omega.cgrad[row_].current, eCf_)) * eCf_
        right1 = np.dot(gradf_, dCf_)
        eCf_ = geom_.dCf.norm(row_, face_)
        mdot_ = rho_ * np.dot(vf_, geom_.Sf.vec(row_, face_))
        aC_dir = mdot_ #gamma_ * (SCf_val / dCf_val)
        bC_dir = -mdot_ * right1 #pool_.omega.cunit[row_].current #(gamma_ * SCf_val / dCf_val) * pool_.k.cunit[row_].current - mdot_ * pool_.k.cunit[row_].current
        return aC_dir, bC_dir

class Correct:
    def symmetry_energy_correct(pool_: Pool, geom_: Geom, row_: int, col_: int):
        # dCf_ = geom_.dCf.vec(row_, col_)
        # n_ = geom_.dCf.norm(row_, col_)
        # gradf_ = pool_.T.cgrad[row_].current - (np.dot(pool_.T.cgrad[row_].current, n_)) * n_ 
        pool_.T.funit[col_].new = pool_.T.cunit[row_].new #+ np.dot(gradf_, dCf_)

    def dirichlet_energy_correct(pool_: Pool, geom_: Geom, row_: int, col_: int):
        pool_.T.funit[col_].new = 300

    def dirichlet_momentum_correct(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                    A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                    lambda_: float, delta_t, isinit: bool = False):
        pool_.u.funit[face_].new = 0
        pool_.v.funit[face_].new = 0
        pool_.w.funit[face_].new = 0

    def ambHTF_correct(theta_, row_: int, col_: int):
        theta_.funit[col_].new = (theta_.cunit[row_].new + 300) / 2
        return

    def conjugate_correct(row_: int, col_: int, face_: int, pool_: Pool):
        pool_.T.funit[face_].new = np.min([(pool_.T.cunit[row_].new + pool_.T.cunit[col_].new) / 2, pool_.T.cunit[col_].new])
        return

    def noslip_correct_pcorrect(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                lambda_: float, delta_t: float, isinit: bool = False):
        # SCf_ = geom_.Sf.vec(row_, face_)
        # DauC_ = Pcorrect.calc_bound_Dauf(row_, face_, pool_, info_, geom_,
        #                 A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # Pb_ = np.dot(pool_.P.cgrad[row_].current, SCf_) / DauC_
        # rhoDiff_ = Prop.calc_rho(pool_.T.funit[face_].current, pool_.P.funit[face_].current, working_density_ = pool_.working_density)
        pool_.Pcor.funit[face_].new = pool_.Pcor.cunit[row_].new
        pool_.P.funit[face_].new = pool_.P.cunit[row_].new #rhoDiff_ * 287.053 * (pool_.T.cunit[row_].current - 300) + pool_.Pcor.cunit[row_].new #pool_.P.cunit[row_].new #pool_.P.cunit[row_].new

    def inlet_correct_pcorrect(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                            A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                            lambda_: float, delta_t: float, isinit: bool = False):

        # Sf_ = geom_.Sf.vec(row_, face_)
        # n_ = geom_.Sf.norm(row_, face_) * -1
        
        # rho_ = Prop.calc_rho(pool_.P.funit[face_].current, pool_.T.funit[face_].current, working_density_ = 0)
        # Dauf_ = Pcorrect.calc_bound_Dauf(row_, face_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # mbstar_ = Pcorrect.calc_bound_mfstar(row_, face_, rho_, Dauf_, info_, pool_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # vf_ = np.array([pool_.u.funit[face_].new, pool_.v.funit[face_].new, pool_.w.funit[face_].new])
        # mbcor_ = rho_ * mbstar_ * Dauf_ * pool_.Pcor.cunit[face_].new * info_.cells[row_].size / (mbstar_ - Dauf_ * pow(rho_, 2) * np.dot(vf_, vf_) * info_.cells[row_].size)

        # pcorf_ = (-mbstar_ * mbcor_ / (rho_ * pow(np.dot(n_, Sf_), 2)))

        pool_.Pcor.funit[face_].new = pool_.Pcor.cunit[row_].new #pcorf_
        pool_.P.funit[face_].new = pool_.P.cunit[row_].new #+ pcorf_

        # pool_.Pcor.funit[face_].new = pool_.Pcor.cunit[row_].new
        # pool_.P.funit[face_].new = pool_.P.cunit[row_].new

    def outlet_correct_pcorrect(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                            A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                            lambda_: float, delta_t: float, isinit: bool = False):
 
        pool_.Pcor.funit[face_].new = 0 #pool_.Pcor.cunit[row_].new
        pool_.P.funit[face_].new = 0 #pool_.P.cunit[row_].new

    def v_correct_doublestar(row_: int, pool_: Pool, info_: Info, geom_: Geom,
                            A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                            lambda_: float, delta_t: float, isinit: bool = False):
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        aC_u = np.sum(A_u.getrow(row_).toarray())
        aC_v = np.sum(A_v.getrow(row_).toarray())
        aC_w = np.sum(A_w.getrow(row_).toarray())

        DC_u = 1 / aC_u #+ rho_ * info_.cells[row_].size / delta_t2)
        DC_v = 1 / aC_v #+ rho_ * info_.cells[row_].size / delta_t2)
        DC_w = 1 / aC_w #+ rho_ * info_.cells[row_].size / delta_t2)

        DC_ = np.array([DC_u, DC_v, DC_w]) * info_.cells[row_].size
        gradpcorC = pool_.Pcor.cgrad[row_].current
        
        bC_func0 = lambda x: -gradpcorC[x] * DC_[x]

        for it1 in [0,1,2]:
            v_l[it1].cunit[row_].new = v_l[it1].cunit[row_].new + bC_func0(it1)

    def v_mdot_correct_doublestar_face(row_: int, col_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                        A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                        lambda_, delta_t, isinit: bool = False):
        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}

        aC_u = np.sum(A_u.getrow(row_).toarray())
        aC_v = np.sum(A_v.getrow(row_).toarray())
        aC_w = np.sum(A_w.getrow(row_).toarray())
        
        aF_u = np.sum(A_u.getrow(col_).toarray())
        aF_v = np.sum(A_v.getrow(col_).toarray())
        aF_w = np.sum(A_w.getrow(col_).toarray())

        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        interpolate_lin = lambda x, y: gC_ * x + (1 - gC_) * y
        gradpcorf = interpolate_lin(pool_.Pcor.cgrad[row_].current, pool_.Pcor.cgrad[col_].current)
        Df_ = np.array([1 / interpolate_lin(aC_u, aF_u), 1 / interpolate_lin(aC_v, aF_v), 1 / interpolate_lin(aC_w, aF_w)])
        Df_ = Df_ * interpolate_lin(info_.cells[row_].size, info_.cells[col_].size)

        for it1 in [0,1,2]:
            v_l[it1].funit[face_].new = v_l[it1].funit[face_].new #- gradpcorf[it1] * Df_[it1] 

    def v_mdot_correct_doublestar_inlet(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                        A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                        lambda_, delta_t, isinit: bool = False):
        
        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}

        Sf_ = geom_.Sf.vec(row_, face_)
        n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(row_, face_) * -1
        
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, pool_.T.funit[face_].current, working_density_ = 0)
        Dauf_ = Pcorrect.calc_bound_Dauf(row_, face_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # mbstar_ = Pcorrect.calc_bound_mfstar(row_, face_, rho_, Dauf_, info_, pool_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # vf_ = np.array([pool_.u.funit[face_].new, pool_.v.funit[face_].new, pool_.w.funit[face_].new])
        
        # # try:
        # #     mbcor_ = rho_ * mbstar_ * Dauf_ * pool_.Pcor.cunit[row_].current * info_.cells[row_].size / (mbstar_ - Dauf_ * pow(rho_, 2) * np.dot(vf_, vf_) * info_.cells[row_].size)
        # # except:
        # #     mbcor_ = 0
        # # vcor_ = n_ * (mbcor_ / (rho_ * np.dot(n_, Sf_)))

        mbcor_ = rho_ * Dauf_ * info_.cells[row_].size * pool_.Pcor.cunit[row_].current
        vcor_ = n_ * (mbcor_ / (rho_ * np.dot(n_, Sf_)))

        for it1 in [0,1,2]:
            v_l[it1].funit[face_].new = v_l[it1].cunit[row_].new #+ vcor_[it1]

    def v_mdot_correct_doublestar_outlet(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                        A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                        lambda_, delta_t, isinit: bool = False):
        
        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}

        Sf_ = geom_.Sf.vec(row_, face_)
        n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(row_, face_) * -1
        
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, pool_.T.funit[face_].current, working_density_ = 0)
        Dauf_ = Pcorrect.calc_bound_Dauf(row_, face_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # mbstar_ = Pcorrect.calc_bound_mfstar(row_, face_, rho_, Dauf_, info_, pool_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        # vf_ = np.array([pool_.u.funit[face_].new, pool_.v.funit[face_].new, pool_.w.funit[face_].new])
        
        # # try:
        # #     mbcor_ = rho_ * mbstar_ * Dauf_ * pool_.Pcor.cunit[row_].current * info_.cells[row_].size / (mbstar_ - Dauf_ * pow(rho_, 2) * np.dot(vf_, vf_) * info_.cells[row_].size)
        # # except:
        # #     mbcor_ = 0
        # # vcor_ = n_ * (mbcor_ / (rho_ * np.dot(n_, Sf_)))

        mbcor_ = rho_ * Dauf_ * info_.cells[row_].size * pool_.Pcor.cunit[row_].current
        vcor_ = n_ * (mbcor_ / (rho_ * np.dot(n_, Sf_)))

        for it1 in [0,1,2]:
            v_l[it1].funit[face_].new = v_l[it1].cunit[row_].new + vcor_[it1]

    def noslip_tke_correct(row_, face_, pool_, info_, geom_):
        # pool_.k.cunit[row_].new = 0
        pool_.k.funit[face_].new = pool_.k.cunit[row_].new

    def dirichlet_tke_correct(row_, face_, pool_, info_, geom_):
        v_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
        kref_ = pow(0.01, 2) * np.dot(v_, v_) / 2
        pool_.k.funit[face_].new = kref_ #pool_.k.cunit[row_].new

    def symmetry_tke_correct(row_, face_, pool_, info_, geom_):
        pool_.k.funit[face_].new = pool_.k.cunit[row_].new

    def noslip_stke_correct(row_, face_, pool_, info_, geom_):
        # mu_ = Prop.calc_mu(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current)
        # rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        # n_ = geom_.Sf.norm(row_, face_)
        # v_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
        # parallelv_ = v_ - np.dot(v_, n_) * n_; parallelv_ = np.sqrt(np.sum([x**2 for x in parallelv_]))
        # ustar_ = calc_u_star(row_, -1, pool_, mu_, rho_, parallelv_, info_.cells[row_].wdist)
        # omega_vis = 6 * mu_ / (rho_ * 0.075 * pow(info_.cells[row_].wdist, 2))
        # omega_log = ustar_ / (0.41 * info_.cells[row_].wdist * np.sqrt(0.09))
        # wref_ = np.sqrt(omega_vis**2 + omega_log**2)
        wref_ = np.sqrt(pool_.k.cunit[row_].current) / (0.41 * pow(0.09, 0.25) * info_.cells[row_].wdist)
        pool_.omega.cunit[row_].new = wref_
        pool_.omega.funit[face_].new = wref_

    def dirichlet_stke_correct(row_, face_, pool_, info_, geom_):
        gamma_t = calc_mut(row_, pool_, info_)
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        try:
            omegaref_ = rho_ * pool_.k.funit[face_].current / gamma_t
        except:
            omegaref_ = 0
        pool_.omega.funit[face_].new = omegaref_ #pool_.omega.cunit[row_].new

    def symmetry_stke_correct(row_, face_, pool_, info_, geom_):
        pool_.omega.funit[face_].new = pool_.omega.cunit[row_].new

class Momentum:
    def rhie_chow_interpolate(row_: int, col_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                            A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                            lambda_: float, delta_t, isinit: bool = False):
        
        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}

        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t

        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        V_avg = gC_ * info_.cells[row_].size + (1 - gC_) * info_.cells[col_].size
        
        af_u, af_v, af_w = Pcorrect.calc_afoverline(row_, col_, face_, pool_, A_u, A_v, A_w, info_, geom_, delta_t2)
        af_ = np.array([af_u, af_v, af_w])
        Df_u, Df_v, Df_w = Pcorrect.calc_Df(row_, col_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        Df_ = np.array([Df_u, Df_v, Df_w])
        
        gradf = pool_.P.fgrad[face_].current
        gradf_overline = gC_ * pool_.P.cgrad[row_].current + (1 - gC_) * pool_.P.cgrad[col_].current

        Bf__ = Pcorrect.calc_Bfdoubleoverline(row_, col_, pool_, info_, geom_)
        Bf_ = Pcorrect.calc_Bfoverline(row_, col_, pool_, info_, geom_)
        
        vC_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
        vF_ = np.array([pool_.u.cunit[col_].current, pool_.v.cunit[col_].current, pool_.w.cunit[col_].current])
        vf_ = np.array([pool_.u.funit[face_].current, pool_.v.funit[face_].current, pool_.w.funit[face_].current])
        v_overline_current = gC_ * vC_ + (1 - gC_) * vF_
        
        rho_last = gC_ * Prop.calc_rho(pool_.P.cunit[row_].last, pool_.T.cunit[row_].last, working_density_ = 0) + (1 - gC_) * \
            Prop.calc_rho(pool_.P.cunit[col_].last, pool_.T.cunit[col_].last, working_density_ = 0)                    
        
        vC_last = np.array([pool_.u.cunit[row_].last, pool_.v.cunit[row_].last, pool_.w.cunit[row_].last])
        vF_last = np.array([pool_.u.cunit[col_].last, pool_.v.cunit[col_].last, pool_.w.cunit[col_].last])
        vf_last = np.array([pool_.u.funit[face_].last, pool_.v.funit[face_].last, pool_.w.funit[face_].last])
        v_overline_last = gC_ * vC_last + (1 - gC_) * vF_last
        
        vC_new = np.array([pool_.u.cunit[row_].new, pool_.v.cunit[row_].new, pool_.w.cunit[row_].new])
        vF_new = np.array([pool_.u.cunit[col_].new, pool_.v.cunit[col_].new, pool_.w.cunit[col_].new])
        v_overline_new = gC_ * vC_new + (1 - gC_) * vF_new
        
        bC_func0 = lambda x: V_avg * (gradf[x] - gradf_overline[x])
        bC_func1 = lambda x: (1 - lambda_) * af_[x] * (vf_[x] - v_overline_current[x])
        bC_func2 = lambda x: V_avg * (Bf_[x] - Bf__[x])
        bC_func3 = lambda x: rho_last * V_avg / delta_t2 * (vf_last[x] - v_overline_last[x])
        
        bC_arr = lambda x: (-bC_func0(x) + bC_func1(x) + bC_func2(x) + bC_func3(x)) * Df_[x]

        # print(row_, col_, face_)
        for it1 in [0,1,2]:
            # print(-bC_func0(it1), bC_func1(it1), bC_func2(it1), bC_func3(it1))
            v_l[it1].funit[face_].new = v_overline_new[it1] #+ bC_arr(it1)
        # print()

    def rhie_chow_inlet_interpolate(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                    A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                    lambda_: float, delta_t, isinit: bool = False):

        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}

        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t

        V_avg = info_.cells[row_].size
        
        af_ = np.array([A_u[row_, row_], A_v[row_, row_], A_w[row_, row_]])
        Df_u, Df_v, Df_w = Pcorrect.calc_bound_Df(row_, face_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        Df_ = np.array([Df_u, Df_v, Df_w])
        
        n_ = geom_.Sf.norm(row_, face_)
        dCf_ = geom_.dCf.vec(row_, face_)
        
        gradf = pool_.P.cgrad[row_].current - (np.dot(pool_.P.cgrad[row_].current, n_)) * n_
        gradC = pool_.P.cgrad[row_].current

        BC__ = Pcorrect.calc_BCdoubleoverline(row_, pool_, info_, geom_)
        BC_ = np.array([0, 9.81 * Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0), 0]) #pool_.working_density), 0])
        
        vC_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
        vf_ = np.array([pool_.u.cunit[face_].current, pool_.v.cunit[face_].current, pool_.w.cunit[face_].current])
        
        rho_last = Prop.calc_rho(pool_.P.funit[face_].last, pool_.T.funit[face_].last, working_density_ = 0)

        vC_last = np.array([pool_.u.cunit[row_].last, pool_.v.cunit[row_].last, pool_.w.cunit[row_].last])
        vf_last = np.array([pool_.u.funit[face_].last, pool_.v.funit[face_].last, pool_.w.funit[face_].last])
        
        vC_new = np.array([pool_.u.cunit[row_].new, pool_.v.cunit[row_].new, pool_.w.cunit[row_].new])

        bC_func0 = lambda x: V_avg * (gradf[x] - gradC[x])
        bC_func1 = lambda x: (1 - lambda_) * af_[x] * (vf_[x] - vC_[x])
        bC_func2 = lambda x: V_avg * (BC_[x] - BC__[x])
        bC_func3 = lambda x: rho_last * V_avg / delta_t2 * (vf_last[x] - vC_last[x])
        
        bC_arr = lambda x: (-bC_func0(x) + bC_func1(x) + bC_func2(x) + bC_func3(x)) * Df_[x]

        # print(row_, face_)
        for it1 in [0,1,2]:
            # print(-bC_func0(it1), bC_func1(it1), bC_func2(it1), bC_func3(it1))
            v_l[it1].funit[face_].new = vC_new[it1] #+ bC_arr(it1)
        # print()

    def rhie_chow_outlet_interpolate(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                                    A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                                    lambda_: float, delta_t, isinit: bool = False):
        
        v_l = {0: pool_.u, 1: pool_.v, 2: pool_.w}

        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t

        V_avg = info_.cells[row_].size
        
        af_ = np.array([A_u[row_, row_], A_v[row_, row_], A_w[row_, row_]])
        Df_u, Df_v, Df_w = Pcorrect.calc_bound_Df(row_, face_, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)
        Df_ = np.array([Df_u, Df_v, Df_w])
        
        n_ = geom_.Sf.norm(row_, face_)
        dCf_ = geom_.dCf.vec(row_, face_)
        
        gradf = pool_.P.cgrad[row_].current - (np.dot(pool_.P.cgrad[row_].current, n_)) * n_
        gradC = pool_.P.cgrad[row_].current

        BC__ = Pcorrect.calc_BCdoubleoverline(row_, pool_, info_, geom_)
        BC_ = np.array([0, 9.81 * Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0), 0]) #pool_.working_density), 0])

        vC_ = np.array([pool_.u.cunit[row_].current, pool_.v.cunit[row_].current, pool_.w.cunit[row_].current])
        vf_ = np.array([pool_.u.funit[face_].current, pool_.v.funit[face_].current, pool_.w.funit[face_].current])
        
        rho_last = Prop.calc_rho(pool_.P.funit[face_].last, pool_.T.funit[face_].last, working_density_ = 0)

        vC_last = np.array([pool_.u.cunit[row_].last, pool_.v.cunit[row_].last, pool_.w.cunit[row_].last])
        vf_last = np.array([pool_.u.funit[face_].last, pool_.v.funit[face_].last, pool_.w.funit[face_].last])
        
        vC_new = np.array([pool_.u.cunit[row_].new, pool_.v.cunit[row_].new, pool_.w.cunit[row_].new])

        bC_func0 = lambda x: V_avg * (gradf[x] - gradC[x])
        bC_func1 = lambda x: (1 - lambda_) * af_[x] * (vf_[x] - vC_[x])
        bC_func2 = lambda x: V_avg * (BC_[x] - BC__[x])
        bC_func3 = lambda x: rho_last * V_avg / delta_t2 * (vf_last[x] - vC_last[x])
        
        bC_arr = lambda x: (-bC_func0(x) + bC_func1(x) + bC_func2(x) + bC_func3(x)) * Df_[x]

        for it1 in [0,1,2]:
            v_l[it1].funit[face_].new = vC_new[it1] #+ bC_arr(it1)

    def calc_bulk_stress(row_: int, which: int, pool_: Pool, info_: Info, geom_: Geom):
        bulk_ = 0
        for it1, it2 in info_.cells[row_].connect.items():
            dCf_ = geom_.dCf.scalar(row_, it2); dFf_ = geom_.dCf.scalar(it1, it2)
            gC_ = dFf_ / (dCf_ + dFf_)
            muf_ = gC_ * Prop.calc_mu(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current) + (1 - gC_) * Prop.calc_mu(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
            graduf_ = gC_ * pool_.u.cgrad[it1].current + (1 - gC_) * pool_.u.cgrad[it1].current
            gradvf_ = gC_ * pool_.v.cgrad[it1].current + (1 - gC_) * pool_.v.cgrad[it1].current
            gradwf_ = gC_ * pool_.w.cgrad[it1].current + (1 - gC_) * pool_.w.cgrad[it1].current
            Sf_ = geom_.Sf.vec(row_, it2)
            bulk_ += muf_ * (graduf_[0] + gradvf_[1] + gradwf_[2]) * Sf_[which]
        return -2/3 * bulk_

    def calc_pressure_grad(row_: int, which: int, pool_: Pool, info_: Info, geom_: Geom):
        pgrad = np.array([0, 0, 0], dtype = float)
        for x, y in info_.cells[row_].connect.items():
            dCf_ = geom_.dCf.scalar(row_, y); dFf_ = geom_.dCf.scalar(row_, y)
            gC_ = dFf_ / (dCf_ + dFf_)
            pb_ = gC_ * pool_.P.cunit[row_].current + (1 - gC_) * pool_.P.cunit[x].current
            Sf_ = geom_.Sf.vec(row_, y)
            pgrad = pgrad + pb_ * Sf_
        return pgrad[which]

    def calc_coef(which: int, pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, isinit: bool = False):
        bound_func = {'inlet': Boundary.inlet_momentum, 'outlet': Boundary.outlet_momentum, 'noslip': Boundary.noslip_momentum}
        V_ = {0: pool_.u, 1: pool_.v, 2: pool_.w}
        dataA_lil = []; dataB_lil = []; row = []; col = []; row_B = []
        approx = []
        max_cfl_mom = 0
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        for it1, it2 in info_.cells.items():
            iswall = False
            if len(list(it2.conj.keys())) > 0:
                iswall = True
            if 'fluid' in it2.group:
                row_B.append(it1)
                rho_ = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = 0)
                rho_last = Prop.calc_rho(pool_.P.cunit[it1].last, pool_.T.cunit[it1].last, working_density_ = 0)
                cp_ = Prop.calc_cp(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                cp_last = Prop.calc_cp(pool_.P.cunit[it1].last, pool_.T.cunit[it1].last)
                gammaC_ = Prop.calc_mu(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                gammaC_t = calc_mut(it1, pool_, info_)
                aC_nwf, aF_nwf, bC_nwf, row_, col_ = linear.NWF(V_[which], it1, 'mu', 'none', pool_, info_, geom_, user_, iswall = iswall)
                aC_temporal, bC_temporal = linear.temporal_conv(V_[which], pool_, info_, it1, delta_t, rho_ = rho_, rho_last = rho_last,
                                                                cp_ = cp_, cp_last = cp_last, isinit_ = isinit)

                bC_momentum_ = -pool_.P.cgrad[it1].current[which] * info_.cells[it1].size #Momentum.calc_pressure_grad(it1, which, pool_, info_, geom_) 
                if which == 1:
                    bC_momentum_ += Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = pool_.working_density) * 9.81 * info_.cells[it1].size

                aC_bound = 0; bC_bound = 0
                
                for it3 in it2.bound:
                    for it4 in info_.faces[it3].group:
                        if it4 in list(bound_func.keys()):
                            aC_b, bC_b = bound_func[it4](V_[which], it1, it3, aC_nwf, gammaC_ + gammaC_t, info_, pool_, geom_, user_)
                            aC_bound += aC_b; bC_bound += bC_b

                if len(list(it2.conj.keys())) > 0:
                    # aC_nwf = 0; aF_nwf = np.array([0 for it1 in aF_nwf]); bC_nwf = 0
                    for it3, it4 in it2.conj.items():
                        aC_noslip, bC_noslip = Boundary.noslip_momentum(V_[which], it1, it4, aC_nwf, gammaC_, info_, pool_, geom_, user_)
                        aC_bound += aC_noslip; bC_bound += bC_noslip
                
                aC_trans, bC_trans = linear.relax_trans(V_[which], pool_, info_, aC_nwf + aC_bound, it1, lambda_, delta_t, isinit_ = isinit)
                row.extend(row_); col.extend(col_)
                dataA_lil.append(aC_nwf + aC_bound + aC_trans + aC_temporal); dataA_lil.extend(aF_nwf)
                dataB_lil.append(bC_nwf + bC_bound + bC_trans + bC_temporal + bC_momentum_)

                max_cfl_mom = np.max([max_cfl_mom, (aC_nwf + aC_bound) * delta_t2 / (rho_last * info_.cells[it1].size)])

                # try:
                #     if (bC_nwf + bC_bound + bC_trans + bC_momentum_) / (aC_nwf + aC_bound + aC_trans + np.sum(aF_nwf)) < 0:
                #         print((bC_nwf + bC_bound + bC_trans + bC_momentum_) / (aC_nwf + aC_bound + aC_trans + np.sum(aF_nwf)))
                # except:
                #     pass
            elif 'fluid' not in info_.cells[it1].group:
                row_B.append(it1)
                row.append(it1); col.append(it1)
                dataA_lil.append(float(0)); dataB_lil.append(float(0))
                approx.append(float(0))
            
        A_ = sparse.coo_matrix((dataA_lil, (row, col)), dtype = float).tocsr()
        B_ = sparse.coo_matrix((dataB_lil, (row_B, list(np.zeros(shape=(1, len(row_B)), dtype = int)[0]))), dtype = float).tocsr()

        return A_, B_, max_cfl_mom

    def correct_value(info_: Info, pool_: Pool, geom_: Geom,
                      new_u_cunit: np.ndarray, new_v_cunit: np.ndarray, new_w_cunit: np.ndarray,
                      A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                      lambda_, delta_t, exitCode, isinit: bool = False):
        # correct inlet and outlet values
        correct_func = {'inlet': Momentum.rhie_chow_inlet_interpolate, 'outlet': Momentum.rhie_chow_outlet_interpolate, 'noslip': Correct.dirichlet_momentum_correct}
        #correct_func = {'inlet': Momentum.rhie_chow_inlet_interpolate}        
        for it1 in range(0, A_u.get_shape()[0]):
            it2 = info_.cells[it1]
            if 'fluid' in it2.group:
                pool_.u.cunit[it1].new = new_u_cunit[it1]
                pool_.v.cunit[it1].new = new_v_cunit[it1]
                pool_.w.cunit[it1].new = new_w_cunit[it1]
                
                if len(list(it2.conj.keys())) > 0:
                    for it3, it4 in it2.conj.items():
                        pool_.u.funit[it4].new = 0
                        pool_.v.funit[it4].new = 0
                        pool_.w.funit[it4].new = 0
                        pool_.mdot.funit[it4].new = 0
        
        pool_.u.renew_gradient(info_, geom_)
        pool_.v.renew_gradient(info_, geom_)
        pool_.w.renew_gradient(info_, geom_)
        
        clear_of_interpolation = []
        for it1 in range(0,  A_u.get_shape()[0]):
            it2 = info_.cells[it1]
            if 'fluid' in it2.group:
                for it3, it4 in it2.connect.items():
                    if it4 not in clear_of_interpolation:
                        Momentum.rhie_chow_interpolate(it1, it3, it4, pool_, info_, geom_, A_u, A_v, A_w,
                                                        lambda_, delta_t, isinit = isinit)
                        clear_of_interpolation.append(it4)
                for it3 in it2.bound:
                    [correct_func[it4](it1, it3, pool_, info_, geom_, A_u, A_v, A_w,
                                        lambda_, delta_t, isinit = isinit) for it4 in info_.faces[it3].group if it4 in list(correct_func.keys())]

        # v_ = np.array([(pool_.u.cunit[it1].new, pool_.v.cunit[it1].new,
        #                 pool_.w.cunit[it1].new, pool_.P.cunit[it1].new) \
        #                 for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group])
        # v_ = np.transpose(v_)
        
        # print('momentum - avg new u: %s, v: %s, w: %s, P: %s (%s)' % (np.mean(v_[0]), np.mean(v_[1]), np.mean(v_[2]), np.mean(v_[3]), exitCode))
           
    def solve(which_: int, pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, isinit: bool = False):
        V_ = {0: pool_.u, 1: pool_.v, 2: pool_.w}
        A_mom, B_mom, max_cfl_mom = Momentum.calc_coef(which_, pool_, info_, geom_, user_, lambda_, delta_t, isinit = isinit); B_mom_arr = B_mom.toarray()
        n = A_mom.get_shape()[0]

        current_mom = np.array([V_[which_].cunit[it1].current for it1 in range(0, n)])
        """
        try:
            lsqr_relaxer = np.eye(n, M = n)
            for it1 in range(lsqr_relaxer.shape[0]):
                lsqr_relaxer[it1][it1] = 0.01 * B_mom[it1, 0]
            Pmom = A_mom + lsqr_relaxer
            ilu = sparse.linalg.spilu(Pmom)
            M_x = lambda x: ilu.solve(x)
            Amom = sparse.linalg.LinearOperator((n, n), M_x)
            new_mom, exitCode_mom = sparse.linalg.lgmres(A_mom, B_mom_arr, x0 = current_mom, M = Amom, tol = tol_, maxiter = maxiter_, inner_m = 1000); new_mom = np.array(new_mom).ravel()
        except:
        """
        try:
            ml = smoothed_aggregation_solver(A_mom)
            Amom = ml.aspreconditioner(cycle = 'V')
            new_mom, exitCode_mom = sparse.linalg.gmres(A_mom, B_mom_arr, x0 = current_mom, M = Amom, atol = tol_, tol = tol_, maxiter = maxiter_); new_mom = np.array(new_mom).ravel()
        except:
            new_mom, exitCode_mom = sparse.linalg.gmres(A_mom, B_mom_arr, x0 = current_mom, atol = tol_, tol = tol_, maxiter = maxiter_); new_mom = np.array(new_mom).ravel()            
        
        #if exitCode_mom > 0:
        #    new_mom = current_mom
        """
        if which_ == 2:
            new_mom = [np.abs(it1) for it1 in new_mom]
        """
        #p_code = {0: 'u', 1: 'v', 2: 'w'}
        #print('avg %s: %s (%s)' % (p_code[which_], np.mean(new_mom), exitCode_mom))

        return A_mom, B_mom, new_mom, exitCode_mom, max_cfl_mom

class Energy:
    def calc_phi(row_: int, pool_: Pool):
        gradv_ = np.array([pool_.u.cgrad[row_].current[0], pool_.v.cgrad[row_].current][1], pool_.w.cgrad[row_].current[2])
        phi_ = 2 * np.sum([it1**2 for it1 in gradv_])
        phi_ += (pool_.u.cgrad[row_].current[1] + pool_.v.cgrad[row_].current[0])**2
        phi_ += (pool_.u.cgrad[row_].current[2] + pool_.w.cgrad[row_].current[0])**2
        phi_ += (pool_.v.cgrad[row_].current[2] + pool_.w.cgrad[row_].current[1])**2
        return phi_

    def calc_coef(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, channel_length, isinit: bool = False, issteady: bool = False):
        bound_func = {'inlet': Boundary.dirichlet_energy, 'outlet': Boundary.symmetry_fluid}
        approx = []; row = []; col = []; row_B = []
        dataA_lil = []; dataB_lil = []
        dataAC_lil_solid = {}; dataAF_lil_solid = {}; dataB_lil_solid = {}
        ctd_abs = 0; ctd_glass = 0

        max_cfl_energy = 0

        abs_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'abs' in info_.cells[it1].group]; abs_T_avg = np.mean(abs_T_avg)
        glass_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'glass' in info_.cells[it1].group]; glass_T_avg = np.mean(glass_T_avg)

        qhwg_ = 5.67 * pow(10, -8) * (abs_T_avg**2 + glass_T_avg**2) * (abs_T_avg + glass_T_avg) \
            / ((1/user_.solid_eps['glass']) + (1/user_.solid_eps['abs']) - 1)
        qhwg_ = qhwg_ * (abs_T_avg - glass_T_avg)

        if isinit is False:
            delta_t2 = 2 * delta_t
        else:
            delta_t2 = delta_t

        for it1, it2 in info_.cells.items():
            iswall = -1
            if len(list(it2.conj.keys())) > 0:
                iswall = it2.conj[list(it2.conj.keys())[0]]
            if 'fluid' in it2.group:
                row_B.append(it1)
                rho_ = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = 0)
                rho_last = Prop.calc_rho(pool_.P.cunit[it1].last, pool_.T.cunit[it1].last, working_density_ = 0)
                cp_ = Prop.calc_cp(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                cp_last = Prop.calc_cp(pool_.P.cunit[it1].last, pool_.T.cunit[it1].last)
                gammaC_ = Prop.calc_k(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                gammaC_t = calc_kt(it1, pool_, info_)
                alpha_ = (gammaC_ + gammaC_t) / (rho_ * cp_)
                aC_nwf, aF_nwf, bC_nwf, row_, col_ = linear.NWF(pool_.T, it1, 'k', 'none', pool_, info_, geom_, user_, cp_ = cp_)
                aC_temporal, bC_temporal = linear.temporal_conv(pool_.T, pool_, info_, it1, delta_t = delta_t, rho_ = rho_,
                                                                rho_last = rho_last, cp_ = cp_, cp_last = cp_last, isinit_ = isinit)
                mu_ = Prop.calc_mu(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                phi_ = Energy.calc_phi(it1, pool_)
                vC_ = np.array([pool_.u.cunit[it1].current, pool_.v.cunit[it1].current, pool_.w.cunit[it1].current])
                
                v_ = np.array([pool_.u.cunit[it1].current, pool_.v.cunit[it1].current, pool_.w.cunit[it1].current])
                gradP_ = pool_.P.cgrad[it1].current

                langrange_P = ((pool_.P.cunit[it1].current - pool_.P.cunit[it1].last) / delta_t2 + np.dot(vC_, gradP_))
                
                bC_energy = 0
                # if iswall >= 0:
                #     n_ = np.array([0,1,0], dtype = float) #geom_.Sf.norm(it1, iswall)
                #     v_ = np.array([pool_.u.cunit[it1].current, pool_.v.cunit[it1].current, pool_.w.cunit[it1].current])
                #     parallelv_ = np.dot(v_, n_) * n_
                #     parallelv_scalar = np.linalg.norm(parallelv_)
                #     utau_ = calc_u_tau(it1, -1, pool_, info_, mu_, rho_, parallelv_scalar, info_.cells[it1].wdist)
                #     ustar_ = calc_u_star(it1, -1, pool_, mu_, rho_, parallelv_scalar, info_.cells[it1].wdist)
                #     bC_energy += rho_ * ustar_ * np.sum([parallelv_[x] * v_[x] for x in [0,1,2]]) / (0.41 * info_.cells[it1].wdist) 
                # elif iswall < 0:
                bC_energy += (mu_ * phi_)
                
                bC_energy = (bC_energy) * info_.cells[it1].size
                aC_bound = 0; bC_bound = 0

                for it3 in it2.bound:
                    for it4 in info_.faces[it3].group:
                        if it4 in list(bound_func.keys()):
                            aC_b, bC_b = bound_func[it4](pool_.T, it1, it3, aC_nwf, gammaC_ + gammaC_t, info_, pool_, geom_, user_, cp_ = cp_)
                            aC_bound += aC_b; bC_bound += bC_b

                if len(list(it2.conj.keys())) > 0:
                    for it3, it4 in it2.conj.items():
                        if 'abs' in info_.cells[it3].group:
                            ctd_abs += 1
                            aC_b_fluid, aF_b_fluid, bC_b_fluid, aC_b_solid, aF_b_solid, bC_b_solid  = \
                                Boundary.conjugateFluidAbs(it1, it3, it4, gammaC_, pool_, info_, geom_,
                                                           user_, channel_length, cp_ = cp_, qhwg_ = -qhwg_)
                            aC_nwf += aC_b_fluid; aF_nwf[-1] += aF_b_fluid; bC_bound += bC_b_fluid
                            dataAC_lil_solid[it3] = aC_b_solid
                            dataAF_lil_solid[it3] = aF_b_solid
                            dataB_lil_solid[it3] = bC_b_solid
                        elif 'glass' in info_.cells[it3].group:
                            ctd_glass += 1
                            aC_b_fluid, aF_b_fluid, bC_b_fluid, aC_b_solid, aF_b_solid, bC_b_solid = \
                                Boundary.conjugateFluidGlass(it1, it3, it4, gammaC_, pool_, info_, geom_,
                                                             user_, channel_length, cp_ = cp_, qhwg_ = qhwg_)
                            aC_nwf += aC_b_fluid; aF_nwf[-1] += aF_b_fluid; bC_bound += bC_b_fluid
                            dataAC_lil_solid[it3] = aC_b_solid
                            dataAF_lil_solid[it3] = aF_b_solid
                            dataB_lil_solid[it3] = bC_b_solid

                aC_trans, bC_trans = linear.relax_trans(pool_.T, pool_, info_, aC_nwf + aC_bound, it1, lambda_, delta_t, cp_ = cp_, cp_last = cp_last, isinit_ = isinit)
                row.extend(row_); col.extend(col_)
                dataA_lil.append(aC_nwf + aC_bound + aC_trans + aC_temporal); dataA_lil.extend(aF_nwf)
                dataB_lil.append(bC_nwf + bC_bound + bC_trans + bC_temporal + bC_energy)

                max_cfl_energy = np.max([max_cfl_energy, (aC_nwf + aC_bound) * delta_t2 / (rho_last * info_.cells[it1].size)])

                # if (bC_trans + bC_nwf + bC_temporal + bC_bound + bC_energy) / (aC_nwf + np.sum(aF_nwf) + aC_trans + aC_temporal + aC_bound) < 300:
                # print(aC_nwf, aF_nwf, np.sum(aF_nwf), bC_nwf, f'|', (aC_nwf + np.sum(aF_nwf)), f'<- sum scheme zero')
                # print(aC_trans, bC_trans, f'|', bC_trans / aC_trans, f'<- sum zero resulting init value')
                # print(aC_temporal, bC_temporal, f'|', bC_temporal / aC_temporal, f'<- sum not equal (temporal term)')
                # print(aC_bound, bC_bound, bC_energy, f'<- sum of other terms zero')
                # print((bC_trans + bC_nwf + bC_temporal + bC_bound + bC_energy), (aC_nwf + np.sum(aF_nwf) + aC_trans + aC_temporal + aC_bound), f'|', (bC_trans + bC_nwf + bC_temporal + bC_bound + bC_energy) / (aC_nwf + np.sum(aF_nwf) + aC_trans + aC_temporal + aC_bound), f'<- numerical error in total')
                # print()
                
        for it1, it2 in info_.cells.items():
            if 'fluid' not in it2.group:
                row_B.append(it1)
                rho_ = user_.solid_rho[it2.group[0]]; cp_ = user_.solid_cp[it2.group[0]]
                aC_nwf, aF_nwf, bC_nwf, row_, col_ = linear.NWF(pool_.T, it1, 'k', 'none', pool_, info_, geom_, user_,
                                                                cp_ = cp_, issolid = True)
                delta_t_solid = delta_t2
                if issteady is True:
                    delta_t_solid = pow(channel_length, 2) / user_.solid_alpha[it2.group[0]]
                aC_temporal, bC_temporal = linear.temporal_conv(pool_.T, pool_, info_, it1, delta_t = delta_t_solid, rho_ = rho_, rho_last = rho_,
                                                                cp_ = cp_, cp_last = cp_, isinit_ = isinit)
                aC_bound = 0; bC_bound = 0
                if it1 in list(dataB_lil_solid.keys()):
                    aC_bound = dataAC_lil_solid[it1]
                    aF_nwf[-1] += dataAF_lil_solid[it1]
                    bC_bound = dataB_lil_solid[it1]

                if 'insul' in it2.group:
                    find_bound = [y for x, y in it2.connect.items() if 'abs' in info_.cells[x].group]; room_f = find_bound[0]
                    Sf_ = info_.faces[room_f].size; dCf_ = geom_.dCf.scalar(it1, room_f)
                    aC_bound += 10 * Sf_
                    bC_bound += 10 * Sf_ * 300

                for it3 in it2.bound:
                    if len(info_.faces[it3].group) < 1:
                        aC_b, bC_b = Boundary.symmetry_solid(pool_.T, it1, it3, user_.solid_k[it2.group[0]], info_, pool_, geom_, user_, cp_ = cp_)
                        aC_bound += aC_b; bC_bound += bC_b

                aC_trans, bC_trans = linear.relax_trans(pool_.T, pool_, info_, aC_nwf + aC_bound, it1, lambda_, delta_t_solid,
                                                        cp_ = user_.solid_cp[it2.group[0]], isinit_ = isinit)
                row.extend(row_); col.extend(col_)
                dataA_lil.append(aC_nwf + aC_trans + aC_bound); dataA_lil.extend(aF_nwf)
                dataB_lil.append(bC_nwf + bC_trans + bC_bound)

                max_cfl_energy = np.max([max_cfl_energy, (aC_nwf + aC_bound) * delta_t2 / (rho_last * info_.cells[it1].size)])

        A_ = sparse.coo_matrix((dataA_lil, (row, col)), dtype = float).tocsr()
        B_ = sparse.coo_matrix((dataB_lil, (row_B, np.zeros(shape=(1,len(row_B)), dtype = int)[0])), dtype = float).tocsr()
        
        return A_, B_, max_cfl_energy

    def correct_value(info_: Info, pool_: Pool, geom_: Geom, new_cunit: np.ndarray, exitCode):
        # correct, noslip-conj, inlet, outlet values
        correct_func = {'inlet': Correct.dirichlet_energy_correct, 'outlet': Correct.symmetry_energy_correct}
        for it1 in range(0, len(new_cunit)):
            it2 = info_.cells[it1]
            pool_.T.cunit[it1].new = new_cunit[it1]
            #pool_.P.cunit[it1].new = new_cunit[it1] * Prop.calc_rho(pool_.P.cunit[it1].current, new_cunit[it1]) * 287.052874
                    #if len(info_.faces[it3].group) == 0 or all([it4 in list(correct_func.keys()) for it4 in info_.faces[it3].group]) is False:
                    #    Correct.dirichlet_energy_correct(pool_.T, it1, it3)
            if len(list(it2.conj.keys())) > 0:
                for it3, it4 in it2.conj.items():
                    pool_.T.funit[it4].new = pool_.T.cunit[it3].new
            if 'fluid' in it2.group:
                for it3 in it2.bound:
                    [correct_func[it4](pool_, geom_, it1, it3) for it4 in info_.faces[it3].group if it4 in list(correct_func.keys())]
        
        pool_.T.update(info_, geom_)

        # for it1 in range(0, len(new_cunit)):
        #     if 'fluid' in it2.group:
        #         for it3 in it2.bound:
        #             [correct_func[it4](pool_, geom_, it1, it3) for it4 in info_.faces[it3].group if it4 in list(correct_func.keys())]

        # pool_.T.update(info_, geom_)

        # T_ = {'fluid': [], 'abs': [], 'glass': [], 'insul': []}
        # [T_[it2.group[0]].append(pool_.T.cunit[it1].current) for it1, it2 in info_.cells.items()]
        # pool_.working_density = 1.293 * (1 - 3400 * pow(10, -6) * (np.min(T_['fluid']) - 300))
        
        # print('avg T fluid: %s; abs: %s; glass: %s; insulation: %s (%s)' % (np.mean(T_['fluid']), np.mean(T_['abs']), np.mean([T_['glass']]), np.mean([T_['insul']]), exitCode))
        # print('working density: %s' % pool_.working_density)
              
    def solve(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, channel_length, isinit: bool = False, issteady: bool = False):
        A_T, B_T, max_cfl_energy = Energy.calc_coef(pool_, info_, geom_, user_, lambda_, delta_t, channel_length, isinit = isinit, issteady = issteady); B_T_arr = B_T.toarray()
        
        #print(A_T)
        #print(B_T)
        """
        x_ = np.full(shape=(A_T.get_shape()[0], 1), fill_value=300)
        Bstar_ = A_T * x_
        Barr_ = B_T.toarray().ravel(); Bstar_ = np.array(Bstar_).ravel()
        rmsr_300 = np.array([(Barr_[it1] - Bstar_[it1])**2 for it1 in range(0, B_T.shape[0])])
        rmsr_300 = math.sqrt(np.sum(rmsr_300) / B_T.shape[0])
        """
        n = A_T.get_shape()[0]

        current_T = np.array([pool_.T.cunit[it1].current for it1 in range(0, n)])
        """
        try:
            lsqr_relaxer = np.eye(n, M = n)
            for it1 in range(lsqr_relaxer.shape[0]):
                lsqr_relaxer[it1][it1] = 0.01 * B_T[it1, 0]
            PT = A_T + lsqr_relaxer
            ilu = sparse.linalg.spilu(PT)
            M_x = lambda x: ilu.solve(x)
            AT = sparse.linalg.LinearOperator((n, n), M_x)
            new_T, exitCode = sparse.linalg.lgmres(A_T, B_T_arr, x0 = current_T, M = AT, maxiter = maxiter_, tol = tol_, inner_m = 1000); new_T = np.array(new_T).ravel()
        except:
        """
        try:
            ml = smoothed_aggregation_solver(A_T)
            AT = ml.aspreconditioner(cycle = 'V')
            new_T, exitCode = sparse.linalg.gmres(A_T, B_T_arr, x0 = current_T, M = AT, maxiter = maxiter_, atol = tol_, tol = tol_); new_T = np.array(new_T).ravel()
        except:
            new_T, exitCode = sparse.linalg.gmres(A_T, B_T_arr, x0 = current_T, maxiter = maxiter_, atol = tol_, tol = tol_); new_T = np.array(new_T).ravel()     
        
        #if exitCode > 0:
        #    new_T = current_T
        """
        for it1 in range(B_T.shape[0]):
            lbound = []
            if all([it2 in info_.cells[it1].group for it2 in ['abs']]):
                [lbound.extend(info_.faces[it2].group) for it2 in info_.cells[it1].bound]; lbound = np.unique(np.array(lbound))
                try:
                    print(it1, info_.cells[it1].group, np.sum(A_T.getrow(it1).toarray()), B_T[it1, 0], B_T[it1, 0] / np.sum(A_T.getrow(it1).toarray()), new_T[it1], lbound)
                except:
                    print(it1, info_.cells[it1].group, np.sum(A_T.getrow(it1).toarray()), B_T[it1, 0], new_T[it1], lbound)
        """
        """
        Bstar_ = A_T * new_T
        Barr_ = B_T.toarray().ravel()
        rmsr_new = np.array([(Barr_[it1] - Bstar_[it1])**2 for it1 in range(0, B_T.shape[0])])
        rmsr_new = math.sqrt(np.sum(rmsr_new) / B_T.shape[0])

        print(new_T)
        print('RMSR 300: %s, RMSR new: %s, code: %s' % (rmsr_300, rmsr_new, exitCode))
        
        new_T = np.array(new_T).ravel()
        """

        return A_T, B_T, new_T, exitCode, max_cfl_energy

class Pcorrect:
    def calc_afoverline(row_: int, col_: int, face_: int, pool_: Pool,
                        A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                        info_: Info, geom_: Geom, delta_t2: float):
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        left_u = np.sum(A_u.getrow(row_).toarray())
        left_v = np.sum(A_v.getrow(row_).toarray())
        left_w = np.sum(A_w.getrow(row_).toarray())
        right_u = np.sum(A_u.getrow(col_).toarray())
        right_v = np.sum(A_v.getrow(col_).toarray())
        right_w = np.sum(A_w.getrow(col_).toarray())
        af_u = gC_ * left_u + (1 - gC_) * right_u
        af_v = gC_ * left_v + (1 - gC_) * right_v
        af_w = gC_ * left_w + (1 - gC_) * right_w
        return np.array([af_u, af_v, af_w])

    def calc_Dauf(row_: int, col_: int, pool_: Pool, info_: Info, geom_: Geom,
                A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                lambda_: float, delta_t: float, isinit: bool = False):
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        face_ = info_.cells[row_].connect[col_]
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        rho_ = gC_ * Prop.calc_rho(pool_.P.cunit[row_].new, pool_.T.cunit[row_].new, working_density_ = 0) + (1 - gC_) * \
            Prop.calc_rho(pool_.P.cunit[col_].new, pool_.T.cunit[col_].new, working_density_ = 0)
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        V_avg = gC_ * info_.cells[row_].size + (1 - gC_) * info_.cells[col_].size
        dCF_ = geom_.dCF.scalar(row_, col_) * geom_.Sf.norm(row_, face_)
        SCf_ = geom_.Sf.vec(row_, face_)

        af_u, af_v, af_w = Pcorrect.calc_afoverline(row_, col_, face_, pool_, A_u, A_v, A_w, info_, geom_, delta_t2)

        Df_u = 1 / af_u #+ rho_ * V_avg / delta_t2)
        Df_v = 1 / af_v #+ rho_ * V_avg / delta_t2)
        Df_w = 1 / af_w #+ rho_ * V_avg / delta_t2)

        try:
            Dauf_ = ((Df_u * SCf_[0])**2 + (Df_v * SCf_[1])**2 + (Df_w * SCf_[2])**2) / \
                    (dCF_[0] * Df_u * SCf_[0] + dCF_[1] * Df_v * SCf_[1] + dCF_[2] * Df_w * SCf_[2])
        except:
            Dauf_ = np.dot(np.array([Df_u, Df_v, Df_w]), SCf_)

        return Dauf_

    def calc_bound_Dauf(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                        A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                        lambda_: float, delta_t: float, isinit: bool = False):
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, pool_.T.funit[face_].current, working_density_ = 0)
        VC_ = info_.cells[row_].size
        dCF_ = geom_.dCf.scalar(row_, face_) * geom_.Sf.norm(row_, face_)
        SCf_ = geom_.Sf.vec(row_, face_)
        aC_u = np.sum(A_u.getrow(row_).toarray())
        aC_v = np.sum(A_v.getrow(row_).toarray())
        aC_w = np.sum(A_w.getrow(row_).toarray())

        Df_u = 1 / aC_u #+ rho_ * VC_ / delta_t2)
        Df_v = 1 / aC_v #+ rho_ * VC_ / delta_t2)
        Df_w = 1 / aC_w #+ rho_ * VC_ / delta_t2)

        try:
            Dauf_ = ((Df_u * SCf_[0])**2 + (Df_v * SCf_[1])**2 + (Df_w * SCf_[2])**2) / \
                    (dCF_[0] * Df_u * SCf_[0] + dCF_[1] * Df_v * SCf_[1] + dCF_[2] * Df_w * SCf_[2])
        except:
            Dauf_ = np.dot(np.array([Df_u, Df_v, Df_w]), SCf_)

        return Dauf_

    def calc_Df(row_: int, col_: int, pool_: Pool, info_: Info, geom_: Geom,
                A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                lambda_: float, delta_t: float, isinit: bool = False):
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        face_ = info_.cells[row_].connect[col_]
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        rho_ = gC_ * Prop.calc_rho(pool_.P.cunit[row_].new, pool_.T.cunit[row_].new, working_density_ = 0) + (1 - gC_) * \
            Prop.calc_rho(pool_.P.cunit[col_].new, pool_.T.cunit[col_].new, working_density_ = 0)
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        V_avg = gC_ * info_.cells[row_].size + (1 - gC_) * info_.cells[col_].size
        dCF_ = geom_.dCF.scalar(row_, col_) * geom_.Sf.norm(row_, face_)
        SCf_ = geom_.Sf.vec(row_, face_)

        af_u, af_v, af_w = Pcorrect.calc_afoverline(row_, col_, face_, pool_, A_u, A_v, A_w, info_, geom_, delta_t2)

        Df_u = 1 / af_u #+ rho_ * V_avg / delta_t2)
        Df_v = 1 / af_v #+ rho_ * V_avg / delta_t2)
        Df_w = 1 / af_w #+ rho_ * V_avg / delta_t2)

        return Df_u, Df_v, Df_w
    
    def calc_bound_Df(row_: int, face_: int, pool_: Pool, info_: Info, geom_: Geom,
                        A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                        lambda_: float, delta_t: float, isinit: bool = False):
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        
        rho_ = Prop.calc_rho(pool_.P.funit[face_].current, pool_.T.funit[face_].current, working_density_ = 0)
        VC_ = info_.cells[row_].size
        aC_u = np.sum(A_u.getrow(row_).toarray())
        aC_v = np.sum(A_v.getrow(row_).toarray())
        aC_w = np.sum(A_w.getrow(row_).toarray())

        Df_u = 1 / aC_u #+ rho_ * VC_ / delta_t2)
        Df_v = 1 / aC_v #+ rho_ * VC_ / delta_t2)
        Df_w = 1 / aC_w #+ rho_ * VC_ / delta_t2)
        
        return Df_u, Df_v, Df_w
    
    def calc_mfstar(row_: int, col_: int, rho_: float, Dauf_: float, info_: Info, pool_: Pool, geom_: Geom, A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                lambda_: float, delta_t: float, isinit: bool = False):
        face_ = info_.cells[row_].connect[col_]
        Sf_ = geom_.Sf.vec(row_, face_)
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        rho_ = gC_ * Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0) + (1 - gC_) * \
            Prop.calc_rho(pool_.P.cunit[col_].current, pool_.T.cunit[col_].current, working_density_ = 0)
        vf_ = np.array([pool_.u.funit[face_].new, pool_.v.funit[face_].new, pool_.w.funit[face_].new])
        return rho_ * np.dot(vf_, Sf_)

    def calc_bound_mfstar(row_: int, face_: int, rho_: float, Dauf_: float, info_: Info, pool_: Pool, geom_: Geom, A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                lambda_: float, delta_t: float, isinit: bool = False, isinlet: bool = True):
        Sf_ = geom_.Sf.vec(row_, face_)
        rho_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = 0)
        vf_ = np.array([pool_.u.funit[face_].new, pool_.v.funit[face_].new, pool_.w.funit[face_].new])

        return rho_ * np.dot(vf_, Sf_)

    def calc_BCdoubleoverline(row_: int, pool_: Pool, info_: Info, geom_: Geom):
        BC_doubleoverline_v = np.array([0, 0, 0], dtype = float)
        rhoC_ = Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = pool_.working_density)
        for it1, it2 in info_.cells[row_].connect.items():
            SCf_ = geom_.Sf.vec(row_, it2)
            dCf_ = geom_.dCf.scalar(row_, it2); dFf_ = geom_.dCf.scalar(it1, it2)
            dCF_ = geom_.dCF.vec(row_, it1)
            gC_ = dFf_ / (dCf_ + dFf_)
            rhoF_ = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = pool_.working_density)
            Bf_ = gC_ * rhoC_ + (1 - gC_) * rhoF_; Bf_ = np.array([0, Bf_ * 9.81, 0])
            BC_doubleoverline_v += gC_ * np.dot(Bf_, dCF_) * SCf_
        for it1 in info_.cells[row_].bound:
            if any([x in ['inlet', 'outlet'] for x in info_.faces[it1].group]) is True:
                dCf_ = geom_.dCf.vec(row_, it1)
                SCf_ = geom_.Sf.vec(row_, it1)
                BC_doubleoverline_v += pool_.P.funit[it1].current * SCf_ #np.dot(BC_, dCf_) * SCf_
            elif len(info_.faces[it1].group) < 1:
                dCf_ = geom_.dCf.vec(row_, it1)
                SCf_ = geom_.Sf.vec(row_, it1)
                BC_doubleoverline_v += pool_.P.cunit[row_].current * SCf_
        for it1, it2 in info_.cells[row_].conj.items():
            dCf_ = geom_.dCf.vec(row_, it1)
            SCf_ = geom_.Sf.vec(row_, it1)
            BC_doubleoverline_v += pool_.P.cunit[row_].current * SCf_
        
        return BC_doubleoverline_v / info_.cells[row_].size

    def calc_Bfdoubleoverline(row_: int, col_: int, pool_: Pool, info_: Info, geom_: Geom):
        face_ = info_.cells[row_].connect[col_]
        BC__ = Pcorrect.calc_BCdoubleoverline(row_, pool_, info_, geom_)
        BF__ = Pcorrect.calc_BCdoubleoverline(col_, pool_, info_, geom_)
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        Bf_v = gC_ * BC__ + (1 - gC_) * BF__
        return Bf_v

    def calc_Bfoverline(row_: int, col_: int, pool_: Pool, info_: Info, geom_: Geom):
        face_ = info_.cells[row_].connect[col_]
        BC__ = np.array([0, 9.81 * Prop.calc_rho(pool_.P.cunit[row_].current, pool_.T.cunit[row_].current, working_density_ = pool_.working_density), 0])
        BF__ = np.array([0, 9.81 * Prop.calc_rho(pool_.P.cunit[col_].current, pool_.T.cunit[col_].current, working_density_ = pool_.working_density), 0])
        dCf_ = geom_.dCf.scalar(row_, face_); dFf_ = geom_.dCf.scalar(col_, face_)
        gC_ = dFf_ / (dCf_ + dFf_)
        Bf_v = gC_ * BC__ + (1 - gC_) * BF__
        return Bf_v

    def calc_coef(pool_: Pool, info_: Info, geom_: Geom, lambda_: float,
                  A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                  delta_t, isinit: bool = False):
        V_ = {0: pool_.u, 1: pool_.v, 2: pool_.w}
        bound_func = {'inlet': Boundary.inlet_pcorrect, 'outlet': Boundary.outlet_pcorrect}
        dataA_lil = []; dataB_lil = []; row_ = []; col_ = []; row_B = []
        approx = []
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        for it1, it2 in info_.cells.items():
            aC_pcor = []; aF_pcor = []; bC_pcor = []
            if 'fluid' in it2.group:
                row_B.append(it1); row_.append(it1); col_.append(it1); bC_ = 0
                sum_mfstar_ = []
                for it3, it4 in it2.connect.items():
                    row_.append(it1); col_.append(it3)
                    SCf_ = geom_.Sf.vec(it1, it4)
                    dCf_ = geom_.dCf.scalar(it1, it4); dFf_ = geom_.dCf.scalar(it3, it4)
                    gC_ = dFf_ / (dCf_ + dFf_)
                    rho_ = gC_ * Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = 0) + (1 - gC_) * \
                        Prop.calc_rho(pool_.P.cunit[it3].current, pool_.T.cunit[it3].current, working_density_ = 0)
                    V_avg = gC_ * info_.cells[it1].size + (1 - gC_) * info_.cells[it3].size
                    Dauf_ = Pcorrect.calc_Dauf(it1, it3, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)

                    mfstar_ = round(Pcorrect.calc_mfstar(it1, it3, rho_, Dauf_, info_, pool_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit), 8)
                    sum_mfstar_.append(mfstar_) 
                    aF_ = rho_ * Dauf_ * V_avg

                    aC_pcor.append(aF_); aF_pcor.append(-aF_); bC_pcor.append(mfstar_)
                
                aC_ = np.sum(aC_pcor); bC_ = np.sum(bC_pcor)
                with_bound = -1
                for it3 in it2.bound:
                    for it4 in info_.faces[it3].group:
                        if it4 in list(bound_func.keys()):
                            with_bound = it3
                            aC_b, bC_b = bound_func[it4](it1, it3, A_u, A_v, A_w,
                                                         lambda_, delta_t, pool_, info_, geom_,
                                                         mfstar_ = sum_mfstar_, isinit = isinit)
                            aC_ += aC_b; bC_ += bC_b       

                for it3, it4 in it2.conj.items():
                    aC_b, bC_b = Boundary.noslip_pcorrect(it1, it4, A_u, A_v, A_w, lambda_, delta_t, pool_, info_, geom_, mfstar_ = sum_mfstar_, isinit = isinit)
                    aC_ += aC_b; bC_ += bC_b

                denom = -math.floor(math.log10(np.abs(aC_)))
                aC_ = aC_ * pow(10, denom + pcor_relax); aF_pcor = np.array([x * pow(10, denom + pcor_relax) for x in aF_pcor])
                
                # if with_bound >= 0:
                #     aF_pcor = np.array([0 for x in aF_pcor]); bC_ = aC_ * pool_.P.funit[with_bound].current
                
                # aC_ = np.abs(np.sum(aF_pcor))
                bC_ = -bC_ #round(bC_, 6)

                # aC_ = round(aC_, 6); aF_pcor = [round(x, 6) for x in aF_pcor]; bC_ = round(bC_, 6)
                # if any([bC_ >= 0, any([x > 0 for x in aF_pcor])]):
                # print(it1)
                # print(aC_, aF_pcor, '|', aC_ + np.sum(aF_pcor))
                # print('|', bC_)
                # try:
                #     print(it1, bC_ / (aC_ + np.sum(aF_pcor)))
                # except:
                #     pass
                # print()

                dataA_lil.append(aC_); dataA_lil.extend(aF_pcor); dataB_lil.append(bC_)

            else:
                row_B.append(it1)
                row_.append(it1); col_.append(it1)
                dataA_lil.append(float(0)); dataB_lil.append(float(0))
                approx.append(0)
            
        A_ = sparse.coo_matrix((dataA_lil, (row_, col_)), dtype = float).tocsr()
        B_ = sparse.coo_matrix((dataB_lil, (row_B, list(np.zeros(shape=(1, len(row_B)), dtype = int)[0]))), dtype = float).tocsr()
        
        return A_, B_, approx

    def correct_value(info_: Info, pool_: Pool, geom_: Geom, new_cunit: np.ndarray,
                      A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
                      lambda_: float, delta_t, isinit: bool = False):
        # correct SIMPLE, noslip, inlet, outlet values
        correct_func = {'inlet': Correct.inlet_correct_pcorrect, 'outlet': Correct.outlet_correct_pcorrect, 'noslip': Correct.noslip_correct_pcorrect}
        v_mdot_correct_func = {'inlet': Correct.v_mdot_correct_doublestar_inlet, 'outlet': Correct.v_mdot_correct_doublestar_outlet}

        # correct P
        for it1 in range(0, A_u.get_shape()[0]):
            it2 = info_.cells[it1]
            if 'fluid' in it2.group:
                pool_.P.cunit[it1].new = pool_.P.cunit[it1].current + new_cunit[it1]
                pool_.Pcor.cunit[it1].new = new_cunit[it1]
                for it3 in it2.bound:
                    [correct_func[it4](it1, it3, pool_, info_, geom_, A_u, A_v, A_w,
                                        lambda_, delta_t, isinit = isinit) for it4 \
                                        in info_.faces[it3].group if it4 in list(correct_func.keys())]
                if len(list(it2.conj.keys())) > 0:
                    for it3, it4 in it2.conj.items():
                        Correct.noslip_correct_pcorrect(it1, it4, pool_, info_, geom_, A_u, A_v, A_w,
                                                        lambda_, delta_t, isinit = isinit)
    
        pool_.P.update(info_, geom_)
        pool_.Pcor.update(info_, geom_)
        
        # correct v and mdot
        for it1, it2 in info_.cells.items():
            if 'fluid' in it2.group:
                Correct.v_correct_doublestar(it1, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)

        # pool_.u.renew_gradient(info_, geom_)
        # pool_.v.renew_gradient(info_, geom_)
        # pool_.w.renew_gradient(info_, geom_)

        clear_of_interpolation = []        
        for it1, it2 in info_.cells.items():
            if 'fluid' in it2.group:
                for it3, it4 in it2.connect.items():
                    if it4 not in clear_of_interpolation:
                        Momentum.rhie_chow_interpolate(it1, it3, it4, pool_, info_, geom_, A_u, A_v, A_w,
                                                                lambda_, delta_t, isinit = isinit)
                        clear_of_interpolation.append(it4)
                for it3 in it2.bound:
                    [v_mdot_correct_func[it4](it1, it3, pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit) \
                     for it4 in info_.faces[it3].group if it4 in list(v_mdot_correct_func.keys())]

        pool_.u.update(info_, geom_, with_grad = True)
        pool_.v.update(info_, geom_, with_grad = True)
        pool_.w.update(info_, geom_, with_grad = True)
        
        # v_ = np.array([(pool_.u.cunit[it1].current, pool_.v.cunit[it1].current,
        #                 pool_.w.cunit[it1].current, pool_.P.cunit[it1].current) \
        #                 for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group])
        # v_ = np.transpose(v_)
        
        # print('pcorrect - avg corrected u: %s, v: %s, w: %s, P: %s' % (np.mean(v_[0]), np.mean(v_[1]), np.mean(v_[2]), np.mean(v_[3])))
    
    def solve(pool_: Pool, info_: Info, geom_: Geom,
            A_u: sparse.csr_matrix, A_v: sparse.csr_matrix, A_w: sparse.csr_matrix,
            lambda_: float, delta_t, maxiter_: int, tol_: float, isinit: bool = False):
        A_Pcor, B_Pcor, approx_pcor = Pcorrect.calc_coef(pool_, info_, geom_, lambda_, A_u, A_v, A_w, delta_t, isinit = isinit)
        B_Pcor_arr = B_Pcor.toarray()

        n = A_Pcor.get_shape()[0]
        current_pcor = np.array([pool_.Pcor.cunit[it1].current for it1 in range(0, n)])

        try:
            ml = smoothed_aggregation_solver(A_Pcor, B = B_Pcor_arr)
            Apcor = ml.aspreconditioner(cycle = 'V')
            new_Pcor, exitCode = sparse.linalg.gmres(A_Pcor, B_Pcor_arr, x0 = current_pcor, M = Apcor, maxiter = maxiter_, atol = tol_, tol = tol_); new_Pcor = np.array(new_Pcor).ravel()
        except:
            new_Pcor, exitCode = sparse.linalg.gmres(A_Pcor, B_Pcor_arr, x0 = current_pcor, maxiter = maxiter_, atol = tol_, tol = tol_); new_Pcor = np.array(new_Pcor).ravel()

        # new_Pcor = np.array([-np.abs(x) for x in new_Pcor])
        new_Pcor = new_Pcor * (1 - lambda_)
        
        pcor_ = np.array([pool_.Pcor.cunit[it1].new \
                for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group])

        print('avg Pcor: %s (%s). Correcting...' % (np.mean(pcor_), exitCode))

        return new_Pcor, exitCode

class TKE:
    def calc_coef(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, channel_length, isinit: bool = False, issteady: bool = False):
        bound_func = {'inlet': Boundary.symmetry_tke, 'outlet': Boundary.symmetry_tke}
        approx = []; row = []; col = []; row_B = []
        dataA_lil = []; dataB_lil = []
        approx = []
        max_cfl_tke = 0
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        for it1, it2 in info_.cells.items():
            iswall = -1
            if len(list(it2.conj.keys())) > 0:
                iswall = it2.conj[list(it2.conj.keys())[0]]
            if 'fluid' in it2.group:
                row_B.append(it1)
                rho_ = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = 0)
                rho_last = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].last, working_density_ = 0)
                cp_ = Prop.calc_cp(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                cp_last = Prop.calc_cp(pool_.P.cunit[it1].last, pool_.T.cunit[it1].current)
                F1 = calc_F1(it1, pool_, info_)
                theta_k = 2 * F1 + (1 - F1)
                gammaC_ = Prop.calc_mu(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                gammaC_t = calc_mut(it1, pool_, info_)
                aC_nwf, aF_nwf, bC_nwf, row_, col_ = linear.NWF(pool_.k, it1, 'mu', 'tke', pool_, info_, geom_, user_, iswall = bool(iswall >= 0))
                aC_temporal, bC_temporal = linear.temporal_conv(pool_.k, pool_, info_, it1, delta_t, rho_ = rho_, rho_last = rho_last,
                                                                isinit_ = isinit)
                
                bC_tke = 0
                eps_tke = 0.09 * rho_ * pool_.k.cunit[it1].current * pool_.omega.cunit[it1].current
                if iswall >= 0:
                    n_ = np.array([0,1,0], dtype = float)
                    v_ = np.array([pool_.u.cunit[it1].current, pool_.v.cunit[it1].current, pool_.w.cunit[it1].current])
                    parallelv_ = np.linalg.norm(np.dot(v_, n_) * n_)            
                    utau_ = calc_u_tau(it1, -1, pool_, info_, gammaC_, rho_, parallelv_, info_.cells[it1].wdist)
                    ustar_ = calc_u_star(it1, -1, pool_, gammaC_, rho_, parallelv_, info_.cells[it1].wdist)
                    bC_tke += (rho_ * ustar_ * pow(utau_, 2) / (0.41 * info_.cells[it1].wdist)) - eps_tke
                elif iswall < 0:
                    Pk_tke = np.min([gammaC_t * Energy.calc_phi(it1, pool_), 10 * pool_.omega.cunit[it1].current * cp_ * pool_.k.cunit[it1].current])
                    bC_tke += (Pk_tke - eps_tke)
                
                bC_tke = bC_tke * info_.cells[it1].size        
                aC_bound = 0; bC_bound = 0

                for it3 in it2.bound:
                    for it4 in info_.faces[it3].group:
                        if it4 in list(bound_func.keys()):
                            aC_b, bC_b = bound_func[it4](pool_.k, it1, it3, gammaC_ + gammaC_t / theta_k, info_, pool_, geom_, user_)
                            aC_bound += aC_b; bC_bound += bC_b

                aC_trans, bC_trans = linear.relax_trans(pool_.k, pool_, info_, aC_nwf + aC_bound, it1, lambda_, delta_t, isinit_ = isinit)
                row.extend(row_), col.extend(col_)
                dataA_lil.append(aC_nwf + aC_bound + aC_trans + aC_temporal); dataA_lil.extend(aF_nwf)
                dataB_lil.append(bC_nwf + bC_bound + bC_trans + bC_temporal + bC_tke)
                
                max_cfl_tke = np.max([max_cfl_tke, (aC_nwf + aC_bound) * delta_t2 / (rho_last * info_.cells[it1].size)])

            else:
                row_B.append(it1)
                row.append(it1); col.append(it1)
                dataA_lil.append(float(0)); dataB_lil.append(float(0))
                approx.append(float(0))
            
        A_ = sparse.coo_matrix((dataA_lil, (row, col)), dtype = float).tocsr()
        B_ = sparse.coo_matrix((dataB_lil, (row_B, list(np.zeros(shape=(1, len(row_B)), dtype = int)[0]))), dtype = float).tocsr()

        return A_, B_, max_cfl_tke

    def correct_value(info_: Info, pool_: Pool, geom_: Geom, new_cunit: np.ndarray, exitCode):
        correct_func = {'noslip': Correct.noslip_tke_correct, 'inlet': Correct.symmetry_tke_correct, 'outlet': Correct.symmetry_tke_correct}
        for it1 in range(0, len(new_cunit)):
            if 'fluid' in info_.cells[it1].group:
                it2 = info_.cells[it1]
                pool_.k.cunit[it1].new = new_cunit[it1]
                for it3 in it2.bound:
                    [correct_func[it4](it1, it3, pool_, info_, geom_) for it4 in info_.faces[it3].group if it4 in list(correct_func.keys())]
                #pool_.P.cunit[it1].new = new_cunit[it1] * Prop.calc_rho(pool_.P.cunit[it1].current, new_cunit[it1]) * 287.052874
                        #if len(info_.faces[it3].group) == 0 or all([it4 in list(correct_func.keys()) for it4 in info_.faces[it3].group]) is False:
                        #    Correct.dirichlet_energy_correct(pool_.T, it1, it3)
                #if len(list(it2.conj.keys())) > 0:
                #    for it3, it4 in it2.conj.items():
                #        if new_cunit[it3] > new_cunit[it1]:
                #            pool_.T.cunit[it1].new = new_cunit[it3]
                #        pool_.T.funit[it4].new = pool_.T.cunit[it1].new
                        #Correct.conjugate_correct(it1, it3, it4, pool_)
        
        pool_.k.update(info_, geom_)

        # k_ = np.array([pool_.k.cunit[it1].current for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group])

        # print('avg k: %s (%s)' % (np.mean(k_), exitCode))

    def solve(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, channel_length, isinit: bool = False, issteady: bool = False):
        A_k, B_k, max_cfl_tke = TKE.calc_coef(pool_, info_, geom_, user_, lambda_, delta_t, channel_length, isinit = isinit, issteady = issteady); B_k_arr = B_k.toarray()
        n = A_k.get_shape()[0]
        current_k = np.array([pool_.k.cunit[it1].current for it1 in range(0, n)])
        
        # try:
        # ml = smoothed_aggregation_solver(A_k)
        # Ak = ml.aspreconditioner(cycle = 'V')
        # new_k, exitCode = sparse.linalg.gmres(A_k, B_k_arr, x0 = current_k, M = Ak, maxiter = maxiter_, atol = tol_, tol = tol_); new_k = np.array(new_k).ravel()
        # except:
        new_k, exitCode = sparse.linalg.gmres(A_k, B_k_arr, x0 = current_k, maxiter = maxiter_, atol = tol_, tol = tol_); new_k = np.array(new_k).ravel()     
        new_k = np.array([np.max([it1, 0]) for it1 in new_k])
        
        return A_k, B_k, new_k, exitCode, max_cfl_tke
  
class STKE:
    def calc_coef(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, channel_length, isinit: bool = False, issteady: bool = False):
        bound_func = {'inlet': Boundary.symmetry_stke, 'outlet': Boundary.symmetry_stke}
        approx = []; row = []; col = []; row_B = []
        dataA_lil = []; dataB_lil = []
        approx = []
        max_cfl_stke = 0
        delta_t2 = delta_t
        if isinit is False:
            delta_t2 = 2 * delta_t
        for it1, it2 in info_.cells.items():
            iswall = -1
            # if len(list(it2.conj.keys())) > 0:
            #     iswall = it2.conj[list(it2.conj.keys())[0]]
            if 'fluid' in it2.group:
                row_B.append(it1)
                rho_ = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, working_density_ = 0)
                rho_last = Prop.calc_rho(pool_.P.cunit[it1].current, pool_.T.cunit[it1].last, working_density_ = 0)
                cp_ = Prop.calc_cp(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                cp_last = Prop.calc_cp(pool_.P.cunit[it1].last, pool_.T.cunit[it1].current)
                F1 = calc_F1(it1, pool_, info_)
                theta_w = 2 * F1 + 1.186 * (1 - F1)
                gammaC_ = Prop.calc_mu(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current)
                gammaC_t = calc_mut(it1, pool_, info_)
                aC_nwf, aF_nwf, bC_nwf, row_, col_ = linear.NWF(pool_.omega, it1, 'mu', 'stke', pool_, info_, geom_, user_, iswall = False)
                aC_temporal, bC_temporal = linear.temporal_conv(pool_.omega, pool_, info_, it1, delta_t, rho_ = rho_, rho_last = rho_last,
                                                                isinit_ = isinit)
                
                c_alpha = 0.5532 * F1 + (1 - F1) * 0.4403
                c_beta = 0.075 * F1 + (1 - F1) * 0.0828
                
                try:
                    Pk_stke = c_alpha * pool_.omega.cunit[it1].current * np.min([gammaC_t * Energy.calc_phi(it1, pool_), 10 * pool_.omega.cunit[it1].current * cp_ * pool_.k.cunit[it1].current]) / pool_.k.cunit[it1].current
                    eps_stke = -c_beta * rho_ * pow(pool_.omega.cunit[it1].current, 2) + 2 * (1 - F1) * 1.186 * rho_ * np.dot(pool_.k.cgrad[it1].current, pool_.omega.cgrad[it1].current) / pool_.omega.cunit[it1].current
                    bC_stke = (Pk_stke + eps_stke) * info_.cells[it1].size
                except:
                    bC_stke = 0
                aC_bound = 0; bC_bound = 0
                
                for it3 in it2.bound:
                    for it4 in info_.faces[it3].group:
                        if it4 in list(bound_func.keys()):
                            aC_b, bC_b = bound_func[it4](pool_.omega, it1, it3, gammaC_ + gammaC_t / theta_w, gammaC_t, info_, pool_, geom_, user_)
                            aC_bound += aC_b; bC_bound += bC_b
                
                aC_trans, bC_trans = linear.relax_trans(pool_.omega, pool_, info_, aC_nwf + aC_bound, it1, lambda_, delta_t, isinit_ = isinit)
                row.extend(row_), col.extend(col_)
                
                # if iswall >= 0:
                #     aC_nwf = 0; aC_bound = 0; aC_trans = 0; aC_temporal = 0
                #     aF_nwf = np.array([0 for x in aF_nwf])
                #     bC_nwf = 0; bC_bound = 0; bC_trans = 0; bC_temporal = 0; bC_stke = 0
                
                dataA_lil.append(aC_nwf + aC_bound + aC_trans + aC_temporal); dataA_lil.extend(aF_nwf)
                dataB_lil.append(bC_nwf + bC_bound + bC_trans + bC_temporal + bC_stke)
                # try:
                #     print(aC_nwf, aF_nwf, np.sum(aF_nwf), bC_nwf, f'|', bC_nwf, (aC_nwf + np.sum(aF_nwf)), f'<- sum scheme zero')
                #     print(aC_trans, bC_trans, f'|', bC_trans / aC_trans, f'<- sum zero resulting init value')
                #     print(aC_temporal, bC_temporal, f'|', bC_temporal / aC_temporal, f'<- sum not equal (temporal term)')
                #     print(aC_bound, bC_bound, bC_stke, f'<- sum of other terms zero')
                #     print((bC_trans + bC_nwf + bC_temporal + bC_bound + bC_stke), (aC_nwf + np.sum(aF_nwf) + aC_trans + aC_temporal + aC_bound), f'|', (bC_trans + bC_nwf + bC_temporal + bC_bound + bC_stke) / (aC_nwf + np.sum(aF_nwf) + aC_trans + aC_temporal + aC_bound), f'<- numerical error in total')
                #     print()
                # except:
                #     pass
                max_cfl_stke = np.max([max_cfl_stke, (aC_nwf + aC_bound) * delta_t2 / (rho_last * info_.cells[it1].size)])

            else:
                row_B.append(it1)
                row.append(it1); col.append(it1)
                dataA_lil.append(float(0)); dataB_lil.append(float(0))
                approx.append(float(0))
            
        A_ = sparse.coo_matrix((dataA_lil, (row, col)), dtype = float).tocsr()
        B_ = sparse.coo_matrix((dataB_lil, (row_B, list(np.zeros(shape=(1, len(row_B)), dtype = int)[0]))), dtype = float).tocsr()
        
        return A_, B_, max_cfl_stke

    def correct_value(info_: Info, pool_: Pool, geom_: Geom, new_cunit: np.ndarray, exitCode):
        correct_func = {'inlet': Correct.symmetry_stke_correct, 'outlet': Correct.symmetry_stke_correct}
        for it1 in range(0, len(new_cunit)):
            if 'fluid' in info_.cells[it1].group:
                it2 = info_.cells[it1]
                pool_.omega.cunit[it1].new = new_cunit[it1]
                for it3, it4 in info_.cells[it1].conj.items():
                    Correct.noslip_stke_correct(it1, it4, pool_, info_, geom_)
                for it3 in it2.bound:
                    [correct_func[it4](it1, it3, pool_, info_, geom_) for it4 in info_.faces[it3].group if it4 in list(correct_func.keys())]

        pool_.omega.update(info_, geom_)

        # omega_ = np.array([pool_.omega.cunit[it1].current for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group])

        # print('avg omega: %s (%s)' % (np.mean(omega_), exitCode))

    def solve(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, channel_length, isinit: bool = False, issteady: bool = False):
        A_omega, B_omega, max_cfl_stke = STKE.calc_coef(pool_, info_, geom_, user_, lambda_, delta_t, channel_length, isinit = isinit, issteady = issteady); B_omega_arr = B_omega.toarray()
        n = A_omega.get_shape()[0]
        current_omega = np.array([pool_.omega.cunit[it1].current for it1 in range(0, n)])
        
        # try:
        # ml = smoothed_aggregation_solver(A_omega)
        # Aomega = ml.aspreconditioner(cycle = 'V')
        # new_omega, exitCode = sparse.linalg.gmres(A_omega, B_omega_arr, x0 = current_omega, M = Aomega, maxiter = maxiter_, atol = tol_, tol = tol_); new_omega = np.array(new_omega).ravel()
        # except:
        new_omega, exitCode = sparse.linalg.gmres(A_omega, B_omega_arr, x0 = current_omega, maxiter = maxiter_, atol = tol_, tol = tol_); new_omega = np.array(new_omega).ravel()     
        
        new_omega = new_omega * 1e-6
        new_omega = np.array([np.max([it1, 0]) for it1 in new_omega])
        
        return A_omega, B_omega, new_omega, exitCode, max_cfl_stke


"""
SOLVE
-----
"""

def check_iter_res(theta_, A_, B_, tol_: float):
    # new corrected v values = x, RMSR(A(n) x* - B(n))
    # forward time with x*
    x_ = np.array([[theta_.cunit[it1].current for it1 in range(0, B_.shape[0])]])
    #xprev_ = np.array([[theta_.cunit[it1].prev for it1 in range(0, B_.shape[0])]])
    x_ = np.transpose(x_)
    #xprev_ = np.transpose(xprev_)
    Bstar_ = A_ * x_
    #Bprev_ = A_ * xprev_
    Bstar_ = np.array(Bstar_).ravel()
    #Bprev_ = np.array(Bprev_).ravel()
    rmsr = np.sum(np.array([round((B_[it1, 0] - Bstar_[it1]), 3)**2 for it1 in range(0, B_.shape[0])])) / B_.shape[0]
    rmsr = np.sqrt(rmsr)
    return bool(rmsr < tol_), rmsr

def check_time_res(theta_, A_, B_, tol_: float):
    # if passed check_iter_res, RMSR(A(n) xoo - B(n))
    # stop
    x_ = np.array([[theta_.cunit[it1].current for it1 in range(0, B_.shape[0])]])
    xlast_ = np.array([[theta_.cunit[it1].last for it1 in range(0, B_.shape[0])]])
    x_ = np.transpose(x_); xlast_ = np.transpose(xlast_)
    Bstar_ = A_ * x_; Blast_ = A_ * xlast_
    Bstar_ = np.array(Bstar_).ravel(); Blast_ = np.array(Blast_).ravel()
    rmsr = np.sum(np.array([round((Bstar_[it1] - Blast_[it1]), 3)**2 for it1 in range(0, B_.shape[0])])) / B_.shape[0]
    rmsr = np.sqrt(rmsr)
    return bool(rmsr < tol_), rmsr

def burn_in_energy(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, maxiterstep_: int, isinit: bool = False):
    stop_cond = False; iterCode = False; mainiter = 0
    rmsr_T = 0
    while stop_cond is False:
        A_T, B_T, new_T = Energy.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, tol_, isinit = isinit)
        Energy.correct_value(info_, pool_, new_T)
        pool_.T.update(info_, geom_)
        iterCode_T, rmsr_T = check_iter_res(pool_.T, A_T, B_T, tol_)
        mainiter += 1
        print('RMSR burn-in T: %s [%s/%s]' % (rmsr_T, mainiter, maxiterstep_))
        #print(iterCode_T)
        iterCode = iterCode_T
        stop_cond = any([iterCode_T, mainiter > maxiterstep_-1])
    print('\n')
    return iterCode, rmsr_T 

def SIMPLEC(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, maxiterstep_: int,
            channel_length, isinit: bool = False):
    
    stop_cond = False; iterCode_time = False; mainiter1 = 1; mainiter2 = 1
    iterCode_u = 0; iterCode_v = 0; iterCode_w = 0; iterCode_T = 1
    rmsr_u_time = 0; rmsr_v_time = 0; rmsr_w_time = 0
    max_cfl_u = 0; max_cfl_v = 0; max_cfl_w = 0; max_cfl_T = 0; max_cfl_tke = 0; max_cfl_stke = 0

    main_inlet = []; main_outlet = []; main_avg = []; abs_T_avg = []; glass_T_avg = []

    max_cfl_list = [0, 0, 0, 0, 0, 0]

    while any([all([iterCode_u, iterCode_v, iterCode_w, iterCode_T]), bool(mainiter1 > maxiterstep_)]) is False:
        A_u, B_u, new_u, exitCode_u, max_cfl_u = Momentum.solve(0, pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, tol_, isinit = isinit)
        A_v, B_v, new_v, exitCode_v, max_cfl_v = Momentum.solve(1, pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, tol_, isinit = isinit)
        A_w, B_w, new_w, exitCode_w, max_cfl_w = Momentum.solve(2, pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, tol_, isinit = isinit)
        
        Momentum.correct_value(info_, pool_, geom_, new_u, new_v, new_w, A_u, A_v, A_w, lambda_, delta_t, [exitCode_u, exitCode_v, exitCode_w], isinit = isinit)

        iterCode_u, rmsr_u = check_iter_res(pool_.u, A_u, B_u, tol_)
        iterCode_v, rmsr_v = check_iter_res(pool_.v, A_v, B_v, tol_)
        iterCode_w, rmsr_w = check_iter_res(pool_.w, A_w, B_w, tol_)

        # pool_.u.update(info_, geom_, with_grad = False)
        # pool_.v.update(info_, geom_, with_grad = False)
        # pool_.w.update(info_, geom_, with_grad = False)

        new_Pcor, exitPcor = Pcorrect.solve(pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, maxiter_, tol_, isinit = isinit)
        Pcorrect.correct_value(info_, pool_, geom_, new_Pcor, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)

        A_T, B_T, new_T, exitCode_T, max_cfl_T = Energy.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, tol_, channel_length, isinit = isinit)
        Energy.correct_value(info_, pool_, geom_, new_T, exitCode_T)

        iterCode_T, rmsr_T = check_iter_res(pool_.T, A_T, B_T, tol_)

        A_k, B_k, new_k, exitCode_k, max_cfl_tke = TKE.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, channel_length, isinit = isinit)
        TKE.correct_value(info_, pool_, geom_, new_k, exitCode_k)

        A_omega, B_omega, new_omega, exitCode_omega, max_cfl_stke = STKE.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, channel_length, isinit = isinit)
        STKE.correct_value(info_, pool_, geom_, new_omega, exitCode_omega)

        print('RMSR iter u: %s, v: %s, w: %s, T: %s [%s/%s]\n' % (rmsr_u, rmsr_v, rmsr_w, rmsr_T, mainiter1, maxiterstep_))
        #print('RMSR iter u: %s, v: %s, w: %s, T: %s [%s/%s]\n' % (rmsr_u, rmsr_v, rmsr_w, 0, mainiter1, maxiterstep_))
        # print('RMSR iter u: %s, v: %s, w: %s, T: %s [%s/%s]\n' % (0, 0, 0, rmsr_T, mainiter1, maxiterstep_))
        #print('RMSR iter T: %s [%s/%s]\n' % (rmsr_T, mainiter1, maxiterstep_))

        # fluid_cell_list = [it1 for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group] 
        # fluid_face_list = []; [fluid_face_list.extend(info_.cells[it1].lface) for it1 in fluid_cell_list]; fluid_face_list = list(np.unique(np.array(fluid_face_list)))

        # #fluid_temp = [pool_.T.cunit[it1].current for it1 in fluid_cell_list]
        # #include_cell_id = [it1 for it1 in fluid_cell_list if pool_.T.cunit[it1].current <= np.mean(fluid_temp)]

        # abs_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'abs' in info_.cells[it1].group]
        # glass_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'glass' in info_.cells[it1].group]
        # insul_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'insul' in info_.cells[it1].group]
        # #abs_avg = [it1 for it1 in abs_T_avg if it1 <= np.mean(abs_T_avg) * 1.2]
        # #glass_avg = [it1 for it1 in glass_T_avg if it1 <= np.mean(glass_T_avg) * 1.2]

        # #outlet_id = [it1 for it1 in fluid_face_list if 'outlet' in info_.faces[it1].group]
        # #outlet_temp = [pool_.T.funit[it1].current for it1 in outlet_id]
        # #include_outlet_id = [it1 for it1 in outlet_id if pool_.T.funit[it1].current <= np.mean(outlet_temp) * 2]
        # #include_outlet_id = [it1 for it1 in outlet_id if pool_.T.funit[it1].current < np.max(outlet_temp)]

        # main_inlet = np.array([(pool_.u.funit[it1].current, pool_.v.funit[it1].current, pool_.w.funit[it1].current, pool_.P.funit[it1].current, pool_.T.funit[it1].current, pool_.k.funit[it1].current, pool_.omega.funit[it1].current) for it1 in fluid_face_list if 'inlet' in info_.faces[it1].group])
        # main_outlet = np.array([(pool_.u.funit[it1].current, pool_.v.funit[it1].current, pool_.w.funit[it1].current, pool_.P.funit[it1].current, pool_.T.funit[it1].current, pool_.k.funit[it1].current, pool_.omega.funit[it1].current) for it1 in fluid_face_list if 'outlet' in info_.faces[it1].group])
        # main_avg = np.array([(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, pool_.k.cunit[it1].current, pool_.omega.cunit[it1].current) for it1 in fluid_cell_list])

        # main_inlet = np.transpose(main_inlet); main_outlet = np.transpose(main_outlet); main_avg = np.transpose(main_avg)

        mainiter1 += 1
        """
        print('Fluid main channel summary (avg.)\n---------------------------------')
        print('Inlet: u: %s, v: %s, w: %s, P %s, T %s' % (np.mean(main_inlet[0]), np.mean(main_inlet[1]), np.mean(main_inlet[2]), np.mean(main_inlet[3]), np.mean(main_inlet[4])))
        print('Outlet: u: %s, v: %s, w: %s, P %s, T: %s' % (np.mean(main_outlet[0]), np.mean(main_outlet[1]), np.mean(main_outlet[2]), np.mean(main_outlet[3]), np.mean(main_outlet[4])))
        print('Channel: P %s, T: %s\n' % (np.mean(main_avg[0]), np.mean(main_avg[1])))
        #print('outlet avg. %s, channel avg. %s' % (np.mean(outlet_temp), np.mean(fluid_temp)))
        print('Solid summary (avg.)\n--------------------')
        print('Absorber: %s, Glass: %s\n' % (np.mean(abs_T_avg), np.mean(glass_T_avg)))
        """

        max_cfl_new = [max_cfl_u, max_cfl_v, max_cfl_w, max_cfl_T, max_cfl_tke, max_cfl_stke]
        max_cfl_list = [np.max([max_cfl_list[x], max_cfl_new[x]]) for x in range(0, 6)]

    iterCode_u_time, rmsr_u_time = check_time_res(pool_.u, A_u, B_u, tol_)
    iterCode_v_time, rmsr_v_time = check_time_res(pool_.v, A_v, B_v, tol_)
    iterCode_w_time, rmsr_w_time = check_time_res(pool_.w, A_w, B_w, tol_)
    iterCode_T_time, rmsr_T_time = check_time_res(pool_.T, A_T, B_T, tol_)
    
    iterCode_time = all([iterCode_u_time, iterCode_v_time, iterCode_w_time, iterCode_T_time])    
    #iterCode_time = all([iterCode_u_time, iterCode_v_time, iterCode_w_time, True])
    # iterCode_time = all([True, True, True, iterCode_T_time]) 

    # print('\nFluid main channel summary (avg.)\n---------------------------------')
    # print('Inlet: u: %s, v: %s, w: %s, P %s, T %s, k: %s, omega: %s' % (np.mean(main_inlet[0]), np.mean(main_inlet[1]), np.mean(main_inlet[2]), np.mean(main_inlet[3]), np.mean(main_inlet[4]), np.mean(main_inlet[5]), np.mean(main_inlet[6])))
    # print('Outlet: u: %s, v: %s, w: %s, P %s, T: %s, k: %s, omega: %s' % (np.mean(main_outlet[0]), np.mean(main_outlet[1]), np.mean(main_outlet[2]), np.mean(main_outlet[3]), np.mean(main_outlet[4]), np.mean(main_outlet[5]), np.mean(main_outlet[6])))
    # print('Channel: P %s, T: %s, k: %s, omega: %s\n' % (np.mean(main_avg[0]), np.mean(main_avg[1]), np.mean(main_avg[2]), np.mean(main_avg[3])))
    # #print('outlet avg. %s, channel avg. %s' % (np.mean(outlet_temp), np.mean(fluid_temp)))
    # print('Solid summary (avg.)\n--------------------')
    # print('Absorber: %s, Glass: %s, Insulation: %s\n' % (np.mean(abs_T_avg), np.mean(glass_T_avg), np.mean(insul_T_avg)))
    print('Max. CFL; u: %s, v: %s, w: %s, T: %s, k: %s, omega: %s\n' % (max_cfl_list[0], max_cfl_list[1], max_cfl_list[2], max_cfl_list[3], max_cfl_list[4], max_cfl_list[5]))

    return iterCode_time, rmsr_u_time, rmsr_v_time, rmsr_w_time, rmsr_T_time, max_cfl_list

def SC_transient_solver(pool_: Pool, info_: Info, geom_: Geom, user_: User, export_: Export, lambda_: float, delta_t_, maxiter_: int, tol_: float, maxiterstep_: int,
                     channel_length, geom_external, qabs_, qglass_, clock_result):
    unique_layer_ids = [it2.layer for it1, it2 in info_.cells.items()]
    unique_layer_ids = np.unique(unique_layer_ids)

    use_columns = deepcopy(export_.column_list['fluid']); use_columns.extend(export_.column_list['io'])
    temp = pd.DataFrame(columns = use_columns)
    df_fluid_ = {'u': deepcopy(temp), 'v': deepcopy(temp), 'w': deepcopy(temp), 'P': deepcopy(temp), 'T': deepcopy(temp), 'k': deepcopy(temp), 'omega': deepcopy(temp)}
    df_solid_ = dict.fromkeys(['T'], pd.DataFrame(columns = export_.column_list['solid']))
    
    time_start_main = process_time()

    max_cfl_total = [0, 0, 0, 0, 0, 0]
    rmsr_time = [0, 0, 0, 0]

    for qstep in range(0, len(qglass_)):
        time_start = process_time()
        user_.solid_q = dict({'glass': qglass_[qstep], 'abs': qabs_[qstep]})
        iterCode_time, rmsr_u_time, rmsr_v_time, rmsr_w_time, rmsr_T_time, max_cfl_list = SIMPLEC(pool_, info_, geom_, user_, lambda_, delta_t_, maxiter_, tol_, maxiterstep_,
                                                                                            channel_length, isinit = bool(qstep == 0))
        max_cfl_total = [np.max([max_cfl_total[x], max_cfl_list[x]]) for x in range(0, 6)]
        rmsr_time = [rmsr_u_time, rmsr_v_time, rmsr_w_time, rmsr_T_time]
        time_stop = process_time()
        processtime = (time_stop - time_start)
        if (qstep+1) % clock_result == 0:
            print('Generating entry...\n')
            export_.generate_dataset_entry(df_fluid_, df_solid_, pool_)
        print("Time %s\n-------\nRMSR at iter; u: %s, v: %s, w: %s, T: %s [%s s]\n" % (qstep+1, rmsr_u_time, rmsr_v_time, rmsr_w_time, rmsr_T_time, processtime))
        pool_.u.forward(); pool_.v.forward(); pool_.w.forward(); pool_.T.forward(); pool_.P.forward(); pool_.Pcor.forward(); pool_.k.forward(); pool_.omega.forward()

    time_stop_main = process_time()
    processtime = (time_stop_main - time_start_main) / 60
    print("\n")
    elapsed_time = (2 * len(qglass_) - 1) * delta_t_
    print("Done at %s s (%s m)\n" % (elapsed_time, processtime))
    result_ = Results(info_, df_fluid_, df_solid_, export_, delta_t_, lambda_, tol_, geom_external, qabs_, qglass_)
    
    return result_, max_cfl_total, processtime, rmsr_time

def SIMPLEC_steady(pool_: Pool, info_: Info, geom_: Geom, user_: User, lambda_: float, delta_t, maxiter_: int, tol_: float, maxiterstep_: int,
            channel_length, isinit: bool = False):
    #iterCode, rmsr_T = burn_in_energy(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, tol_, maxiterstep_, isinit = isinit)
    #print('RMSR burn-in T: %s\n' % rmsr_T)

    stop_cond = False; iterCode_time = False; mainiter1 = 1; mainiter2 = 1
    iterCode_u = 0; iterCode_v = 0; iterCode_w = 0; iterCode_T = 1
    rmsr_u_time = 0; rmsr_v_time = 0; rmsr_w_time = 0

    main_inlet = []; main_outlet = []; main_avg = []; abs_T_avg = []; glass_T_avg = []

    while any([iterCode_time, bool(mainiter1 > maxiterstep_)]) is False:
        A_u, B_u, new_u, exitCode_u = Momentum.solve(0, pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, isinit = isinit)
        A_v, B_v, new_v, exitCode_v = Momentum.solve(1, pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, isinit = isinit)
        A_w, B_w, new_w, exitCode_w = Momentum.solve(2, pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, isinit = isinit)
        
        Momentum.correct_value(info_, pool_, geom_, new_u, new_v, new_w, A_u, A_v, A_w, lambda_, delta_t, [exitCode_u, exitCode_v, exitCode_w], isinit = isinit)

        iterCode_u, rmsr_u = check_iter_res(pool_.u, A_u, B_u, tol_)
        iterCode_v, rmsr_v = check_iter_res(pool_.v, A_v, B_v, tol_)
        iterCode_w, rmsr_w = check_iter_res(pool_.w, A_w, B_w, tol_)

        new_Pcor, exitPcor = Pcorrect.solve(pool_, info_, geom_, A_u, A_v, A_w, lambda_, delta_t, maxiter_, 1e-4, isinit = isinit)
        # if exitPcor == 0:
        #     new_Pcor = np.zeros(shape=(1, len(new_T)))[0]
        Pcorrect.correct_value(info_, pool_, geom_, new_Pcor, A_u, A_v, A_w, lambda_, delta_t, isinit = isinit)

        A_T, B_T, new_T, exitCode_T = Energy.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, channel_length, isinit = isinit, issteady = True)
        Energy.correct_value(info_, pool_, geom_, new_T, exitCode_T)

        iterCode_T, rmsr_T = check_iter_res(pool_.T, A_T, B_T, tol_)

        A_k, B_k, new_k, exitCode_k = TKE.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, channel_length, isinit = isinit)
        TKE.correct_value(info_, pool_, geom_, new_k, exitCode_k)

        A_omega, B_omega, new_omega, exitCode_omega = STKE.solve(pool_, info_, geom_, user_, lambda_, delta_t, maxiter_, 1e-4, channel_length, isinit = isinit)
        STKE.correct_value(info_, pool_, geom_, new_omega, exitCode_omega)

        # iterCode_u_time, rmsr_u_time = check_time_res(pool_.u, A_u, B_u, tol_)
        # iterCode_v_time, rmsr_v_time = check_time_res(pool_.v, A_v, B_v, tol_)
        # iterCode_w_time, rmsr_w_time = check_time_res(pool_.w, A_w, B_w, tol_)

        print('RMSR iter u: %s, v: %s, w: %s, T: %s [%s/%s]\n' % (rmsr_u, rmsr_v, rmsr_w, rmsr_T, mainiter1, maxiterstep_))
        #print('RMSR iter u: %s, v: %s, w: %s, T: %s [%s/%s]\n' % (rmsr_u, rmsr_v, rmsr_w, 0, mainiter1, maxiterstep_))
        #print('RMSR iter u: %s, v: %s, w: %s, T: %s [%s/%s]\n' % (0, 0, 0, rmsr_T, mainiter1, maxiterstep_))
        #print('RMSR iter T: %s [%s/%s]\n' % (rmsr_T, mainiter1, maxiterstep_))

        #pool_.u.forward(); pool_.v.forward(); pool_.w.forward(); pool_.T.forward(); pool_.P.forward(); pool_.Pcor.forward()
        
        mainiter1 += 1
        iterCode_time = all([iterCode_u, iterCode_v, iterCode_w, iterCode_T]) 

        
        # print('Fluid main channel summary (avg.)\n---------------------------------')
        # print('Inlet: u: %s, v: %s, w: %s, P %s, T %s' % (np.mean(main_inlet[0]), np.mean(main_inlet[1]), np.mean(main_inlet[2]), np.mean(main_inlet[3]), np.mean(main_inlet[4])))
        # print('Outlet: u: %s, v: %s, w: %s, P %s, T: %s' % (np.mean(main_outlet[0]), np.mean(main_outlet[1]), np.mean(main_outlet[2]), np.mean(main_outlet[3]), np.mean(main_outlet[4])))
        # print('Channel: P %s, T: %s\n' % (np.mean(main_avg[0]), np.mean(main_avg[1])))
        # #print('outlet avg. %s, channel avg. %s' % (np.mean(outlet_temp), np.mean(fluid_temp)))
        # print('Solid summary (avg.)\n--------------------')
        # print('Absorber: %s, Glass: %s\n' % (np.mean(abs_T_avg), np.mean(glass_T_avg)))
        

    # #iterCode_time = all([iterCode_u_time, iterCode_v_time, iterCode_w_time, iterCode_T_time])    
    # #iterCode_time = all([iterCode_u_time, iterCode_v_time, iterCode_w_time, True])

    fluid_cell_list = [it1 for it1 in list(info_.cells.keys()) if 'fluid' in info_.cells[it1].group] 
    fluid_face_list = []; [fluid_face_list.extend(info_.cells[it1].lface) for it1 in fluid_cell_list]; fluid_face_list = list(np.unique(np.array(fluid_face_list)))

    #fluid_temp = [pool_.T.cunit[it1].current for it1 in fluid_cell_list]
    #include_cell_id = [it1 for it1 in fluid_cell_list if pool_.T.cunit[it1].current <= np.mean(fluid_temp)]

    abs_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'abs' in info_.cells[it1].group]
    glass_T_avg = [pool_.T.cunit[it1].current for it1 in list(info_.cells.keys()) if 'glass' in info_.cells[it1].group]
    #abs_avg = [it1 for it1 in abs_T_avg if it1 <= np.mean(abs_T_avg) * 1.2]
    #glass_avg = [it1 for it1 in glass_T_avg if it1 <= np.mean(glass_T_avg) * 1.2]

    #outlet_id = [it1 for it1 in fluid_face_list if 'outlet' in info_.faces[it1].group]
    #outlet_temp = [pool_.T.funit[it1].current for it1 in outlet_id]
    #include_outlet_id = [it1 for it1 in outlet_id if pool_.T.funit[it1].current <= np.mean(outlet_temp) * 2]
    #include_outlet_id = [it1 for it1 in outlet_id if pool_.T.funit[it1].current < np.max(outlet_temp)]

    main_inlet = np.array([(pool_.u.funit[it1].current, pool_.v.funit[it1].current, pool_.w.funit[it1].current, pool_.P.funit[it1].current, pool_.T.funit[it1].current, pool_.k.funit[it1].current, pool_.omega.funit[it1].current) for it1 in fluid_face_list if 'inlet' in info_.faces[it1].group])
    main_outlet = np.array([(pool_.u.funit[it1].current, pool_.v.funit[it1].current, pool_.w.funit[it1].current, pool_.P.funit[it1].current, pool_.T.funit[it1].current, pool_.k.funit[it1].current, pool_.omega.funit[it1].current) for it1 in fluid_face_list if 'outlet' in info_.faces[it1].group])
    main_avg = np.array([(pool_.P.cunit[it1].current, pool_.T.cunit[it1].current, pool_.k.cunit[it1].current, pool_.omega.cunit[it1].current) for it1 in fluid_cell_list])

    main_inlet = np.transpose(main_inlet); main_outlet = np.transpose(main_outlet); main_avg = np.transpose(main_avg)

    print('\nFluid main channel summary (avg.)\n---------------------------------')
    print('Inlet: u: %s, v: %s, w: %s, P %s, T %s, k: %s, omega: %s' % (np.mean(main_inlet[0]), np.mean(main_inlet[1]), np.mean(main_inlet[2]), np.mean(main_inlet[3]), np.mean(main_inlet[4]), np.mean(main_inlet[5]), np.mean(main_inlet[6])))
    print('Outlet: u: %s, v: %s, w: %s, P %s, T: %s, k: %s, omega: %s' % (np.mean(main_outlet[0]), np.mean(main_outlet[1]), np.mean(main_outlet[2]), np.mean(main_outlet[3]), np.mean(main_outlet[4]), np.mean(main_outlet[5]), np.mean(main_outlet[6])))
    print('Channel: P %s, T: %s, k: %s, omega: %s\n' % (np.mean(main_avg[0]), np.mean(main_avg[1]), np.mean(main_avg[2]), np.mean(main_avg[3])))
    #print('outlet avg. %s, channel avg. %s' % (np.mean(outlet_temp), np.mean(fluid_temp)))
    print('Solid summary (avg.)\n--------------------')
    print('Absorber: %s, Glass: %s\n' % (np.mean(abs_T_avg), np.mean(glass_T_avg)))

    return

def SC_steady_solver(pool_: Pool, info_: Info, geom_: Geom, user_: User, export_: Export, lambda_: float, delta_t_, maxiter_: int, tol_: float, maxiterstep_: int,
                     channel_length, geom_external, qabs_, qglass_):
    unique_layer_ids = [it2.layer for it1, it2 in info_.cells.items()]
    unique_layer_ids = np.unique(unique_layer_ids)

    temp = pd.DataFrame(columns = export_.column_list['fluid'])
    df_fluid_ = {'u': deepcopy(temp), 'v': deepcopy(temp), 'w': deepcopy(temp), 'P': deepcopy(temp), 'T': deepcopy(temp), 'k': deepcopy(temp), 'omega': deepcopy(temp)}
    df_solid_ = dict.fromkeys(['T'], pd.DataFrame(columns = export_.column_list['solid']))
    
    time_start_main = process_time()

    time_start = process_time()
    user_.solid_q = dict({'glass': qglass_[0], 'abs': qabs_[0]})

    SIMPLEC_steady(pool_, info_, geom_, user_, lambda_, delta_t_, maxiter_, tol_, maxiterstep_, channel_length)
    
    time_stop = process_time()
    processtime = (time_stop - time_start)

    print('Generating entry...\n')
    export_.generate_dataset_entry(df_fluid_, df_solid_, pool_)

    time_stop_main = process_time()
    processtime = (time_stop_main - time_start_main) / 60
    print("\n")
    elapsed_time = (2 * len(qglass_) - 1) * delta_t_
    print("Done at %s s (%s m)\n" % (elapsed_time, processtime))
    result_ = Results(info_, df_fluid_, df_solid_, export_, delta_t_, lambda_, tol_, geom_external, qabs_, qglass_)
    
    return result_

def initialize(fileloc: str, extrude_len: float, extrude_part: int, channel_gap: float, nhorizontal: int, P_: float = 0, Pinlet_: float = 0, Poutlet_: float = 0,
               T_: float = 300, v_: float = 0.0, w_: float = 0.0, mdot_: float = 0.0):
    print('initializing...')
    info_ = Info(fileloc, extrude_len, extrude_part, nhorizontal)
    geom_ = Geom(info_)
    pool_ = Pool(info_, geom_, P_ = P_, v_ = v_, w_ = w_, T_ = T_, mdot_ = mdot_)
    user_ = User(0, 0)
    export_ = Export.find_member_layers(info_)
    for it1, it2 in info_.cells.items():
        it2.wdist = np.abs(np.min([np.abs(it2.loc[2] - (channel_gap / 2)), np.abs(it2.loc[2] + (channel_gap / 2))]))
    for it1, it2 in info_.faces.items():
        if 'inlet' in it2.group:
            pool_.P.funit[it1].new = Pinlet_
        elif 'outlet' in it2.group:
            pool_.P.funit[it1].new = Poutlet_
    pool_.P.update(info_, geom_)
    return info_, geom_, pool_, user_, export_

def solve(info_: Info, geom_: Geom, pool_: Pool, user_: User, export_: Export, channel_length, geom_external, qabs_, qglass_, case_name, result_loc, lambda_: float = 0.8, delta_t_ = 30, maxiter_: int = 250,
          tol_: float = 5e-4, maxiterstep_: int = 5, clock_result: int = 5, save_: bool = False):
    print('solving...\n')
    result_, max_cfl_total, processtime, rmsr_time = SC_transient_solver(pool_, info_, geom_, user_, export_, lambda_, delta_t_, maxiter_,
                                                              tol_, maxiterstep_, channel_length, geom_external, qabs_, qglass_, clock_result)
    
    if save_ is True:
        if not os.path.exists(f'{result_loc}\\{case_name}'):
            try:
                original_umask = os.umask(0)
                os.makedirs(f'{result_loc}\\{case_name}', 0o777)
            finally:
                os.umask(original_umask)
        os.chdir(f'{result_loc}\\{case_name}')
        print('Saving results...')
        filehandler = open(f'{case_name}_info', 'wb'); pickle.dump(info_, filehandler); filehandler.close()
        filehandler = open(f'{case_name}_geom', 'wb'); pickle.dump(geom_, filehandler); filehandler.close()
        filehandler = open(f'{case_name}_pool', 'wb'); pickle.dump(pool_, filehandler); filehandler.close()
        filehandler = open(f'{case_name}_result', 'wb'); pickle.dump(result_, filehandler); filehandler.close()
        print(f'Saved to {result_loc}\\{case_name}.\n')
        for it1, it2 in result_.fluid.items():
            it2.to_csv(f'{result_loc}\\{case_name}\\{case_name}_fluid_{it1}.csv')
        result_.solid['T'].to_csv(f'{result_loc}\\{case_name}\\{case_name}_solid_T.csv')
        text_file = open(f'{case_name}_report.txt', "wt")
        n = text_file.write(f'CFL\n---------\nu: {max_cfl_total[0]}; v: {max_cfl_total[1]}; w: {max_cfl_total[2]}; T: {max_cfl_total[3]}; k: {max_cfl_total[4]}, omega: {max_cfl_total[5]}\nRMSR\n---------\nu: {rmsr_time[0]}; v: {rmsr_time[1]}; w: {rmsr_time[2]}; T: {rmsr_time[3]}\nDone in {processtime} m.')
        text_file.close()
    return

def case_manager(mesh_loc, case_no: int, nhorizontal_: int, irr: float, lambda_: float, tol_: float, delta_t: float, time_step: int, count_step: int):
    meshloc = os.getcwd() + f'\\{mesh_loc}\\mesh{case_no}.msh'
    if not os.path.exists(f'\\data'):
        try:
            original_umask = os.umask(0)
            os.makedirs(f'', 0o777)
        finally:
            os.umask(original_umask)
    resultloc = f'\\data'
    casename = f'case{case_no}_irr{irr}'
    
    # glass transmissivity 0.86, reflectivity 0.08
    qabs_ = [irr]*time_step
    qglass_ = [irr * (1 - 0.86)]*time_step
    geom_external = [0.022, 0.022, 0.050, 2.5, 2.5] # abs, glass, insul, fluid area, channel length in meters

    info_, geom_, pool_, user_, export_ = initialize(meshloc, 2.5, 11, 0.5, nhorizontal_, T_ = 300, v_ = 0.001, w_ = 0.0, mdot_ = 0, P_ = 0, Pinlet_ = 0, Poutlet_ = 0) # I = 200 w/m2, abs 163.4
    solve(info_, geom_, pool_, user_, export_, 5, geom_external, qabs_, qglass_, casename, resultloc, lambda_ = lambda_,
        delta_t_ = delta_t, maxiter_ = 20, tol_ = tol_, maxiterstep_ = 1, clock_result = count_step, save_ = True)

mesh_folder = 'mesh'
case_manager(mesh_loc, 1, 7, 100, 0.9, 1e-4, 5, 3, 3)

# import matplotlib.pyplot as plt

# meshloc = f'C:\\Users\\SolVer\\Downloads\\old\\fuck_git\\mesh\\mesh1.msh'
# resultloc = f'C:\\Users\\Solver\\Downloads\\old\\fuck_git\\results\\irr100'
# info_, geom_, pool_, user_, export_ = initialize(meshloc, 2.5, 11, 0.5, 7, T_ = 300, v_ = 0.001, w_ = 0.0, mdot_ = 0, P_ = 0, Pinlet_ = 0, Poutlet_ = 0) # I = 200 w/m2, abs 163.4

# node_list = []
# for it1, it2 in info_.nodes.items():
#     node_list.append(list(it2.loc))
# # for it1, it2 in info_.faces.items():
# #     node_list.append(list(it2.loc))

# node_list = np.transpose(np.array(node_list))
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(node_list[0], node_list[1], node_list[2])
# # fig.show()
# # print()

# for it1, it2 in info_.cells.items():
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     node_list = []
#     bound_names = ''
#     # for it3 in it2.lnode:
#     #     node_list.append(list(info_.nodes[it3].loc))
#     # node_list = np.transpose(np.array(node_list))
#     # ax.scatter(node_list[0], node_list[1], node_list[2])
#     for it3, it4 in it2.connect.items():
#         dCf_ = geom_.dCf.vec(it1, it4)
#         line = it2.loc + dCf_
#         neigh_loc = info_.cells[it3].loc
#         ax.plot([it2.loc[0], line[0]], [it2.loc[1], line[1]], [it2.loc[2], line[2]])
#         ax.scatter(neigh_loc[0], neigh_loc[1], neigh_loc[2])
#         ax.scatter(info_.faces[it4].loc[0], info_.faces[it4].loc[1], info_.faces[it4].loc[2])
#     for it3 in it2.bound:
#         dCf_ = geom_.dCf.vec(it1, it3)
#         line = it2.loc + dCf_
#         ax.plot([it2.loc[0], line[0]], [it2.loc[1], line[1]], [it2.loc[2], line[2]])
#         ax.scatter(info_.faces[it3].loc[0], info_.faces[it3].loc[1], info_.faces[it3].loc[2])
#         if len(info_.faces[it3].group) > 0:
#             bound_names += f'{info_.faces[it3].group[0]} ,'
#     for it3, it4 in it2.conj.items():
#         dCf_ = geom_.dCf.vec(it1, it4)
#         line = it2.loc + dCf_
#         neigh_loc = info_.cells[it3].loc
#         ax.plot([it2.loc[0], line[0]], [it2.loc[1], line[1]], [it2.loc[2], line[2]])
#         ax.scatter(neigh_loc[0], neigh_loc[1], neigh_loc[2])
#         ax.scatter(info_.faces[it4].loc[0], info_.faces[it4].loc[1], info_.faces[it4].loc[2])
#     fig.suptitle(bound_names)
#     fig.show()
#     print()

# for it1, it2 in info_.cells.items():
#     list_sf = []
#     sum_sf = np.array([0, 0, 0], dtype = float)
#     sum_dCf = np.array([0, 0, 0], dtype = float)
#     face_list = []
#     area_list = []
#     node_list = [list(it2.loc)]
#     for it3, it4 in it2.connect.items():
#         sum_sf += geom_.Sf.vec(it1, it4)
#         sum_dCf += geom_.dCf.vec(it1, it4)
#         list_sf.append(geom_.Sf.vec(it1, it4))
#         area_list.append(info_.faces[it4].size)
#     for it3 in it2.bound:
#         sum_sf += geom_.Sf.vec(it1, it3)
#         sum_dCf += geom_.dCf.vec(it1, it3)
#         list_sf.append(geom_.Sf.vec(it1, it3))
#         area_list.append(info_.faces[it3].size)
#     for it3, it4 in it2.conj.items():
#         sum_sf += geom_.Sf.vec(it1, it4)
#         sum_dCf += geom_.dCf.vec(it1, it4)
#         list_sf.append(geom_.Sf.vec(it1, it4))
#         area_list.append(info_.faces[it4].size)
#     for it3 in it2.lnode:
#         node_list.append(list(info_.nodes[it3].loc))
#     area_list.sort()
#     node_list = np.transpose(np.array(node_list))
#     print(f'connect: ', it2.connect)
#     print(f'conj: ', it2.conj)
#     print(f'bound: ', it2.bound)
#     print(f'size: ', area_list)
#     # print(list_sf)
#     print(f'{it1}: ', sum_sf, sum_dCf)
    
#     # fig = plt.figure()
#     # ax = fig.add_subplot(projection='3d')
#     # ax.scatter(node_list[0], node_list[1], node_list[2])
#     # fig.show()    
#     print()

# case_l = np.array([list(range(5,10))]).T

# for cases_ in case_l:
#     case_manager(*cases_, 7, 100, 0.9, 1e-4, 5, 60, 12)
#     case_manager(*cases_, 7, 250, 0.9, 1e-4, 5, 60, 12)
#     case_manager(*cases_, 7, 500, 0.9, 1e-4, 5, 60, 12)

