import numpy as np
import parse_gmsh
import scipy.sparse as sparse
import scipy.linalg as linalg
from copy import deepcopy

# ----
# MESH

# elem_obj: struct; loc, type, size, lnode, lcell, lface_neigh, lface_bound
# node: elem_obj; {loc}
# face: elem_obj; {loc, type, size, lnode, lcell}
# cell: elem_obj; {loc, type, size, lnode, lcell, lface_neigh, lface_bound}

class mesh_obj:
    __tag = -1
    __loc = None
    __group = None
    __size = None
    __lnode = None
    __lcell = None
    __lface = None
    __wdist = None

    def __init__(self):
        pass
    @property
    def tag(self):
        return self.__tag
    @tag.setter
    def tag(self, tag_: int):
        self.__tag = tag_
    @property
    def loc(self):
        return self.__loc
    @loc.setter
    def loc(self, loc_: np.ndarray):
        self.__loc = loc_
    @property
    def group(self):
        return self.__group
    @group.setter
    def group(self, group_: str):
        self.__group = group_
    @property
    def size(self):
        return self.__size
    @size.setter
    def size(self, size_: float):
        self.__size = size_
    @property
    def lnode(self):
        return self.__lnode
    @lnode.setter
    def lnode(self, lnode_: list):
        self.__lnode = lnode_
    @property
    def lface(self):
        return self.__lface
    @lface.setter
    def lface(self, lface_: list):
        self.__lface = lface_
    @property
    def lcell(self):
        return self.__lcell
    @lcell.setter
    def lcell(self, lcell_: list):
        self.__lcell = lcell_
    @property
    def wdist(self):
        return self.__wdist
    @wdist.setter
    def wdist(self, wdist_: float):
        self.__wdist = wdist_
class node(mesh_obj):
    def __init__(self, loc_: np.ndarray):
        self.loc(loc_)
class face(mesh_obj):
    def __init__(self, tag_: int, group_, lnode_: list, nodes_: dict):
        self.tag(tag_)
        self.group(group_)
        self.lnode(lnode_)
        if len(self.lnode()) == 3:
            centroid = np.array([0, 0, 0], dtype = float)
            for it1 in self.lnode():
                centroid += nodes_[it1]
            self.loc(centroid / 3)
            v1 = nodes_[self.lnode()[0]] - nodes_[self.lnode()[1]]
            v1 = nodes_[self.lnode()[1]] - nodes_[self.lnode()[2]]
            self.size(np.sqrt(np.sum(np.array([it2**2 for it2 in np.cross(v1, v2)]))) / 2)
        if len(self.lnode()) == 4:
            center = np.array([0, 0, 0], dtype = float)
            for it1 in self.lnode():
                center += nodes_[it1]
            center = center / len(self.lnode())
            diag1 = []
            dist = 0.0
            for it1 in range(0, len(self.lnode())):
                for it2 in range(it2, len(self.lnode())):
                    check = np.sqrt(np.sum(np.array([it3**2 for it3 in nodes_[self.lnode()[it1]] - nodes_[self.lnode()[it2]]])))
                    if check > dist:
                        dist = check; diag1 = [it1, it2]
            diag2 = [it1 for it1 in self.lnode() if it1 not in diag1]
            diag1.insert(0, diag1[0]); diag1.append(diag1[-1])
            diag2 = diag2.extend(diag2)
            tri_sets = list(zip(diag1, diag2))
            tri_vec = []; tri_centroid = []; tri_area = []
            for it1 in tri_sets:
                v1 = center - np.array(nodes_[self.lnode()[it1[0]]])
                v2 = center - np.array(nodes_[self.lnode()[it1[1]]])
                tri_vec.append([v1,v2])
                tmp_centroid = center
                for it2 in it1:
                    tmp_centroid += nodes_[it2]
                tri_centroid.append(tmp_centroid / 3)
            for it1 in tri_vec:
                tri_area.append(np.sqrt(np.sum(np.array([it2**2 for it2 in np.cross(it1[0], it2[1])]))) / 2)
            centroid = np.array([0, 0, 0], dtype = float)
            for it1 in range(0, len(tri_area)):
                centroid += tri_centroid[it1] * tri_area[it1]
            self.size(np.sum(tri_area)); self.loc(centroid / self.size()) 
class cell(mesh_obj):
    def __init__(self, tag_: int, group_, lnode_: list, nodes_: dict, faces_: dict):
        self.tag(tag_)
        self.group(group_)
        self.lnode(lnode_)
        lface_ = []
        for it1, it2 in faces_.items():
            check = np.array([it3 in self.lnode() for it3 in it2.lnode()])
            if all(check) is True:
                lface_.append(it1)
        self.lface(lface_)
        center = np.array([0, 0, 0], dtype = float)
        for it1 in self.lnode():
            center += nodes_[it1]
        center = center / len(self.lnode())
        prism_height = []; prism_vol = []
        for it1 in self.lface():
            v0 = center - faces_[it1].loc(); v0_scale = np.sqrt(np.sum(np.array([it2**2 for it2 in v0])))
            vA1 = faces_[it1].lnode()[0] - faces_[it1].loc()
            vA2 = faces_[it1].lnode()[1] - faces_[it1].loc()
            vS = np.cross(vA1, vA2); vS_scale = np.sqrt(np.sum(np.array([it2**2 for it2 in vS])))
            vS = vS / vS_scale
            if np.dot(v0, vS) < 0:
                vS = vS * -1
            cos = ( np.dot(v0, vS) ) / (v0_scale * vS_scale)
            vH_scale = np.sqrt(np.sum(np.array([it2**2 for it2 in (vS * (v0_scale * cos))])))
            vH = v0 * ((vH_scale / 4) / cos)
            prism_height.append(vH); prism_vol.append(vH_scale * faces_[it1].size() / 3)
        centroid = np.array([0, 0, 0], dtype = float)
        for it1 in range(0, len(prism_height)):
            centroid += prism_height[it1] * prism_vol[it1]
        self.size(np.sum(prism_vol))
        self.loc(centroid / self.size())
    def neigh(self, cells_: dict):
        lcell = []
        for it1, it2 in cells_.items():
            check = np.array([it3 in it2.lface() for it3 in self.lface()]) 
            if any(check) is True:
                face_id = self.lface()[np.where(check == True)[0][0]]
                lcell.append((it1, face_id))
        self.lcell(lcell)
class mesh:
    __nodes = None
    __faces = None
    __cells = None

    @property
    def nodes(self):
        return self.__nodes
    @property
    def faces(self):
        return self.__faces
    @property
    def cells(self):
        return self.__cells

    def __init__(self, fileloc: str):
        gmsh = parse_gmsh.parse(fileloc)
        nodes_ = {}; faces_ = {}; cells_ = {}
        ctd_node = 0; ctd_face = 0; ctd_cell = 0
        for entity_ in gmsh.get_node_entities():
            for node_ in entity_.get_nodes():
                nodes_[ctd_node] = node(node_.get_tag(), node_.get_coordinate())
                ctd_node += 1
        for entity_ in gmsh.get_element_entities():
            group_ = entity_.get_group
            if entity_.get_dimension() == 2:
                for face_ in entity_.get_elements():
                    faces_[ctd_face] = face(face_.get_tag(), group_, [it1-1 for it1 in face_.get_connectivity()], nodes_)
                    ctd_face += 1
            if entity_.get_dimension() == 3:
                for cell_ in entity_.get_elements():
                    cells_[ctd_cell] = cell(cell_.get_tag(), group_, [it1-1 for it1 in cell_.get_connectivity()], nodes_, faces_)
        for it1, it2 in cells_.items():
            it2.neigh(cells_)
        self.nodes(nodes_)
        self.faces(faces_)
        self.cells(cells_)
        a_coef = []
        b_coef = []
        for it1, it2 in self.cells().items():
            aC = 0.0
            a_row = np.zeros(shape=(1, len(list(self.cells.keys()))), dtype = float)[0]
            for it3 in it2.lcell():
                face_ = self.faces()[it3[1]]; neigh_ = self.cells()[it3[0]]
                dCF = neigh_.loc() - it2.loc(); dCF_scale = np.sqrt(np.sum(np.array([it4**2 for it4 in dCF])))
                eCF = dCF / dCF_scale
                vf1 = self.nodes()[face_.lnode()[0]].loc() - face_.loc()
                vf2 = self.nodes()[face_.lnode()[1]].loc() - face_.loc()
                vS = np.cross(vf1, vf2); vS_scale = np.sqrt(np.sum(np.array([it2**2 for it2 in vS])))
                vS = vS / vS_scale
                if np.dot(dCF, vS) < 0:
                    vS = vS * -1
                Sf = vS * face_.size()
                ECf = eCF * (np.dot(Sf, Sf)) / (np.dot(eCF, Sf))
                coef = ECf / dCF
                aC += coef
                a_row[it3[0]] = coef
            a_row[it1] = aC
            a_coef.append(list(a_row))
            b_coef.append([it2.size()])
        A = sparse.lil_matrix(a_coef).tocsr()
        B = sparse.lil_matrix(b_coef).tocsr()
        c, low = linalg.cho_factor(A)
        wdist_ = linalg.cho_solve((c, low), B); wdist_ = np.array(wdist_).ravel()
        for it1, it2 in self.cells().items():
            it2.wdist(wdist_[it1])

# / dCF, dCf, eCF, eCf, Sf, Ef, Tf
# geom: obj; x, y, z -> const sparse matrix

class geom_obj:
    __x = None
    __y = None
    __z = None
    
    def __init__(self, x_: sparse.csr_matrix, y_: sparse.csr_matrix, z_: sparse.csr_matrix):
        self.__x = x_; self.__y = y_; self.__z = z_
    def vec(self, row: int, col: int):
        x_ = self.__x[row, col]; y_ = self.__y[row, col]; z_ = self.__z[row, col]
        return np.array([x_, y_, z_])
    def scalar(self, row: int, col: int):
        return np.sqrt(np.sum([it1**2 for it1 in self.vec(row, col)]))
    def norm(self, row: int, col: int):
        return self.vec(row, col) / self.scalar(row, col)
class geom:
    __dCf = None
    __dCF = None
    __Sf = None
    __Ef = None
    __Tf = None

    @property
    def dCf(self):
        return self.__dCf
    @dCf.setter
    def dCf(self, dCf_: geom_obj):
        self.__dCf = dCf_
    @property
    def dCF(self):
        return self.__dCF
    @dCF.setter
    def dCF(self, dCF_: geom_obj):
        self.__dCF = dCF_
    @property
    def Sf(self):
        return self.__Sf
    @Sf.setter
    def Sf(self, Sf_: geom_obj):
        self.__Sf = Sf_
    @property
    def Ef(self):
        return self.__Ef
    @Ef.setter
    def Ef(self, Ef_: geom_obj):
        self.__Ef = Ef_
    @property
    def Tf(self):
        return self.__Tf
    @Tf.setter
    def Tf(self, Tf_: geom_obj):
        self.__Tf = Tf_

    def __init__(self, msh: mesh):
        lcell_ = len(list(msh.cells().keys()))
        lface_ = len(list(msh.cells().keys()))
        cf = np.zeros(shape=(lcell_, lface_), dtype = float)
        cc = np.zeros(shape=(lcell_, lcell_), dtype = float)
        # dCF
        dCF_x = deepcopy(cc); dCF_y = deepcopy(cc); dCF_z = deepcopy(cc)
        for it1, it2 in msh.cells().items():
            for it3 in it2.lcell():
                dCF_ = msh.cells()[it3[0]].loc() - it2.loc()
                dCF_x[it1][it3] = dCF_[0]; dCF_y[it1][it3] = dCF_[1]; dCF_z[it1][it3] = dCF_[2]
        self.dCF(geom_obj(sparse.lil_matrix(dCF_x).tocsr(), sparse.lil_matrix(dCF_y).tocsr(),
                 sparse.lil_matrix(dCF_z).tocsr()))        
        # dCf, Sf, Ef, Tf
        dCf_x = deepcopy(cf); dCf_y = deepcopy(cf); dCf_z = deepcopy(cf)
        Sf_x = deepcopy(cf); Sf_y = deepcopy(cf); Sf_z = deepcopy(cf)
        Ef_x = deepcopy(cf); Ef_y = deepcopy(cf); Ef_z = deepcopy(cf)
        Tf_x = deepcopy(cf); Tf_y = deepcopy(cf); Tf_z = deepcopy(cf)
        for it1, it2 in msh.cells().items():
            for it3 in it2.lface():
                dCf_ = msh.faces()[it3].loc() - it2.loc()
                v1 = msh.faces()[it3].loc() - msh.nodes()[msh.faces()[it3].lnode()[0]].loc()
                v2 = msh.faces()[it3].loc() - msh.nodes()[msh.faces()[it3].lnode()[1]].loc()
                vS = np.cross(v1, v2); vS_scalar = np.sqrt(np.sum([it4**2 for it4 in vS]))
                if np.dot(vS, dCf_) < 0:
                    vS = -1 * vS
                Sf_ = vS * msh.faces()[it3].size() / vS_scalar
                if all([it3 in list(it2.lcell().values())]) is True:
                    neigh_id = it2.lcell()[list(it2.lcell().keys())[\
                        np.where(np.array(list(it2.lcell().values()))==it3)[0][0]]]
                    eCF_ = self.dCF().norm(it1, neigh_id)
                else:
                    eCF_ = deepcopy(dCf_)
                Ef_ = eCF_ * (np.dot(Sf_, Sf_) / np.dot(eCF_, Sf_))
                Tf_ = Sf_ - Ef_
                dCf_x[it1][it3] = dCf_[0]; dCf_y[it1][it3] = dCf_[1]; dCf_z[it3] = dCf_[2]
                Sf_x[it1][it3] = Sf_[0]; Sf_y[it1][it3] = Sf_[1]; Sf_z[it1][it3] = Sf_[2]
                Ef_x[it1][it3] = Ef_[0]; Ef_y[it1][it3] = Ef_[1]; Ef_z[it1][it3] = Ef_[2]
                Tf_x[it1][it3] = Tf_[0]; Tf_y[it1][it3] = Tf_[1]; Tf_z[it1][it3] = Tf_[2]
        self.dCf(geom_obj(sparse.lil_matrix(dCf_x).tocsr(), sparse.lil_matrix(dCf_y).tocsr(),
                 sparse.lil_matrix(dCf_z).tocsr()))
        self.Sf(geom_obj(sparse.lil_matrix(Sf_x).tocsr(), sparse.lil_matrix(Sf_y).tocsr(),
                 sparse.lil_matrix(Sf_z).tocsr()))
        self.Ef(geom_obj(sparse.lil_matrix(Ef_x).tocsr(), sparse.lil_matrix(Ef_y).tocsr(),
                 sparse.lil_matrix(Ef_z).tocsr()))
        self.Tf(geom_obj(sparse.lil_matrix(Tf_x).tocsr(), sparse.lil_matrix(Tf_y).tocsr(),
                 sparse.lil_matrix(Tf_z).tocsr()))
