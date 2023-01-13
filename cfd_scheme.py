import numpy as np; import pandas as pd
import math; import os
import meshio
from CoolProp.HumidAirProp import HAPropsSI
import scipy.sparse as sparse

class user:
    def __init__(self, *args):
        # args init_value : str, solid_props : str, const_value : str
        dirname = os.getcwd()
        self.inits = pd.read_csv(dirname + "\\problem\\test\\" + args[0])
        self.solid_props = pd.read_csv(dirname + "\\problem\\test\\" + args[1], index_col = 0)
        self.constants = pd.read_csv(dirname + "\\problem\\test\\" + args[2])

class props:
    def __init__(self, *args):
        # args what = np.array([str]), P [Pa]: float, T [K]: float, W [kg w / kg da]: float,
        # constants = dict({"k": ..., "eps": ...})
        for i in args[0]:
            # fluid
            if "rho" == i:
                rho__ = HAPropsSI("Vha", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
                self.rho = np.array([1 / rho__, 1 / rho__])
            elif "miu" == i:
                miu__ = HAPropsSI("mu", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
                self.miu = np.array([miu__, miu__], dtype = float)
            elif "cp" == i:
                cp__ = HAPropsSI("cp_ha", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
                self.cp = np.array([cp__, cp__], dtype = float)
            elif "alpha" == i:
                k__ = HAPropsSI("k", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
                rho__ = HAPropsSI("Vha", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
                cp__ = HAPropsSI("cp_ha", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
                alpha__ = k__ * rho__ / cp__
                self.alpha = np.array([alpha__, alpha__], dtype = float)
            # solid
            elif "k" == i:
                self.k = np.array([args[4]["k"]], dtype = float)
            elif "eps" == i:
                self.eps = np.array([args[4]["eps"]], dtype = float)
        return
    def updateprop(self, *args):
        # fluid only
        # rho, miu, cp, alpha
        # args P [Pa] : float, T [K] : float, W [kg w / kg da] 
        rho__ = 1 / HAPropsSI("Vha", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
        miu__ = HAPropsSI("mu", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
        cp__ = HAPropsSI("cp_ha", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
        k__ = HAPropsSI("k", "P", float(args[1]), "T", float(args[2]), "W", float(args[3]))
        alpha__ = k__ / (rho__ * cp__)
        self.rho[-1] = rho__
        self.miu[-1] = miu__
        self.cp[-1] = cp__
        self.alpha[-1] = alpha__
        return
    def forwardtimeprop(self):
        # fluid
        for i in ["rho", "miu", "cp", "alpha"]:
            self.__dict__[i] = np.array([self.__dict__[i][-1], self.__dict__[i][-1]])
        return

class face:
    def __init__(self):
        pass
    def appendElements(self, *args):
        # args node : np.array([]), centroid : np.array([]), area = float
        self.node = args[0]
        self.centroid = args[1]
        self.area = args[2]
        return
    def appendBound(self, *args):
        # args bound_name = str, user : user, what : str ("fluid", "solid", "conj"), solid_name = str
        self.bound = args[0].split(" ")
        self.value = dict({})
        self.grad = dict({})
        if "fluid" == args[2]:
            for i in ["P", "Pcor", "u", "v", "w", "k", "e", "T"]:
                self.value[i] = np.array([args[1].inits.loc[0, i], args[1].inits.loc[0, i]],
                                dtype = float)
                self.grad[i] = np.array([np.array([0.00, 0.00, 0.00], dtype = float),
                               np.array([0.00, 0.00, 0.00], dtype = float)])
            self.prop = props(["rho", "miu", "cp", "alpha"],
                        args[1].inits.loc[0, "P"], args[1].inits.loc[0, "T"],
                        args[1].constants.loc[0, "Wamb"])
        elif "solid" == args[2]:
            self.value["T"] = np.array([args[2].inits.loc[0, "T"], args[2].inits.loc[0, "T"]],
                              dtype = float)
            self.grad["T"] =  np.array([np.array([0.00, 0.00, 0.00], dtype = float),
                              np.array([0.00, 0.00, 0.00], dtype = float)])
            constant_solid_prop = dict({"k": args[1].solid_prop.loc["k", args[3]], "eps": args[1].solid_prop.loc["eps", args[3]]})
            self.prop = props(np.array(["k", "eps"]), args[2].inits.loc[0, "P"], args[2].inits.loc[0, "T"], 
                          args[2].constants.loc[0, "Wamb"], constant_solid_prop)
        elif "conj" == args[2]:
            for i in ["P", "Pcor", "u", "v", "w", "k", "e", "T"]:
                self.value[i] = np.array([args[1].inits.loc[0, i], args[1].inits.loc[0, i]], 
                                dtype = float)
                self.grad[i] = np.array([np.array([0.00, 0.00, 0.00], dtype = float), 
                               np.array([0.00, 0.00, 0.00], dtype = float)])
            constant_solid_prop = dict({"k": args[1].solid_prop.loc["k", args[3]], "eps": args[1].solid_prop.loc["eps", args[3]]})
            self.prop = props(["rho", "miu", "cp", "alpha", "k", "eps"], 
                        args[1].inits.loc[0, "P"], args[1].inits.loc[0, "T"], 
                        args[1].constants.loc[0, "Wamb"], constant_solid_prop)
        return
        
class cell:
    def __init__(self):
        self.conj_id = -1
    def appendElements(self, *args):
        # args node : np.array([]), member : np.array([]), centroid = np.array([]), volume : float
        self.node = args[0]
        self.member = args[1]
        self.centroid = args[2]
        self.volume = args[3]
        return
    def appendDomain(self, *args):
        # args domain_name = str, user : user
        self.domain = args[0].split(" ")
        self.value = dict({})
        self.grad = dict({})
        if "fluid" in args[0]:
            for i in ["P", "Pcor", "u", "v", "w", "k", "e", "T"]:
                self.value[i] = np.array([args[1].inits.loc[0, i], args[1].inits.loc[0, i]], 
                                dtype = float)
                self.grad[i] = np.array([np.array([0.00, 0.00, 0.00], dtype = float), 
                               np.array([0.00, 0.00, 0.00], dtype = float)])
                self.prop = props(["rho", "miu", "cp", "alpha"], 
                                  args[1].inits.loc[0, "P"], args[1].inits.loc[0, "T"], 
                                  args[1].constants.loc[0, "Wamb"])
        else:
            self.value["T"] = np.array([args[1].inits.loc[0, "T"], args[1].inits.loc[0, "T"]], 
                            dtype = float)
            self.grad["T"] = np.array([np.array([0.00, 0.00, 0.00], dtype = float), 
                             np.array([0.00, 0.00, 0.00], dtype = float)])
            constant_solid_prop = dict({"k": args[1].solid_props.loc["k", args[0]], "eps": args[1].solid_props.loc["eps", args[0]]})
            self.prop = props(np.array(["k", "eps"]), args[1].inits.loc[0, "P"], args[1].inits.loc[0, "T"], 
                          args[1].constants.loc[0, "Wamb"], constant_solid_prop)
        return
        
class clust:
    def __init__(self, *args):
        # args 
        self.value = args[0]
        self.member = args[1]

class geom:
    def __init__(self, *args):
        pass
    def __call__(self, *args):
        # args (what : str, row : int, col : int, val : bool)
        arr = self.__dict__[args[0]].toarray()
        vec = np.array([self.__dict__[args[0]][0][args[1]][args[2]], 
                       self.__dict__[args[0]][1][args[1]][args[2]], 
                       self.__dict__[args[0]][2][args[1]][args[2]]])
        if args[3] is True:
            return np.sqrt(np.sum(np.array([map(lambda x: x^2, vec)]))) 
        else:
            return vec

class template:
    def __init__(self, *args):
        pass
        
class mesh:
    def __init__(self, *args):
        # args mesh_loc : str, user : user
        dirname = os.getcwd()
        spec = meshio.read(dirname + "\\problem\\test\\" + args[0])
        self.getElements(spec)
        # self.getInfo(spec, args[1])
    def getMember(self, *args):
        # args cnode : np.array([], dtype = int)
        member__ = []
        for i, j in self.faces.items():
            check = [x in args[0] for x in j.node]
            if all(check) is True:
                print(args[0], j.node)
                member__.append(i)
        return np.array(member__)
    def getSize(self, *args):
        # args : isface : bool, nodes / faces : np.array([], dtype = int)
        # triangle and quad area-based only for now
        size__ = 0.00
        centroid__ = np.array([0.00, 0.00, 0.00])
        if args[0] is True:
            # face
            if args[1].shape[0] == 3:
                # triangle
                size__ = np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                         np.cross(self.nodes[args[1][1]] - self.nodes[args[1][0]], 
                         self.nodes[args[1][2]] - self.nodes[args[1][0]])))])))
                centroid__ = (self.nodes[args[1][0]] + self.nodes[args[1][1]] + 
                             self.nodes[args[1][2]]) / 3
            elif args[1].shape[0] == 4:
                # quads
                # triagulate
                fcentre__ = np.array([0.00, 0.00, 0.00], dtype = float)
                for i in args[1]:
                    fcentre__ += self.nodes[i]
                fcentre__ = fcentre__ / args[1].shape[0]
                rhs__ = 0.00
                diag1__ = np.array([0, 0])
                for i in range(1, 4):
                    if np.sqrt(np.sum(np.array([list(map(lambda x: x**2, 
                       self.nodes[args[1][i]] - self.nodes[args[1][0]]))]))) > rhs__:
                        diag1__[-1] = i
                diag2__ = np.delete(args[1], diag1__)
                size__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                          np.cross(self.nodes[diag2__[0]] - fcentre__, 
                          self.nodes[diag1__[0]] - fcentre__)))])))
                centroid__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                              np.cross(self.nodes[diag2__[0]] - fcentre__, 
                              self.nodes[diag1__[0]] - fcentre__)))]))) * \
                              (self.nodes[diag2__[0]] + self.nodes[diag1__[0]] +
                              fcentre__) / 3
                size__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                          np.cross(self.nodes[diag2__[1]] - fcentre__, 
                          self.nodes[diag1__[0]] - fcentre__)))])))
                centroid__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                              np.cross(self.nodes[diag2__[1]] - fcentre__, 
                              self.nodes[diag1__[0]] - fcentre__)))]))) * \
                              (self.nodes[diag2__[1]] + self.nodes[diag1__[0]] +
                              fcentre__) / 3    
                size__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                          np.cross(self.nodes[diag2__[0]] - fcentre__, 
                          self.nodes[diag1__[1]] - fcentre__)))])))
                centroid__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                              np.cross(self.nodes[diag2__[0]] - fcentre__, 
                              self.nodes[diag1__[1]] - fcentre__)))]))) * \
                              (self.nodes[diag2__[0]] + self.nodes[diag1__[1]] +
                              fcentre__) / 3               
                size__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                          np.cross(self.nodes[diag2__[1]] - fcentre__, 
                          self.nodes[diag1__[1]] - fcentre__)))])))
                centroid__ += np.sqrt(np.sum(np.array([list(map(lambda x : x**2, 
                              np.cross(self.nodes[diag2__[1]] - fcentre__, 
                              self.nodes[diag1__[1]] - fcentre__)))]))) * \
                              (self.nodes[diag2__[1]] + self.nodes[diag1__[1]] + 
                              fcentre__) / 3
                centroid__ = centroid__ / size__ 
            else:
                pass
        else:
            # cell
            # correct ccentre to Sf
            ccentre__ = np.array([0.00, 0.00, 0.00], dtype = float)
            for i in args[1]:
                for j in self.faces[i].node:
                    ccentre__ += self.nodes[j]
            ccentre__ = ccentre__ / args[1].shape[0]
            for i in args[1]:
                dCF__ = ccentre__ - self.faces[i].centroid
                dCF_val__ = np.sqrt(np.sum(np.array(list(map(lambda x: x**2, dCF__)))))
                vec1__ = self.nodes[self.faces[i].node[1]] - self.nodes[self.faces[i].node[0]]
                vec2__ = self.nodes[self.faces[i].node[2]] - self.nodes[self.faces[i].node[0]]
                Sf__ = np.cross(vec1__, vec2__)
                Sf_val__ = np.sqrt(np.sum(np.array(list(map(lambda x: x**2, Sf__)))))
                Sf__ = Sf__ / Sf_val__
                if np.dot(dCF__, Sf__) >= 0.00:
                    # cos
                    cos__ = np.dot(Sf__, dCF__) / dCF_val__
                    volume__ = cos__ * dCF_val__ * self.faces[i].area / 3
                    size__ += volume__
                    centroid__ += volume__ * (self.faces[i].centroid + dCF__ / 4)
                else:
                    cos__ = np.dot(-1 * Sf__, dCF__) / dCF_val__
                    volume__ = cos__ * dCF_val__ * self.faces[i].area / 3
                    size__ += volume__
                    centroid__ += volume__ * (self.faces[i].centroid + dCF__ / 4)
                centroid__ = centroid__ / size__
        return size__, centroid__    
    def getElements(self, spec):
        # args : spec : meshio.mesh
        self.nodes = dict({})
        for i in range(0, spec.points.shape[0]):
            self.nodes[i] = spec.points[i]
        for i, j in self.nodes.items():
            print("{}: {}".format(i, j))
        self.faces = dict({}); face_id__ = 0; 
        for i in spec.cells_dict.keys():
            if spec.cells_dict[i][0].shape[0] < 5:
                # face
                for j in range(0, spec.cells_dict[i].shape[0]):
                    fnode__ = spec.cells_dict[i][j]
                    farea__, fcentroid__ = self.getSize(True, fnode__)
                    face__ = face()
                    face__.appendElements(fnode__, fcentroid__, farea__)
                    self.faces[j + face_id__] = face__
                face_id__ += spec.cells_dict[i].shape[0]
            else:
                continue
        for i, j in self.faces.items():
            print("{}: nodes: {}, area: {}, centroid: {}".format(i, j.node, j.area, j.centroid))
        self.cells = dict({}); cell_id__ = 0
        for i in spec.cells_dict.keys():
            if spec.cells_dict[i][0].shape[0] >= 5:            
                # cell
                for j in range(0, spec.cells_dict[i].shape[0]):
                    cnode__ = spec.cells_dict[i][j]
                    cface__ = self.getMember(cnode__)
                    cvolume__, ccentroid__ = self.getSize(False, cface__)
                    cell__ = cell()
                    cell__.appendElements(cnode__, cface__, ccentroid__, cvolume__)
                    self.cells[j + cell_id__] = cell__
                cell_id__ += spec.cells_dict[i].shape[0]
            else:
                continue
        for i, j in self.cells.items():
            print("{}: nodes: {}, faces: {}, volume: {}, centroid: {}".format(i, j.node, j.member, j.volume, j.centroid))
        return
    def getTemplate(self):
        isbound = dict({})
        for i in range(0, len(list(self.cells.keys()))):
            isbound[i] = self.cells[i].member
        row_neigh__ = dict({}); col_neigh__ = dict({}); data_neigh__ = dict({})
        row_bound__ = dict({}); col_bound__ = dict({}); data_bound__ = dict({})
        for i in range(0, len(list(self.cells.keys()))-1):
            domain1__ = [k for k in ["fluid", "solid"] if any(list(map(lambda x: k in x, 
                          self.cells[i].domain))) is True][0]
            for j in range(i, len(list(self.cells.keys()))):
                domain2__ = [k for k in ["fluid", "solid"] if any(list(map(lambda x: k in x, 
                            self.cells[j].domain))) is True][0]
                check = [x in self.cells[j].member for x in self.cells[i].member]
                if any(check) is True:
                    neighbor__ = self.cells[i].member[check.index(True)]
                    if len(np.array([domain1__, domain2__])) != \
                       len(np.unique(np.array([domain1__, domain2__]))):
                        if domain1__ not in list(row_neigh__.keys()):
                            row_neigh__[domain1__] = [i]; row_neigh__[domain1__].append(j)
                            col_neigh__[domain1__] = [j]; col_neigh__[domain1__].append(i)
                            data_neigh__[domain1__] = [neighbor__]; data_neigh__[domain1__].append(neighbor__)
                            row_bound__[domain1__] = []; col_bound__[domain1__] = []; data_bound__[domain1__] =  []
                        else:
                            row_neigh__[domain1__].append(i); row_neigh__[domain1__].append(j)
                            col_neigh__[domain1__].append(j); col_neigh__[domain1__].append(i)
                            data_neigh__[domain1__].append(neighbor__); data_neigh__[domain1__].append(neighbor__)
                        isbound[i] = np.delete(isbound[i], isbound[i] == neighbor__)
                        isbound[j] = np.delete(isbound[j], isbound[j] == neighbor__)
                    else:
                        if "conj" not in list(row_neigh__.keys()):
                            row_neigh__["conj"] = [i]; row_neigh__["conj"].append(j)
                            col_neigh__["conj"] = [j]; col_neigh__["conj"].append(i)
                            data_neigh__["conj"] = [neighbor__]; data_neigh__["conj"].append(neighbor__)
                            self.cells[i].conj_id = neighbor__; self.cells[j].conj_id = neighbor__
                        else:
                            row_neigh__["conj"].append(i); row_neigh__["conj"].append(j)
                            col_neigh__["conj"].append(j); col_neigh__["conj"].append(i)
                            data_neigh__["conj"].append(neighbor__); data_neigh__["conj"].append(neighbor__)
                            self.cells[i].conj_id = neighbor__; self.cells[j].conj_id = neighbor__
            if len(isbound[i]) != 0:
                for k in isbound[i]:
                    row_bound__[domain1__].extend(list(np.full((len(isbound[i])), i, dtype = int)))
                    col_bound__[domain1__].extend(isbound[i])
                    data_bound__[domain1__].extend(list(np.full((len(isbound[i])), 1, dtype = int)))
        row_neigh__["all"] = []; col_neigh__["all"] = []; data_neigh__["all"] = []
        for i in row_neigh__.keys():
            if i != "all":
                row_neigh__["all"].extend(row_neigh__[i])
                col_neigh__["all"].extend(col_neigh__[i])
                data_neigh__["all"].extend(data_neigh__[i])
        template__ = template(); template__.neighbor = dict({}); template__.boundary = dict({})
        for i in row_neigh__.keys():
            template__.neighbor[i] = sparse.csr_matrix((data_neigh__[i], (row_neigh__[i], col_neigh__[i])), 
                                     (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = int)
        for i in row_bound__.keys():
            template__.boundary[i] = sparse.csr_matrix((data_bound__[i], (row_bound__[i], col_bound__[i])), 
                                     (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = int)
        self.templates = template__
        return
    def getGeom(self):
        self.geoms = geom()
        Sf_input__ = [[], [], []]
        Ef_input__ = [[], [], []]
        Tf_input__ = [[], [], []]
        dCf_input__ = [[], [], []]
        eCf_input__ = [[], [], []]
        dCF_input__ = [[], [], []]
        eCF_input__ = [[], [], []]
        neigh__ = self.templates.neighbor["all"].tocoo()
        for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data): 
            dCF__ = self.cells[j].centroid - self.cells[i].centroid
            dCf__ = self.faces[k].centroid - self.cells[i].centroid
            eCF__ = dCF__ / np.sqrt(np.sum(list(map(lambda x: x**2, dCF__))))
            eCf__ = dCf__ / np.sqrt(np.sum(list(map(lambda x: x**2, dCf__))))
            Sf__ = np.dot(self.nodes[self.faces[k].node[1]] - self.nodes[self.faces[k].node[0]], 
                   self.nodes[self.faces[k].node[2]] - self.nodes[self.faces[k].node[0]])
            if np.dot(Sf__, eCf__) > 0.00:
                Sf__ = Sf__ * self.faces[k].area / np.sqrt(np.sum(list(map(lambda x: x**2, Sf__))))
            else:
                Sf__ = -1 * Sf__ * self.faces[k].area / np.sqrt(np.sum(list(map(lambda x: x**2, Sf__))))
            Ef__ = eCF__ * np.dot(Sf__, Sf__) / np.dot(eCF__, Sf__)
            Tf__ = Sf__ - Ef__
            # input
            Sf_input__[0] = [i, k, Sf__[0]]; Sf_input__[1] = [i, k, Sf__[1]]; Sf_input__[2] = [i, k, Sf__[2]]
            Ef_input__[0] = [i, k, Ef__[0]]; Ef_input__[1] = [i, k, Ef__[1]]; Ef_input__[2] = [i, k, Ef__[2]]
            Tf_input__[0] = [i, k, Tf__[0]]; Tf_input__[1] = [i, k, Tf__[1]]; Tf_input__[2] = [i, k, Tf__[2]]
            dCf_input__[0] = [i, k, dCf__[0]]; dCf_input__[1] = [i, k, dCf__[1]]; dCf_input__[2] = [i, k, dCf__[2]]
            eCf_input__[0] = [i, k, eCf__[0]]; eCf_input__[1] = [i, k, eCf__[1]]; eCf_input__[2] = [i, k, eCf__[2]]
            dCF_input__[0] = [i, j, dCF__[0]]; dCF_input__[1] = [i, j, dCF__[1]]; dCF_input__[2] = [i, j, dCF__[2]]
            eCF_input__[0] = [i, j, eCF__[0]]; eCF_input__[1] = [i, j, eCF__[1]]; eCF_input__[2] = [i, j, eCF__[2]]
        for i in [0, 1, 2]:
            Sf_input__[i] = list(np.transpose(np.array(Sf_input__[i])))
            Ef_input__[i] = list(np.transpose(np.array(Ef_input__[i])))
            Tf_input__[i] = list(np.transpose(np.array(Tf_input__[i])))
            dCf_input__[i] = list(np.transpose(np.array(dCf_input__[i])))
            eCf_input__[i] = list(np.transpose(np.array(eCf_input__[i])))
            dCF_input__[i] = list(np.transpose(np.array(dCF_input__[i])))
            eCF_input__[i] = list(np.transpose(np.array(eCF_input__[i])))
        self.geoms.Sf = [sparse.csr_matrix((Sf_input__[0][2], (Sf_input__[0][0], Sf_input__[0][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                         sparse.csr_matrix((Sf_input__[1][2], (Sf_input__[1][0], Sf_input__[1][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                         sparse.csr_matrix((Sf_input__[2][2], (Sf_input__[2][0], Sf_input__[2][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float)]
        self.geoms.Ef = [sparse.csr_matrix((Ef_input__[0][2], (Ef_input__[0][0], Ef_input__[0][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                         sparse.csr_matrix((Ef_input__[1][2], (Ef_input__[1][0], Ef_input__[1][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                         sparse.csr_matrix((Ef_input__[2][2], (Ef_input__[2][0], Ef_input__[2][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float)]
        self.geoms.Tf = [sparse.csr_matrix((Tf_input__[0][2], (Tf_input__[0][0], Tf_input__[0][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                         sparse.csr_matrix((Tf_input__[1][2], (Tf_input__[1][0], Tf_input__[1][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                         sparse.csr_matrix((Tf_input__[2][2], (Tf_input__[2][0], Tf_input__[2][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float)]
        self.geoms.dCf = [sparse.csr_matrix((dCf_input__[0][2], (dCf_input__[0][0], dCf_input__[0][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                          sparse.csr_matrix((dCf_input__[1][2], (dCf_input__[1][0], dCf_input__[1][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                          sparse.csr_matrix((dCf_input__[2][2], (dCf_input__[2][0], dCf_input__[2][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float)]
        self.geoms.eCf = [sparse.csr_matrix((eCf_input__[0][2], (eCf_input__[0][0], eCf_input__[0][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                          sparse.csr_matrix((eCf_input__[1][2], (eCf_input__[1][0], eCf_input__[1][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float),
                          sparse.csr_matrix((eCf_input__[2][2], (eCf_input__[2][0], eCf_input__[2][1])), (len(list(self.cells.keys())), len(list(self.faces.keys()))), dtype = float)]
        self.geoms.dCF = [sparse.csr_matrix((dCF_input__[0][2], (dCF_input__[0][0], dCF_input__[0][1])), (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = float),
                          sparse.csr_matrix((dCF_input__[1][2], (dCF_input__[1][0], dCF_input__[1][1])), (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = float),
                          sparse.csr_matrix((dCF_input__[2][2], (dCF_input__[2][0], dCF_input__[2][1])), (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = float)]
        self.geoms.eCF = [sparse.csr_matrix((eCF_input__[0][2], (eCF_input__[0][0], eCF_input__[0][1])), (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = float),
                          sparse.csr_matrix((eCF_input__[1][2], (eCF_input__[1][0], eCF_input__[1][1])), (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = float),
                          sparse.csr_matrix((eCF_input__[2][2], (eCF_input__[2][0], eCF_input__[2][1])), (len(list(self.cells.keys())), len(list(self.cells.keys()))), dtype = float)]
        return
    def calcView(self, *args):
        # args clust1_id : clust, clust2_id : clust
        area_clust__ = np.sum(np.array([self.faces[i].area for i in self.clusts[args[0]].member]))
        r__ = self.faces[args[1]].centroid - self.faces[args[0]].centroid
        r_val__ = np.sqrt(np.sum(list(map(lambda x: x**2, r__))))
        view__ = 0.00
        for i in self.clusts[args[0]].member:
            Sf_C__ = np.cross(self.nodes[self.faces[i].nodes[0]], self.nodes[self.faces[i].nodes[0]])
            if np.dot(Sf_C__, r__) < 0.00:
                Sf_C__ = -1 * Sf_C__
            Sf_C__ = Sf_C__ / np.sqrt(np.sum(list(map(lambda x: x**2, Sf_C__))))
            cos1__ = np.dot(Sf_C__, r__) / (r_val__)
            for j in self.clusts[args[1]].member:
                Sf_f__ = np.cross(self.nodes[self.faces[j].nodes[0]], self.nodes[self.faces[j].nodes[1]])
                if np.dot(Sf_f__, r__) < 0.00:
                    Sf_f__ = -1 * Sf_f__
                Sf_f__ = Sf_f__ / np.sqrt(np.sum(list(map(lambda x: x**2, Sf_f__))))
                cos2__ = np.dot(Sf_f__, r__) / (r_val__)
                view__ += cos1__ * cos2__ / (math.pi * r_val__**2)
        view__ = view__ / area_clust__
        return view__
    def getClust(self, *args):
        self.clusts = dict({})
        s2s_dict_keys__ = [i for i in args[0].keys() if "s2s" in i]
        for i in s2s_dict_keys__:
            index = i.split(" "); index = [int(i[-1]) for i in index if "s2s" in i][0]
            self.clusts[index] = clust( dict({"q": np.array([0.00, 0.00])}), args[0][i])
        self.geoms.view = np.zeros(shape=(len(list(self.clusts.keys())), len(list(self.clusts.keys()))), dtype = float)
        for i in range(0, self.geoms.view.shape[0]):
            col__ = np.array(range(0, self.geoms.view.shape[0]))
            col__ = np.delete(col__, col__ == i)
            self.geoms.view[i] = [self.calcView(i, j) for j in col__]
        return
    def getInfo(self, spec, __user):
        # args spec : meshio.spec, user__ : user
        # spec.cells_dict based into face_id, cell_id based
        cell_sets_keys = list(spec.cell_sets_dict.keys())
        cell_sets_keys = np.delete(cell_sets_keys, -1)
        face_bound_dict = dict({}); face_id__ = 0
        for i in spec.cells_dict.keys():
            # face boundary
            if spec.cells_dict[i][0].shape[0] < 5:
                for j in cell_sets_keys:
                    if i in list(spec.cell_sets_dict[j].keys()):
                        if i not in list(face_bound_dict.keys()):
                            face_bound_dict[j] = np.array(list(map(lambda x: x + face_id__, 
                                                 spec.cell_sets_dict[j][i])))
                        else:
                            face_bound_dict[j] = np.append(face_bound_dict[j], 
                                                 np.array(list(map(lambda x: x + face_id__, 
                                                 spec.cell_sets_dict[j][i]))))
                face_id__ += spec.cells_dict[i].shape[0]
            else:
                continue   
        cell_domain_dict = dict({}); cell_id__ = 0
        for i in spec.cells_dict.keys():
            # cell domain
            if spec.cells_dict[i][0].shape[0] >= 5:
                for j in cell_sets_keys:
                    if i in list(spec.cell_sets_dict[j].keys()):
                        if i not in list(cell_domain_dict.keys()):
                            cell_domain_dict[j] = np.array(list(map(lambda x: x + cell_id__, 
                                                  spec.cell_sets_dict[j][i])))
                        else:
                            cell_domain_dict[j] = np.append(cell_domain_dict[j], 
                                                  np.array(list(map(lambda x: x + cell_id__, 
                                                  spec.cell_sets_dict[j][i]))))
                cell_id__ += spec.cells_dict[i].shape[0]
            else:
                continue
        # cell_domain_dict appendDomain
        # args bound = str, user : user
        for i in cell_domain_dict.keys():
            for j in self.cells.keys():
                if j in cell_domain_dict[i]:
                    self.cells[j].appendDomain(i, __user)
        # make template and geom based on cell domain
        self.getTemplate(); self.getGeom()
        # face_domain_dict appendBound
        # args bound_name = str, user : user, what : str ("fluid", "solid", "conj"), solid_name = str
        # list and delete if already written
        if "fluid" in list(self.templates.neighbor.keys()):
            neigh__ = self.templates.neighbor["fluid"].tocoo()
            for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
                self.faces[k].appendBound("", __user, "fluid")
        elif "solid" in list(self.templates.neighbor.keys()):
            neigh__ = self.templates.neighbor["solid"].tocoo()
            for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data):
                self.faces[k].appendBound("", __user, "solid", self.cells[i].domain)
        elif "conj" in list(self.templates.neighbor.keys()):
            neigh__ = self.templates.neighbor["conj"].tocoo()
            for i, j, k in zip(neigh__.row, neigh__.col, neigh__.data): 
                if "solid" in self.cells[i].domain:
                    issolid = i
                else:
                    issolid = j
                for l in face_bound_dict.keys():
                    if k in face_bound_dict[l]:
                        self.faces[k].appendBound(l, __user, "conj", self.cells[issolid].domain)
                        break
        if "fluid" in list(self.templates.boundary.keys()):
            neigh__ = self.templates.boundary["fluid"].tocoo()
            for i, j in zip(neigh__.row, neigh__.col, neigh__.data):
                for k in face_bound_dict.keys():
                    if j in face_bound_dict[k]:
                        self.faces[j].appendBound(k, __user, "fluid")
                        break
        elif "solid" in list(self.template.boundary.keys()):
            neigh__ = self.templates.boundary["solid"].tocoo()
            for i, j in zip(neigh__.row, neigh__.col, neigh__.data):
                for k in face_bound_dict.keys():
                    if j in face_bound_dict[k]:
                        self.faces[j].appendBound(k, __user, "solid", self.cells[i].domain)
                        break
        # make clust if s2s in face_bound_dict.keys()
        if any(list(map(lambda x: "s2s" in x, list(face_bound_dict.keys())))) is True:
            self.getClust(face_bound_dict)     
        return

def make_scheme(*args):
    # args init_file : str, solid_prop_file : str, const_value_file : str, spec_file : str
    user__ = user(args[0], args[1], args[2])
    mesh__ = mesh(args[3], user__)
    return user__, mesh__

if __name__ == "__main__":
    filename = ["init_values.csv", "solid_props_values.csv", "constant_values.csv", "sc.msh"]
    user_test, mesh_test = make_scheme(*filename)