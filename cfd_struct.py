import numpy as np
from numpy import linalg
import pandas as pd
import scipy.sparse
import itertools
import math
import gmsh

class props:
    def __init__(self, spec):
        self.__rho = np.array([])
        self.__mu = np.array([])
        self.__cp = np.array([])
        self.__k = np.array([])
        self.__alpha = np.array([])
        self.__eps = np.array([])
    def __call__(self, *args):
        return self.__dict__[args[0]]
    def __updateprop(self, *args):
        self.__rho[-1] = 1 / HAPropsSI("Vha", "P", args[0], \
                         "T", args[1], "W", args[2])
        self.__miu[-1] = HAPropsSI("mu", "P", args[0], \
                         "T", args[1], "W", args[2])
        self.__cp[-1] = HAPropsSI("cp_ha", "P", args[0], \
                        "T", args[1], "W", args[2])
        self.__k[-1] = HAPropsSI("k", "P", args[0], \
                       "T", args[1], "W", args[2])
   
class face:
    def __init__(self, *args):
        # args
        self.__value = dict{}
        self.__grad = dict{}
        self.__centroid = np.array([0, 0, 0])
        self.__area = 0
        self.__bound = np.array([])
        self.__prop = props()
    def __call__(self, *args):
        return self.__dict__[args[0]]
     
class cell:
    def __init__(self, spec):
        self.__value = dict{"u": np.array([])}
        self.__grad = dict{"u": np.array([])}
        self.__centroid = np.array([])
        self.__volume = np.array([])
        self.__domain = np.array([])
        self.__prop = props()
        self.__iswall = -1
    def __call__(self, *args):
        return self.__dict__[args[0]]

class clust:
    def __init__(self, spec):
        self.__value = dict{"q" : np.array([])}
        self.__member = np.array([])
    def __call__(self, *args):
        return self.__dict__[args[0]]

class geom:
    def __init__(self, spec):
        # fc
        self.__Sf = np.array([])
        self.__Ef = np.array([])
        self.__Tf = np.array([])
        self.__dCf = np.array([])
        self.__eCf = np.array([])
        # cc
        self.__dCF = np.array([])
        self.__eCF = np.array([])
        # s2s
        self.__view = np.array([[]])
    def __call__(self, *args):
        # args (what : str, row : int, col : int, val : bool)
        if args[3] is True:
            return np.sqrt(np.sum(np.array( \
                          [map(lambda x: x^2, \
                          self.__dict__[args[0]][args[1]][args])]))) 
        else:
            return self.__dict__[args[0]][args[1]][args[2]]    

class template:
    def __init__(self, spec):
        self.__neigh = dict{"fluid": scipy.sparse.csc_matrix(),
                            "solid": scipy.sparse.csc_matrix(),
                            "conj" : scipy.sparse.csc_matrix(),
                            "all" : scipy.sparse.csc_matrix(),
                            "s2s": scipy.sparse.csc_matrix()}
        self.__bound = dict{"fluid": scipy.sparse.csc_matrix(),
                            "solid": scipy.sparse.csc_matrix(),
                            "conj" : scipy.sparse.csc_matrix(),
                            "all" : scipy.sparse.csc_matrix(),
                            "s2s": scipy.sparse.csc_matrix()}
    def __call__(self, *args):
        return self.__dict__[args[0]]

class mesh:
    def __init__(self, spec):
        self.__nodes = dict{}
        self.__cells = dict{}
        self.__faces = dict{}
        self.__clusts = dict{}
        self.__geoms = geom(spec)
        self.__templates = template()
    def __call__(self, *args):
        return self.__dict__[args[0]]
    def __makemesh(self, *args):
        spec = gmsh.load_mesh(args[0])
        
        return
    
    
class user:
    def __init__(self, *args):
        self.__init_val = pd.read_csv(args[0])
        self.__solid_prop = pd.read_csv(args[1])
        self.__source = pd.read_csv(args[2])
    def __call__(self, *args):
        # args var name : str
        return self.__dict__[args[0]]

class linear:
    def __init__(self, mesh_):
        self.__current_time = 0
        self.__end_iter_res = np.array([]) # max
        self.__lhs = scipy.sparse.csc_matrix()
        self.__rhs = np.array([[]])
    def __call__(self, *args):
        return self.__dict__[args[0]]
    def _linearitr(*args):
        # args, mesh : mesh, what : [variable name, dict key],
        # cell id 1 : int, cell id 2 : int, face id : int
        gC = args[0]("geoms")("dCf", args[3], args[4], True) / \
             (args[0]("geoms")("dCf", args[3], args[4], True) + \
             args[0]("geoms")("dCf", args[2], args[4], True))
        return gC * args[0]("cells")[args[2]](args[1][0])[args[1][1]][-1] + \
               (1 - gC) * args[0]("cells")[args[3]](args[1][0])[args[1][1]][-1]
    def _QUICKgrad(self, *args):
        # args mesh : mesh, what : dict key, cell id 1 : int, cell id 2 : int, face id : int
        grad__ = self._linearitr(args[0], ["grad", args[1]], args[2], args[3], args[4])
        dCF__ = args[0]("geoms")("dCF", args[2], args[3], True)
        eCF__ = args[0]("geoms")("eCF", args[2], args[3], False)
        return grad__ + ((args[0]("cells")[args[3]]("value")[args[1]][-1] + \
               args[0]("cells")[args[2]]("value")[args[1]][-1]) / dCF__) - \
               (np.dot(grad__, eCF__)) * eCF__
    def _QUICKvalue(*args):
        # args mesh : mesh, what : dict key, cell id : int, face id : int
        return args[0]("cells")[args[2]]("value")[args[1]][-1] + 0.5 * np.dot(
               (args[0]("cells")[args[2]]("grad")[args[1]][-1] +
               args[0]("faces")[args[3]]("grad")[args[1]][-1]),
               args[0]("geoms")("dCf", args[2], args[3], False))
    def _calctransient(self, *args):
        # void
        # args mesh : mesh, what : str, time_step : int/double
        for i, j, k in itertools.izip(self("lhs").row, self("lhs").col, self("lhs").data):
            self("lhs")[i][j] = k + args[0]("cells")[i]("prop")("rho")[-1] * args[0]("cells")[i]("volume") / \
                (2 * args[1])
        for i in range(0, len(self("rhs"))):
            self("rhs")[i][0] = self("rhs")[i][0] - args[0]("cells")[i]("prop")("rho")[-2] * \
                                args[0]("cells")[i]("volume") * args[0]("cells")[i]("value")[args[1]][-2] / \
                                args[2]
        return
    def _singlesolve(self, *args):
        # void GMRES
        # args mesh : mesh, what : str, under_relax : double, tol : double, max_iter : int
        A = lambda x: scipy.sparse.linalg.spsolve(self("lhs"), x)
        A = A * args[2]
        b = np.transpose(self("rhs") * args[2])
        x, exitCode = scipy.sparse.linalg.gmres(A, b, tol = args[3], maxiter = args[4])
        print("status: %i", exitCode)
        # return 2d np.array
        return x
    
class pcorrect(linear):
    def __init__(self, *args):
        # args mesh : mesh
        super().__init__(args)
    def __call__(self, *args):
        super().__call__(args)
    def __calccoef(self, *args):
        # args mesh : mesh, u_ref : momentum, v_ref : momentum, w_ref : momentum
        # what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["fluid"].row, \
                       args[0]("templates")("neigh")["fluid"].col, args[0]("templates")("neigh")["fluid"].data):
            Df_x__ = ((args[0]("cells")[i]("volume") / args[1]("lhs")[i][j]) + \
                     (args[0]("cells")[j]("volume") / args[1]("lhs")[i][j])) / 2
            Df_y__ = ((args[0]("cells")[i]("volume") / args[2]("lhs")[i][j]) + \
                     (args[0]("cells")[j]("volume") / args[2]("lhs")[i][j])) / 2
            Df_z__ = ((args[0]("cells")[i]("volume") / args[3]("lhs")[i][j]) + \
                     (args[0]("cells")[j]("volume") / args[3]("lhs")[i][j])) / 2
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            dCF__ = args[0]("geoms")("dCF", i, j, False)
            Dau_f__ = (pow(Df_x__ * Sf__[0], 2) + pow(Df_y__ * Sf__[1], 2) +
                      pow(Df_z__ * Sf__[2], 2)) / (dCF__[0] * Df_x__ * Sf__[0] +
                      dCF__[1] * Df_y__ * Sf__[1] + dCF__[2] * Df_z__ * Sf__[2])
            if prev_row == i:
                aC__ += args[0]("faces")[k]("prop")("rho")[-1] * Dau_f__
                bC__ += args[0]("faces")[k]("prop")("rho")[-1] * \
                        np.dot(np.array([args[1]("faces")[k]("value")["u"][-1],
                        args[1]("faces")[k]("value")["v"][-1],
                        args[1]("faces")[k]("value")["w"][-1]]), Sf__) - \
                        np.dot(np.array([[Df_x__, 0, 0], [0, Df_y__, 0], [0, 0, Df_z__]]) * \
                        args[0]("faces")[k]("grad")["P"] - super()._linearitr(args[0], ["grad", "P"], i, j, k), \
                        Sf__)
                self("lhs")[i][j] = -args[0]("faces")[k]("prop")("rho")[-1] * Dau_f__
                prev_row = i
            else:
                self("lhs")[prev_row][prev_row] = aC__
                self("rhs")[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += args[0]("faces")[k]("prop")("rho")[-1] * Dau_f__
                bC__ += args[0]("faces")[k]("prop")("rho")[-1] * \
                        np.dot(np.array([args[1]("faces")[k]("value")["u"][-1],
                        args[1]("faces")[k]("value")["v"][-1],
                        args[1]("faces")[k]("value")["w"][-1]]), Sf__) - \
                        np.dot(np.array([[Df_x__, 0, 0], [0, Df_y__, 0], [0, 0, Df_z__]]) * \
                        args[0]("faces")[k]("grad")["P"] - super()._linearitr(args[0], ["grad", "P"], i, j, k), \
                        Sf__)
                self("lhs")[i][j] = -args[0]("faces")[k]("prop")("rho")[-1] * Dau_f__
                prev_row = i
        self("lhs")[prev_row][prev_row] = aC__
        self("rhs")[prev_row][0] = bC__
        for i, j, k in itertools.izip(args[0]("template")("boundary")["fluid"].row, \
                       args[0]("template")("boundary")["fluid"].col, args[0]("template")("boundary")["fluid"].data):
            for l in args[0]("faces")[j]("bound"):
                self.__calcbound(args[0], l, i, j)
        return
    def __calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int
        if "noslip" in args[1]:
            args[0]("faces")[args[3]]("value")["P"][-1] = args[0]("cells")[args[2]]("value")["P"][-1] - \
                                                          np.dot(args[0]("cells")[args[2]]("grad")["P"][-1], \
                                                          args[0]("geoms")("Sf", args[2], args[3], False)) - \
                                                          np.dot(args[0]("faces")[args[3]]("grad")["P"][-1], \
                                                          args[0]("geoms")("Tf", args[2], args[3], False)) / \
                                                          (self("lhs")[args[2]][args[2]] / \
                                                          args[0]("cells")[args[2]]("prop")("rho")[-1])
        elif "inlet" in args[1]:
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("rho")[-1] * \
                                             self("lhs")[args[2]][args[2]] / \
                                             args[0]("cells")[args[2]]("prop")("rho")[-1]
        elif "outlet" in args[1]:
            self("lhs")[args[2]][0] += args[0]("faces")[args[3]]("prop")("rho")[-1] * \
                                       self("lhs")[args[2]][args[2]] / \
                                       args[0]("cells")[args[2]]("prop")("rho")[-1]
        else:
            pass
        return
    def __calccorrect(self, *args):
        # void
        # args mesh : mesh, u_ref : momentum, v_ref : momentum, w_ref : momentum
        for i in args[0]("cells").keys():
            pcor_C__ = -args[0]("cells")[i]("prop")("rho")[-1] * args[0]("cells")[i]("volume") * \
                        args[0]("cells")[i]("grad")["Pcor"][-1] / self("lhs")[i][i] 
            args[0]("cells")[i]("value")["u"][-1] += pcor_C__[0]
            args[0]("cells")[i]("value")["v"][-1] += pcor_C__[1]
            args[0]("cells")[i]("value")["w"][-1] += pcor_C__[2]
            args[0]("cells")[i]("value")["P"][-1] += args[0]("cells")[i]("value")["Pcor"][-1]
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["fluid"].row, \
                       args[0]("templates")("neigh")["fluid"].col, args[0]("templates")("neigh")["fluid"].data):
            Df_x__ = ((args[0]("cells")[i]("volume") / args[1]("lhs")[i][j]) + \
                     (args[0]("cells")[j]("volume") / args[1]("lhs")[i][j])) / 2
            Df_y__ = ((args[0]("cells")[i]("volume") / args[2]("lhs")[i][j]) + \
                     (args[0]("cells")[j]("volume") / args[2]("lhs")[i][j])) / 2
            Df_z__ = ((args[0]("cells")[i]("volume") / args[3]("lhs")[i][j]) + \
                     (args[0]("cells")[j]("volume") / args[3]("lhs")[i][j])) / 2
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            pcor_f__ = -args[0]("faces")[k]("prop")("rho")[-1] * \
                       np.dot(np.array([[Df_x__, 0, 0], [0, Df_y__, 0], [0, 0, Df_z__]]) \
                       * args[0]("faces")[k]("grad")["Pcor"][-1], Sf__)
            args[0]("faces")[k]("value")["u"][-1] += pcor_f__[0]
            args[0]("faces")[k]("value")["v"][-1] += pcor_f__[1]
            args[0]("faces")[k]("value")["w"][-1] += pcor_f__[2]  
        return
    def __itersolve():
        
        return

class momentum(linear):
    def __init__(self, *args):
        # args mesh : mesh, axis : int
        super().__init__(args)
        self.__axis = args[1]
    def __call__(self, *args):
        super().__call__(args)
    def __calccoef(self, *args):
        # args mesh : mesh, user : user
        # what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        coor_dict = {0: "u", 1: "v", 2: "w"}
        axes = np.delete(np.array([0, 1, 2]), self("axis"))
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["fluid"].row, \
                       args[0]("templates")("neigh")["fluid"].col, args[0]("templates")("neigh")["fluid"].data):
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            dCf__ = args[0]("geoms")("dCf", i, k, False)
            eCF__ = args[0]("geoms")("eCF", i, j, False)
            dCF__ = args[0]("geoms")("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[self("axis")] = args[0]("faces")[k]("value")[coor_dict[self("axis")]][-1]
            v__[axes[0]] = args[0]("faces")[k]("value")[coor_dict[axes[0]]][-1]
            v__[axes[1]] = args[0]("faces")[k]("value")[coor_dict[axes[1]]][-1]
            Ret__ = args[0]("faces")[k]("prop")("rho")[-1] * pow(args[0]("faces")[k]("value")["k"][-1], 2) / \
                    (args[0]("faces")[k]("prop")("miu")[-1] * args[0]("faces")[k]("value")["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0]("faces")[k]("grad")["u"][-1], args[0]("faces")[k]("grad")["v"][-1], \
                          args[0]("faces")[k]("grad")["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0]("faces")[k]("value")["k"][-1] / args[0]("faces")[k]("value")["e"][-1], \
                   args[0]("faces")[k]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0]("faces")[k]("prop")("rho")[-1] * cmiu__ * args[0]("faces")[k]("value")["k"][-1] * ts__
            graditr__  = super._linearitr(args[0], ["grad", coor_dict(self("axis"))], i, j, k)
            if prev_row == i:
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")[coor_dict[self("axis")]][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0]("faces")[k]("prop")("miu")[-1] \
                        + miut__)
                bC__ += -args[0]("faces")[k]("value")["P"][-1] + (2 * args[0]("faces")[k]("prop")("rho")[-1] * \
                         args[0]("faces")[k]("value")["k"][-1] / 3) * Sf__[self("axis")]
                self("lhs")[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                prev_row = i
            else:
                if self("axis") == 1:
                    bC__ += args[0]("cells")[prev_row]("prop")("rho")[-1] * 9.81 * args[0]("cells")[prev_row]("volume")
                self("lhs")[prev_row][prev_row] = aC__
                self("rhs")[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")[coor_dict[self("axis")]][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0]("faces")[k]("prop")("miu")[-1] \
                        + miut__)
                bC__ += -args[0]("faces")[k]("value")["P"][-1] + (2 * args[0]("faces")[k]("prop")("rho")[-1] * \
                         args[0]("faces")[k]("value")["k"][-1] / 3) * Sf__[self("axis")]
                self("lhs")[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                prev_row = i
        if self("axis") == 1:
            bC__ += args[0]("cells")[prev_row]("prop")("rho")[-1] * 9.81 * args[0]("cells")[prev_row]("volume")
        self("lhs")[prev_row][prev_row] = aC__
        self("rhs")[prev_row][0] = bC__
        for i, j, k in itertools.izip(args[0]("template")("boundary")["fluid"].row, \
                       args[0]("template")("boundary")["fluid"].col, args[0]("template")("boundary")["fluid"].data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[self("axis")] = args[0]("faces")[k]("value")[coor_dict[self("axis")]][-1]
            v__[axes[0]] = args[0]("faces")[k]("value")[coor_dict[axes[0]]][-1]
            v__[axes[1]] = args[0]("faces")[k]("value")[coor_dict[axes[1]]][-1]
            for l in args[0]("faces")[j]("bound"):
                self.__calcbound(args[0], l, i, j, v__, args[1])
        return
    def __calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        coor_dict = {0: "u", 1: "v", 2: "w"}
        axes = np.delete(np.array([0, 1, 2]), self("axis"))
        if "noslip" in args[1]:
            dperp__ = pow(linalg.norm(args[0]("cells")[args[2]]("grad")[coor_dict[self("axis")]][-1])**2 + \
                      2 * args[0]("cells")[args[2]]("value")[coor_dict[self("axis")]][-1], 0.5) - \
                      linalg.norm(args[0]("cells")[args[2]]("grad")[coor_dict[self("axis")]][-1])
            Sf__ = args[0]("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("miu") * \
                                             Sf__[self("axis")] * (1 - eCf__[self("axis")]**2) / dperp__
            self("rhs")[args[2]][0] += (args[0]("faces")[args[3]]("prop")("miu")[-1] * Sf__[self("axis")] / dperp__) * \
                                       ((args[0]("faces")[args[3]]("value")[coor_dict[self("axis")]][-1] * (1 - eCf__[self("axis")**2]))
                                       + ((args[0]("cells")[args[2]]("value")[axes[0]][-1] - args[0]("faces")[args[3]]("value")[axes[0]][-1]) * eCf__[self("axis")] * eCf__[axes[0]])
                                       + ((args[0]("cells")[args[2]]("value")[axes[1]][-1] - args[0]("faces")[args[3]]("value")[axes[1]][-1]) * eCf__[self("axis")] * eCf__[axes[1]])) \
                                       - (args[0]("faces")[args[3]]("value")["P"][-1] * Sf__[self("axis")])
            Re__ = args[0]("cells")[i]("prop")("rho")[-1] * np.sqrt(np.sum(np.array([map(lambda x: x^2, args[4])]))) * \
                   pow(args[0]("cells")[i]("volume") / args[0]("cells")[i]("prop")("miu"))
            tau__ = args[4][self("axis")] * 8 * args[0]("cells")[i]("prop")("rho")[-1] / Re__
            self("rhs")[args[2]][0] += -tau__ / (args[0]("cells")[args[2]]("prop")("rho")[-1] * 2 * args[0]("cells")[args[2]]("volume") / args[0]("faces")[args[3]]("area"))
        elif "inlet" in args[1]:
            # specified static pressure and velocity direction
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0]("cells")[args[2]]("grad")[coor_dict[self("axis")]][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")[coor_dict[self("axis")]][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0]("cells")[args[2]]("grad")[axes[0]][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")[axes[0]][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0]("cells")[args[2]]("grad")[axes[1]][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")[axes[1]][-1], eCf__) * eCf__))
            vin_v0_ = args[0]("cells")[args[2]]("value")[coor_dict[self("axis")]][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0]("cells")[args[2]]("value")[axes[0]][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0]("cells")[args[2]]("value")[axes[1]][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([0.00, 0.00, 0.00], dtype = float)
            vin__[self("axis")] = vin_v0_
            vin__[axes[0]] = vin_v1_
            vin__[axes[1]] = vin_v2_
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__)
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * np.dot(grad_vin_v0_, dCf__) -\
                                        args[5]("init_val").loc[self("current_time"), "P"] * Sf__[self("axis")]
        elif "outlet" int args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0]("cells")[args[2]]("grad")[coor_dict[self("axis")]][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")[coor_dict[self("axis")]][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0]("cells")[args[2]]("grad")[axes[0]][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")[axes[0]][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0]("cells")[args[2]]("grad")[axes[1]][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")[axes[1]][-1], eCf__) * eCf__))
            vout_v0_ = args[0]("cells")[args[2]]("value")[coor_dict[self("axis")]][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0]("cells")[args[2]]("value")[axes[0]][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0]("cells")[args[2]]("value")[axes[1]][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([0.00, 0.00, 0.00], dtype = float)
            vout__[self("axis")] = vout_v0_
            vout__[axes[0]] = vout_v1_
            vout__[axes[1]] = vout_v2_
            pout__ = args[0]("cells")[args[2]]("value")["P"][-1] + np.dot(args[0]("cells")[args[2]]("grad")["P"][-1], dCf__)
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__)
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__) * np.dot(grad_vout_v0_, dCf__) -\
                                        pout__ * Sf__[self("axis")]
        else:
            pass
        return
    def __calcwall(self, *args):
        # args mesh : mesh
        coor_dict = {0: "u", 1: "v", 2: "w"}
        for i in args[0]("cells").keys():
            if args[0]("cells")[i]("iswall") >= 0:
                v__ = np.array([0.00, 0.00, 0.00], dtype = float)
                v__[0] = args[0]("cells")[i]("value")["u"][-1]
                v__[1] = args[0]("cells")[i]("value")["v"][-1]
                v__[2] = args[0]("cells")[i]("value")["w"][-1]
                v_val__ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v__)])))
                Sf_wall__ = -args[0]("geoms")("Sf", i, args[0]("cells")[i]("iswall"), False)
                v_parallel_ = np.cross(np.cross(v__, Sf_wall__), Sf_wall__)
                v_parallel_val_ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v_parallel_)])))
                check = np.dot(v_val__, v_parallel__)
                if check >= 0:
                    theta = math.acos(np.dot(v__, v_parallel) / (v_val__ * v_parallel_val_))
                else:
                    theta = math.acos(np.dot(v__, -v_parallel) / (v_val__ * v_parallel_val_))
                v_val__ = v_val__ * math.sin(theta)
                Ret__ = args[0]("cells")[i]("prop")("rho")[-1] * pow(args[0]("cells")[i]("value")["k"][-1], 2) / \
                        (args[0]("cells")[i]("prop")("miu")[-1] * args[0]("cells")[i]("value")["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                gradCfluid__ = args[0]("cells")[i]("grad")[coor_dict[self("axis")]][-1]
                dperp__ = (np.sqrt(2 * args[0]("cells")[i]("value")[coor_dict[self("axis")]][-1]) - 1) * \
                            np.sqrt(gradCfluid__[0]**2 + gradCfluid__[1]**2 + gradCfluid__[2]**2)
                dCplus__ = dperp__ * pow(cmiu__, 0.25) * np.sqrt(args[0]("cells")[i]("value")["k"][-1]) * \
                            args[0]("cells")[i]("prop")("rho")[-1] / args[0]("cells")[i]("prop")("miu")[-1]
                dCplus__ = np.max(np.array([dCplus__, 11.06]))
                miutau__ = v_val_ * 0.41 / (np.log(dCplus__) + 5.25)
                dplusv__ = dperp__ * miutau__ * args[0]("cells")[i]("prop")("rho")[-1] / \
                            args[0]("cells")[i]("prop")("miu")[-1]
                vplus__ = np.log(dplusv__) / 0.41 + 5.25
                args[0]("cells")[i]("value")[coor_dict[self("axis")]][-1] = vplus__
        return
    def __itersolve():
        
        return

class turb_k(linear):
    def __init__(self, *args):
        super().__init__(args)
    def __call__(self, *args):
        super().__call__(args)
    def __calccoef(self, *args):
        # args mesh : mesh, user : user
        # what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["fluid"].row, \
                       args[0]("templates")("neigh")["fluid"].col, args[0]("templates")("neigh")["fluid"].data):
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            dCf__ = args[0]("geoms")("dCf", i, k, False)
            eCF__ = args[0]("geoms")("eCF", i, j, False)
            dCF__ = args[0]("geoms")("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            Ret__ = args[0]("faces")[k]("prop")("rho")[-1] * pow(args[0]("faces")[k]("value")["k"][-1], 2) / \
                    (args[0]("faces")[k]("prop")("miu")[-1] * args[0]("faces")[k]("value")["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0]("faces")[k]("grad")["u"][-1], args[0]("faces")[k]("grad")["v"][-1], \
                          args[0]("faces")[k]("grad")["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0]("faces")[k]("value")["k"][-1] / args[0]("faces")[k]("value")["e"][-1], \
                   args[0]("faces")[k]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0]("faces")[k]("prop")("rho")[-1] * cmiu__ * args[0]("faces")[k]("value")["k"][-1] * ts__
            graditr__  = super._linearitr(args[0], ["grad", "k"], i, j, k)
            if prev_row == i:
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["k"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0]("faces")[k]("prop")("miu")[-1] \
                        + miut__)
                self("lhs")[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                prev_row = i
            else:
                gradC_u__ = args[0]("cells")[prev_row]("grad")["u"][-1]
                gradC_v__ = args[0]("cells")[prev_row]("grad")["v"][-1]
                gradC_w__ = args[0]("cells")[prev_row]("grad")["w"][-1]
                phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                          + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
                Ret_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * pow(args[0]("cells")[prev_row]("value")["k"][-1], 2) / \
                        (args[0]("cells")[prev_row]("prop")("miu")[-1] * args[0]("cells")[prev_row]("value")["e"][-1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([args[0]("cells")[prev_row]("grad")["u"][-1], args[0]("cells")[prev_row]("grad")["v"][-1], \
                            args[0]("cells")[prev_row]("grad")["w"][-1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
                ts_C_ = np.min(np.array([args[0]("cells")[prev_row]("value")["k"][-1] / args[0]("cells")[prev_row]("value")["e"][-1], \
                    args[0]("cells")[prev_row]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * cmiu_C_ * args[0]("cells")[prev_row]("value")["k"][-1] * ts_C_
                bC__ += (miut_C_ * phi_v__ - args[0]("cells")[prev_row]("prop")("rho")[-1] * args[0]("cells")[prev_row]("value")["e"][-1]) \
                        * args[0]("cells")[prev_row]("volume")
                self("lhs")[prev_row][prev_row] = aC__
                self("rhs")[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["k"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0]("faces")[k]("prop")("miu")[-1] \
                        + miut__)
                self("lhs")[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__)
                prev_row = i
        gradC_u__ = args[0]("cells")[prev_row]("grad")["u"][-1]
        gradC_v__ = args[0]("cells")[prev_row]("grad")["v"][-1]
        gradC_w__ = args[0]("cells")[prev_row]("grad")["w"][-1]
        phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                    + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
        Ret_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * pow(args[0]("cells")[prev_row]("value")["k"][-1], 2) / \
                (args[0]("cells")[prev_row]("prop")("miu")[-1] * args[0]("cells")[prev_row]("value")["e"][-1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([args[0]("cells")[prev_row]("grad")["u"][-1], args[0]("cells")[prev_row]("grad")["v"][-1], \
                    args[0]("cells")[prev_row]("grad")["w"][-1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
        ts_C_ = np.min(np.array([args[0]("cells")[prev_row]("value")["k"][-1] / args[0]("cells")[prev_row]("value")["e"][-1], \
            args[0]("cells")[prev_row]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
        miut_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * cmiu_C_ * args[0]("cells")[prev_row]("value")["k"][-1] * ts_C_
        bC__ += (miut_C_ * phi_v__ - args[0]("cells")[prev_row]("prop")("rho")[-1] * args[0]("cells")[prev_row]("value")["e"][-1]) \
                * args[0]("cells")[prev_row]("volume")
        self("lhs")[prev_row][prev_row] = aC__
        self("rhs")[prev_row][0] = bC__
        for i, j, k in itertools.izip(args[0]("template")("boundary")["fluid"].row, \
                       args[0]("template")("boundary")["fluid"].col, args[0]("template")("boundary")["fluid"].data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            for l in args[0]("faces")[j]("bound"):
                self.__calcbound(args[0], l, i, j, v__, args[1])
        return
    def __calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        if "inlet" in args[1]:
            # specified value; zero gradient at inlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0]("cells")[args[2]]("grad")["u"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["u"][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0]("cells")[args[2]]("grad")["v"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["v"][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0]("cells")[args[2]]("grad")["w"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["w"][-1], eCf__) * eCf__))
            vin_v0_ = args[0]("cells")[args[2]]("value")["u"][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0]("cells")[args[2]]("value")["v"][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0]("cells")[args[2]]("value")["w"][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([vin_v0_, vin_v1_, vin_v2_], dtype = float)
            grad_kin_ = np.dot(args[0]("cells")[args[2]]("grad")["k"][-1] - \
                         (np.dot(args[0]("cells")[args[2]]("grad")["k"][-1], eCf__) * eCf__))
            kin_ = 0.5 * np.dot(vin__, vin__) * 0.01**2
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * kin_
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * np.dot(grad_kin_, dCf__)
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0]("cells")[args[2]]("grad")["u"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["u"][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0]("cells")[args[2]]("grad")["v"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["v"][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0]("cells")[args[2]]("grad")["w"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["w"][-1], eCf__) * eCf__))
            vout_v0_ = args[0]("cells")[args[2]]("value")["u"][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0]("cells")[args[2]]("value")["v"][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0]("cells")[args[2]]("value")["w"][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([vout_v0_, vout_v1_, vout_v2_], dtype = float)
            grad_kout_ = np.dot(args[0]("cells")[args[2]]("grad")["k"][-1] - \
                         (np.dot(args[0]("cells")[args[2]]("grad")["k"][-1], eCf__) * eCf__))
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__)
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__) * np.dot(grad_kout_, dCf__)
        else:
            pass
        return
    def __calcwall(self, *args):
        # args mesh : mesh
        for i in args[0]("cells").keys():
            if args[0]("cells")[i]("iswall") >= 0:
                Ret__ = args[0]("cells")[i]("prop")("rho")[-1] * pow(args[0]("cells")[i]("value")["k"][-1], 2) / \
                        (args[0]("cells")[i]("prop")("miu")[-1] * args[0]("cells")[i]("value")["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                args[0]("cells")[i]("value")["k"][-1] = 1 / np.sqrt(cmiu__)
        return
    def __itersolve():
        
        return

class turb_e(linear):
    def __init__(self, *args):
        super().__init__(args)
    def __call__(self, *args):
        super().__call__(args)
    def __calccoef(self, *args):
        # args mesh : mesh, user : user
        # what : str, time_step : int/double
        # fluid only
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["fluid"].row, \
                       args[0]("templates")("neigh")["fluid"].col, args[0]("templates")("neigh")["fluid"].data):
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            dCf__ = args[0]("geoms")("dCf", i, k, False)
            eCF__ = args[0]("geoms")("eCF", i, j, False)
            dCF__ = args[0]("geoms")("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            Ret__ = args[0]("faces")[k]("prop")("rho")[-1] * pow(args[0]("faces")[k]("value")["k"][-1], 2) / \
                    (args[0]("faces")[k]("prop")("miu")[-1] * args[0]("faces")[k]("value")["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0]("faces")[k]("grad")["u"][-1], args[0]("faces")[k]("grad")["v"][-1], \
                          args[0]("faces")[k]("grad")["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0]("faces")[k]("value")["k"][-1] / args[0]("faces")[k]("value")["e"][-1], \
                   args[0]("faces")[k]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0]("faces")[k]("prop")("rho")[-1] * cmiu__ * args[0]("faces")[k]("value")["k"][-1] * ts__
            graditr__  = super._linearitr(args[0], ["grad", "e"], i, j, k)
            if prev_row == i:
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__ / 1.3)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["e"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0]("faces")[k]("prop")("miu")[-1] \
                        + miut__ / 1.3)
                self("lhs")[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__ / 1.3)
                prev_row = i
            else:
                gradC_u__ = args[0]("cells")[prev_row]("grad")["u"][-1]
                gradC_v__ = args[0]("cells")[prev_row]("grad")["v"][-1]
                gradC_w__ = args[0]("cells")[prev_row]("grad")["w"][-1]
                phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                          + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
                Ret_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * pow(args[0]("cells")[prev_row]("value")["k"][-1], 2) / \
                        (args[0]("cells")[prev_row]("prop")("miu")[-1] * args[0]("cells")[prev_row]("value")["e"][-1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([args[0]("cells")[prev_row]("grad")["u"][-1], args[0]("cells")[prev_row]("grad")["v"][-1], \
                            args[0]("cells")[prev_row]("grad")["w"][-1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
                ts_C_ = np.min(np.array([args[0]("cells")[prev_row]("value")["k"][-1] / args[0]("cells")[prev_row]("value")["e"][-1], \
                    args[0]("cells")[prev_row]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * cmiu_C_ * args[0]("cells")[prev_row]("value")["k"][-1] * ts_C_
                ceps2 = 1.92 * (1 - 0.3 * math.exp(-1 * Ret_C_**2)) 
                bC__ += (1.44 * miut_C_ * phi_v__ / ts_C_ - ceps2 * args[0]("cells")[prev_row]("prop")("rho")[-1] * args[0]("cells")[prev_row]("value")["e"][-1] / ts_C_) \
                        * args[0]("cells")[prev_row]("volume")
                self("lhs")[prev_row][prev_row] = aC__
                self("rhs")[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__ / 1.3)
                bC__ += np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["e"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * (args[0]("faces")[k]("prop")("miu")[-1] \
                        + miut__ / 1.3)
                self("lhs")[i][j] = (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("miu")[-1] + miut__ / 1.3)
                prev_row = i
        gradC_u__ = args[0]("cells")[prev_row]("grad")["u"][-1]
        gradC_v__ = args[0]("cells")[prev_row]("grad")["v"][-1]
        gradC_w__ = args[0]("cells")[prev_row]("grad")["w"][-1]
        phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                    + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
        Ret_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * pow(args[0]("cells")[prev_row]("value")["k"][-1], 2) / \
                (args[0]("cells")[prev_row]("prop")("miu")[-1] * args[0]("cells")[prev_row]("value")["e"][-1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([args[0]("cells")[prev_row]("grad")["u"][-1], args[0]("cells")[prev_row]("grad")["v"][-1], \
                    args[0]("cells")[prev_row]("grad")["w"][-1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
        ts_C_ = np.min(np.array([args[0]("cells")[prev_row]("value")["k"][-1] / args[0]("cells")[prev_row]("value")["e"][-1], \
            args[0]("cells")[prev_row]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
        miut_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * cmiu_C_ * args[0]("cells")[prev_row]("value")["k"][-1] * ts_C_
        ceps2 = 1.92 * (1 - 0.3 * math.exp(-1 * Ret_C_**2)) 
        bC__ += (1.44 * miut_C_ * phi_v__ / ts_C_ - ceps2 * args[0]("cells")[prev_row]("prop")("rho")[-1] * args[0]("cells")[prev_row]("value")["e"][-1] / ts_C_) \
                * args[0]("cells")[prev_row]("volume")
        self("lhs")[prev_row][prev_row] = aC__
        self("rhs")[prev_row][0] = bC__
        for i, j, k in itertools.izip(args[0]("template")("boundary")["fluid"].row, \
                       args[0]("template")("boundary")["fluid"].col, args[0]("template")("boundary")["fluid"].data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            for l in args[0]("faces")[j]("bound"):
                self.__calcbound(args[0], l, i, j, v__, args[1])
        return
    def __calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        if "inlet" in args[1]:
            # specified value; zero gradient at inlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0]("cells")[args[2]]("grad")["u"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["u"][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0]("cells")[args[2]]("grad")["v"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["v"][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0]("cells")[args[2]]("grad")["w"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["w"][-1], eCf__) * eCf__))
            vin_v0_ = args[0]("cells")[args[2]]("value")["u"][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0]("cells")[args[2]]("value")["v"][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0]("cells")[args[2]]("value")["w"][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([vin_v0_, vin_v1_, vin_v2_], dtype = float)
            grad_ein_ = np.dot(args[0]("cells")[args[2]]("grad")["e"][-1] - \
                         (np.dot(args[0]("cells")[args[2]]("grad")["e"][-1], eCf__) * eCf__))
            ein_ = pow(0.5 * np.dot(vin__, vin__) * 0.01**2, 1.5) * 0.09 / (0.1 * args[0]("cells")[args[2]]("volume"))
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * ein_
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * np.dot(grad_ein_, dCf__)
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0]("cells")[args[2]]("grad")["u"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["u"][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0]("cells")[args[2]]("grad")["v"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["v"][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0]("cells")[args[2]]("grad")["w"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["w"][-1], eCf__) * eCf__))
            vout_v0_ = args[0]("cells")[args[2]]("value")["u"][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0]("cells")[args[2]]("value")["v"][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0]("cells")[args[2]]("value")["w"][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([vout_v0_, vout_v1_, vout_v2_], dtype = float)
            grad_eout_ = np.dot(args[0]("cells")[args[2]]("grad")["e"][-1] - \
                         (np.dot(args[0]("cells")[args[2]]("grad")["e"][-1], eCf__) * eCf__))
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__)
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__) * np.dot(grad_eout_, dCf__)
        else:
            pass
        return
    def __calcwall(self, *args):
        # args mesh : mesh
        for i in args[0]("cells").keys():
            if args[0]("cells")[i]("iswall") >= 0:
                v__ = np.array([0.00, 0.00, 0.00], dtype = float)
                v__[0] = args[0]("cells")[i]("value")["u"][-1]
                v__[1] = args[0]("cells")[i]("value")["v"][-1]
                v__[2] = args[0]("cells")[i]("value")["w"][-1]
                v_val_ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v__)])))
                Ret__ = args[0]("cells")[i]("prop")("rho")[-1] * pow(args[0]("cells")[i]("value")["k"][-1], 2) / \
                        (args[0]("cells")[i]("prop")("miu")[-1] * args[0]("cells")[i]("value")["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                gradCfluid__ = args[0]("cells")[i]("grad")["T"][-1]
                dperp__ = (np.sqrt(2 * args[0]("cells")[i]("value")["e"][-1]) - 1) * \
                            np.sqrt(gradCfluid__[0]**2 + gradCfluid__[1]**2 + gradCfluid__[2]**2)
                dCplus__ = dperp__ * pow(cmiu__, 0.25) * np.sqrt(args[0]("cells")[i]("value")["k"][-1]) * \
                            args[0]("cells")[i]("prop")("rho")[-1] / args[0]("cells")[i]("prop")("miu")[-1]
                dCplus__ = np.max(np.array([dCplus__, 11.06]))
                miutau__ = v_val_ * 0.41 / (np.log(dCplus__) + 5.25)
                eplus__ =  args[0]("cells")[i]("prop")("miu")[-1] / (args[0]("cells")[i]("prop")("rho")[-1] * miutau__ * 0.41 * dperp__)
                args[0]("cells")[i]("value")["e"][-1] = eplus__
        return
    def __itersolve():
        
        return

class energy(linear):
    def __init__(self, *args):
        super().__init__(args)
    def __call__(self, *args):
        super().__call__(args)
    def __calccoef(self, *args):
        # args mesh : mesh, user : user
        # what : str, time_step : int/double
        # fluid
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["fluid"].row, \
                       args[0]("templates")("neigh")["fluid"].col, args[0]("templates")("neigh")["fluid"].data):
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            dCf__ = args[0]("geoms")("dCf", i, k, False)
            eCF__ = args[0]("geoms")("eCF", i, j, False)
            dCF__ = args[0]("geoms")("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            Ret__ = args[0]("faces")[k]("prop")("rho")[-1] * pow(args[0]("faces")[k]("value")["k"][-1], 2) / \
                    (args[0]("faces")[k]("prop")("miu")[-1] * args[0]("faces")[k]("value")["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0]("faces")[k]("grad")["u"][-1], args[0]("faces")[k]("grad")["v"][-1], \
                          args[0]("faces")[k]("grad")["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0]("faces")[k]("value")["k"][-1] / args[0]("faces")[k]("value")["e"][-1], \
                   args[0]("faces")[k]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0]("faces")[k]("prop")("rho")[-1] * cmiu__ * args[0]("faces")[k]("value")["k"][-1] * ts__
            graditr__  = super._linearitr(args[0], ["grad", "T"], i, j, k)
            Pr__ = args[0]("faces")[k]("prop")("miu")[-1] / (args[0]("faces")[k]("prop")("rho")[-1] * args[0]("faces")[k]("prop")("alpha")[-1])
            if prev_row == i:
                aC__ += args[0]("faces")[k]("prop")("cp")[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0]("faces")[k]("prop")("cp")[-1] * (args[0]("faces")[k]("prop")("miu")[-1] / Pr__ + miut__ / 0.9)
                bC__ += args[0]("faces")[k]("prop")("cp")[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) *  args[0]("faces")[k]("prop")("cp")[-1] * \
                        (args[0]("faces")[k]("prop")("miu")[-1] / Pr__ + miut__ / 0.9)
                self("lhs")[i][j] = args[0]("faces")[k]("prop")("cp")[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * \
                                    args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * args[0]("faces")[k]("prop")("cp")[-1] * (args[0]("faces")[k]("prop")("miu")[-1] \
                                      / Pr__ + miut__ / 0.9)
                prev_row = i
            else:
                gradC_u__ = args[0]("cells")[prev_row]("grad")["u"][-1]
                gradC_v__ = args[0]("cells")[prev_row]("grad")["v"][-1]
                gradC_w__ = args[0]("cells")[prev_row]("grad")["w"][-1]
                phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                          + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
                Ret_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * pow(args[0]("cells")[prev_row]("value")["k"][-1], 2) / \
                        (args[0]("cells")[prev_row]("prop")("miu")[-1] * args[0]("cells")[prev_row]("value")["e"][-1])
                cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
                St_tensor_C_ = np.array([args[0]("cells")[prev_row]("grad")["u"][-1], args[0]("cells")[prev_row]("grad")["v"][-1], \
                            args[0]("cells")[prev_row]("grad")["w"][-1]])
                St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
                St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
                ts_C_ = np.min(np.array([args[0]("cells")[prev_row]("value")["k"][-1] / args[0]("cells")[prev_row]("value")["e"][-1], \
                    args[0]("cells")[prev_row]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
                miut_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * cmiu_C_ * args[0]("cells")[prev_row]("value")["k"][-1] * ts_C_
                bC__ += (args[0]("cell")[prev_row]("prop")("miu")[-1] + miut_C_) * phi_v__ * args[0]("cells")[prev_row]("volume")
                self("lhs")[prev_row][prev_row] = aC__
                self("rhs")[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += args[0]("faces")[k]("prop")("cp")[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0]("faces")[k]("prop")("cp")[-1] * (args[0]("faces")[k]("prop")("miu")[-1] / Pr__ + miut__ / 0.9)
                bC__ += args[0]("faces")[k]("prop")("cp")[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) *  args[0]("faces")[k]("prop")("cp")[-1] * \
                        (args[0]("faces")[k]("prop")("miu")[-1] / Pr__ + miut__ / 0.9)
                self("lhs")[i][j] = args[0]("faces")[k]("prop")("cp")[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * \
                                    args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * args[0]("faces")[k]("prop")("cp")[-1] * (args[0]("faces")[k]("prop")("miu")[-1] \
                                      / Pr__ + miut__ / 0.9)
                prev_row = i
        gradC_u__ = args[0]("cells")[prev_row]("grad")["u"][-1]
        gradC_v__ = args[0]("cells")[prev_row]("grad")["v"][-1]
        gradC_w__ = args[0]("cells")[prev_row]("grad")["w"][-1]
        phi_v__ = 2 * ( gradC_u__[0]**2  + gradC_v__[1]**2 + gradC_w__[2]**2 ) + pow(gradC_u__[1] + gradC_v__[0], 2) \
                    + pow(gradC_u__[2] + gradC_w__[0], 2) + pow(gradC_v__[2] + gradC_w__[1], 2)
        Ret_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * pow(args[0]("cells")[prev_row]("value")["k"][-1], 2) / \
                (args[0]("cells")[prev_row]("prop")("miu")[-1] * args[0]("cells")[prev_row]("value")["e"][-1])
        cmiu_C_ = 0.09 * math.exp(-3.4 / pow(1 + Ret_C_/50, 2))
        St_tensor_C_ = np.array([args[0]("cells")[prev_row]("grad")["u"][-1], args[0]("cells")[prev_row]("grad")["v"][-1], \
                    args[0]("cells")[prev_row]("grad")["w"][-1]])
        St_tensor_C_ = (St_tensor_C_ + np.transpose(St_tensor_C_)) * 0.5
        St_C_ = np.sqrt(St_tensor_C_.dot(St_tensor_C_))
        ts_C_ = np.min(np.array([args[0]("cells")[prev_row]("value")["k"][-1] / args[0]("cells")[prev_row]("value")["e"][-1], \
            args[0]("cells")[prev_row]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu_C_ * St_C_)]))
        miut_C_ = args[0]("cells")[prev_row]("prop")("rho")[-1] * cmiu_C_ * args[0]("cells")[prev_row]("value")["k"][-1] * ts_C_
        bC__ += (args[0]("cell")[prev_row]("prop")("miu")[-1] + miut_C_) * phi_v__ * args[0]("cells")[prev_row]("volume")
        self("lhs")[prev_row][prev_row] = aC__
        self("rhs")[prev_row][0] = bC__
        for i, j, k in itertools.izip(args[0]("template")("boundary")["fluid"].row, \
                       args[0]("template")("boundary")["fluid"].col, args[0]("template")("boundary")["fluid"].data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            for l in args[0]("faces")[j]("bound"):
                self.__calcbound(args[0], l, i, j, v__, args[1])
        # solid
        prev_row = 0
        aC__ = 0.00
        bC__ = 0.00
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["solid"].row, \
                       args[0]("templates")("neigh")["solid"].col, args[0]("templates")("neigh")["solid"].data):
            Sf__  = args[0]("geoms")("Sf", i, k, False)
            dCf__ = args[0]("geoms")("dCf", i, k, False)
            eCF__ = args[0]("geoms")("eCF", i, j, False)
            dCF__ = args[0]("geoms")("dCF", i, j, True)
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            Ret__ = args[0]("faces")[k]("prop")("rho")[-1] * pow(args[0]("faces")[k]("value")["k"][-1], 2) / \
                    (args[0]("faces")[k]("prop")("miu")[-1] * args[0]("faces")[k]("value")["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            St_tensor__ = np.array([args[0]("faces")[k]("grad")["u"][-1], args[0]("faces")[k]("grad")["v"][-1], \
                          args[0]("faces")[k]("grad")["w"][-1]])
            St_tensor__ = (St_tensor__ + np.transpose(St_tensor__)) * 0.5
            St__ = np.sqrt(St_tensor__.dot(St_tensor__))
            ts__ = np.min(np.array([args[0]("faces")[k]("value")["k"][-1] / args[0]("faces")[k]("value")["e"][-1], \
                   args[0]("faces")[k]("prop")("alpha")[-1] / (np.sqrt(6) * cmiu__ * St__)]))
            miut__ = args[0]("faces")[k]("prop")("rho")[-1] * cmiu__ * args[0]("faces")[k]("value")["k"][-1] * ts__
            graditr__  = super._linearitr(args[0], ["grad", "T"], i, j, k)
            if prev_row == i:
                aC__ += args[0]("faces")[k]("prop")("cp")[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0]("faces")[k]("prop")("k")[-1]
                bC__ += args[0]("faces")[k]("prop")("cp")[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * args[0]("faces")[k]("prop")("k")[-1]
                self("lhs")[i][j] = args[0]("faces")[k]("prop")("cp")[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("k")[-1])
                prev_row = i
            else:
                for l in args[0]("cells")[i]("domain"):
                    if l in args[1]("source").columns:
                        bC__ += args[i]("source").loc[self(current_time), l] * args[0]("geoms")("Sf", prev_row, k, True)
                self("lhs")[prev_row][prev_row] = aC__
                self("rhs")[prev_row][0] = bC__
                aC__ = 0.00
                bC__ = 0.00
                aC__ += args[0]("faces")[k]("prop")("cp")[-1] * (1 - np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] \
                        * np.dot(v__, Sf__)
                aC__ += (np.dot(eCF__, Sf__) / dCF__) * args[0]("faces")[k]("prop")("k")[-1]
                bC__ += args[0]("faces")[k]("prop")("cp")[-1] * np.dot(np.dot(np.dot(graditr__, eCF__) * eCF__ - \
                        (args[0]("cells")[i]("grad")["T"][-1] +  graditr__), dCf__) / 2, dCf__) * \
                        args[0]("faces")[k]("prop")("rho")[-1] * np.dot(v__, Sf__) 
                bC__ += np.dot(graditr__ - (np.dot(graditr__, eCF__) * eCF__), Sf__) * args[0]("faces")[k]("prop")("k")[-1]
                self("lhs")[i][j] = args[0]("faces")[k]("prop")("cp")[-1] * (np.dot(eCF__, dCf__) / (2 * dCF__)) * args[0]("faces")[k]("prop")("rho")[-1] * \
                                     np.dot(v__, Sf__)
                self("lhs")[i][j] += -(np.dot(eCF__, Sf__) / dCF__) * (args[0]("faces")[k]("prop")("k")[-1])
                prev_row = i
        for l in args[0]("cells")[i]("domain"):
            if l in args[1]("source").columns:
                bC__ += args[i]("source").loc[self(current_time), l] * args[0]("geoms")("Sf", prev_row, k, True)
        self("lhs")[prev_row][prev_row] = aC__
        self("rhs")[prev_row][0] = bC__
        for i, j, k in itertools.izip(args[0]("template")("boundary")["solid"].row, \
                       args[0]("template")("boundary")["solid"].col, args[0]("template")("boundary")["solid"].data):
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            for l in args[0]("faces")[j]("bound"):
                self.__calcbound(args[0], l, i, j, v__, args[1])
        # conj
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["conj"].row, \
                       args[0]("templates")("neigh")["conj"].col, args[0]("templates")("neigh")["conj"].data):
            if "fluid" in args[0]("cells")[i]("domain"):
                fluid_id = i
                solid_id = j
            else:
                fluid_id = j
                solid_id = j
            v__ = np.array([0.00, 0.00, 0.00], dtype = float)
            v__[0] = args[0]("faces")[k]("value")["u"][-1]
            v__[1] = args[0]("faces")[k]("value")["v"][-1]
            v__[2] = args[0]("faces")[k]("value")["w"][-1]
            Ret__ = args[0]("faces")[k]("prop")("rho")[-1] * pow(args[0]("faces")[k]("value")["k"][-1], 2) / \
                    (args[0]("faces")[k]("prop")("miu")[-1] * args[0]("faces")[k]("value")["e"][-1])
            cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
            gradCfluid__ = args[0]("cells")[fluid_id]("grad")["T"][-1]
            hb__ = args[0]("cells")[fluid_id]("prop")("rho")[-1] * args[0]("cells")[fluid_id]("prop")("cp")[-1] * \
                   pow(cmiu__, 0.25) * np.sqrt(args[0]("cells")[fluid_id]("value")["k"][-1]) / args[0]("cells")[args[2]]("value")["T"][-1]
            graditr__  = super._linearitr(args[0], ["grad", "T"], i, j, k)
            self("lhs")[fluid_id][fluid_id] += hb__ * np.dot(args[0]("geoms")("eCF", fluid_id, solid_id, False), \
                                               args[0]("geoms")("dCf", fluid_id, k, False)) * \
                                               args[0]("geoms")("Sf", fluid_id, k, True) / args[0]("geoms")("dCF", \
                                               fluid_id, solid_id, True)
            self("lhs")[solid_id][solid_id] += hb__ * np.dot(args[0]("geoms")("eCF", solid_id, fluid_id, False), \
                                               args[0]("geoms")("dCf", solid_id, k, False)) * \
                                               args[0]("geoms")("Sf", solid_id, k, True) / args[0]("geoms")("dCF", \
                                               solid_id, fluid_id, True)
            self("lhs")[fluid_id][solid_id] += -hb__ * np.dot(args[0]("geoms")("eCF", fluid_id, solid_id, False), \
                                               args[0]("geoms")("dCf", fluid_id, k, False)) * \
                                               args[0]("geoms")("Sf", fluid_id, k, True) / args[0]("geoms")("dCF", \
                                               fluid_id, solid_id, True)
            self("lhs")[solid_id][fluid_id] += -hb__ * np.dot(args[0]("geoms")("eCF", solid_id, fluid_id, False), \
                                               args[0]("geoms")("dCf", solid_id, k, False)) * \
                                               args[0]("geoms")("Sf", solid_id, k, True) / args[0]("geoms")("dCF", \
                                               solid_id, fluid_id, True)
            self("rhs")[fluid_id][0] += hb__ * np.dot((graditr__ - np.dot(graditr__, args[0]("geoms")("eCF", \
                                        fluid_id, solid_id, False)) * args[0]("geoms")("eCF", fluid_id, solid_id, False)), \
                                        args[0]("geoms")("dCf", fluid_id, k, False))
            self("rhs")[solid_id][0] += -hb__ * np.dot((graditr__ - np.dot(graditr__, args[0]("geoms")("eCF", \
                                        solid_id, fluid_id, False)) * args[0]("geoms")("eCF", solid_id, fluid_id, False)), \
                                        args[0]("geoms")("dCf", solid_id, k, False))
        return
    def __calcbound(self, *args):
        # void
        # args mesh : mesh, bound name : str, cell id : int, face id : int, v__ : np.array([]), user : user
        if "hamb" in args[1]:
            Tamb__ = args[5]("source")[self("current_time"), "Tamb"]
            Tsky__ = 0.0552 * pow(Tamb__, 1.5)
            hsky = 5.67 * pow(10, -8) * args[0]("faces")[args[3]]("prop")("eps")[-1] * \
                   (args[0]("faces")[args[3]]("value")["T"][-1] + Tsky__) * \
                   (args[0]("faces")[args[3]]("value")["T"][-1]**2 + Tsky__**2) * \
                   (args[0]("faces")[args[3]]("value")["T"][-1] - Tsky__) / \
                   (args[0]("faces")[args[3]]("value")["T"][-1] - Tamb__) 
            Tfilm__ = (args[0]("faces")[args[3]]("value")["T"][-1] + Tamb__) / 2
            rho_film__ = 1 / HAPropsSI("Vha", "P", args[5]("source").loc[self("current_time"), "Pamb"], "T", Tfilm__, "W", \
                         args[5]("source").loc[self("current_time"), "Wamb"]
            miu_film__ = HAPropsSI("mu", "P", args[5]("source").loc[self("current_time"), "Pamb"], "T", Tfilm__, "W", \
                       args[5]("source", self("current_time"), "Wamb"))
            alpha_film__ = HAPropsSI("alpha", "P", args[5]("source").loc[self("current_time"), "Pamb"], "T", Tfilm__, "W", \
                           args[5]("source").loc[self("current_time"), "Wamb"])
            RaL__ = 9.81 * (args[0]("faces")[args[3]]("value")["T"][-1] - Tamb__) * \
                    np.sqrt(args[0]("faces")[args[3]]("area"))**3 * rho_film__ / \
                    (Tfilm__ * miu_film__ * alpha_film__)   
            if RaL__ <= pow(10, 7):
                Nu_N__ = 0.54 * pow(RaL__, 0.25)
            else:
                Nu_N__ = 0.15 * pow(RaL__, 1/3)
            hconv = Nu_N__ * args[0]("faces")[args[3]]("prop")("k")[-1] / np.sqrt(args[0]("faces")[args[3]]("area"))         
            self("rhs")[args[2]][0] += -(hsky + hconv) * (args[0]("faces")[args[3]]("value")["T"][-1] - Tamb__) \
                                             * args[0]("geoms")("Sf", args[2], args[3], True)
        elif "s2s" in args[1]:
            # von Neumann
            self("rhs")[args[2]][0] += -args[0]("clusts")[int(args[1][-1])]("value")["q"][-1] * \
                                       args[0]("geoms")("Sf", args[2], args[3], True)
        elif "inlet" in args[1]:
            # specified value; zero gradient at inlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vin_v0_ = np.dot(args[0]("cells")[args[2]]("grad")["u"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["u"][-1], eCf__) * eCf__))
            grad_vin_v1_ = np.dot(args[0]("cells")[args[2]]("grad")["v"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["v"][-1], eCf__) * eCf__))
            grad_vin_v2_ = np.dot(args[0]("cells")[args[2]]("grad")["w"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["w"][-1], eCf__) * eCf__))
            vin_v0_ = args[0]("cells")[args[2]]("value")["u"][-1] + np.dot(grad_vin_v0_, dCf__)
            vin_v1_ = args[0]("cells")[args[2]]("value")["v"][-1] + np.dot(grad_vin_v1_, dCf__)
            vin_v2_ = args[0]("cells")[args[2]]("value")["w"][-1] + np.dot(grad_vin_v2_, dCf__)
            vin__ = np.array([vin_v0_, vin_v1_, vin_v2_], dtype = float)
            grad_Tin_ = np.dot(args[0]("cells")[args[2]]("grad")["T"][-1] - \
                         (np.dot(args[0]("cells")[args[2]]("grad")["T"][-1], eCf__) * eCf__))
            Tin_ = pow(0.5 * np.dot(vin__, vin__) * 0.01**2, 1.5) * 0.09 / (0.1 * args[0]("cells")[args[2]]("volume"))
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * Tin_
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vin__, Sf__) * np.dot(grad_Tin_, dCf__)
        elif "outlet" in args[1]:
            # fully developed flow; zero gradient at outlet
            Sf__ = args[0]("geoms")("Sf", args[2], args[3], False)
            eCf__ = args[0]("geoms")("eCf", args[2], args[3], False)
            dCf__ = args[0]("geoms")("dCf", args[2], args[3], False)
            grad_vout_v0_ = np.dot(args[0]("cells")[args[2]]("grad")["u"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["u"][-1], eCf__) * eCf__))
            grad_vout_v1_ = np.dot(args[0]("cells")[args[2]]("grad")["v"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["v"][-1], eCf__) * eCf__))
            grad_vout_v2_ = np.dot(args[0]("cells")[args[2]]("grad")["w"][-1] - \
                          (np.dot(args[0]("cells")[args[2]]("grad")["w"][-1], eCf__) * eCf__))
            vout_v0_ = args[0]("cells")[args[2]]("value")["u"][-1] + np.dot(grad_vout_v0_, dCf__)
            vout_v1_ = args[0]("cells")[args[2]]("value")["v"][-1] + np.dot(grad_vout_v1_, dCf__)
            vout_v2_ = args[0]("cells")[args[2]]("value")["w"][-1] + np.dot(grad_vout_v2_, dCf__)
            vout__ = np.array([vout_v0_, vout_v1_, vout_v2_], dtype = float)
            grad_Tout_ = np.dot(args[0]("cells")[args[2]]("grad")["e"][-1] - \
                         (np.dot(args[0]("cells")[args[2]]("grad")["e"][-1], eCf__) * eCf__))
            self("lhs")[args[2]][args[2]] += args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__)
            self("rhs")[args[2]][0] += -args[0]("faces")[args[3]]("prop")("rho")[-1] * np.dot(vout__, Sf__) * np.dot(grad_Tout_, dCf__)
        else:
            pass
        return
    def __calcwall(self, *args):
        # args mesh : mesh
        for i in args[0]("cells").keys():
            if args[0]("cells")[i]("iswall") >= 0:
                v__ = np.array([0.00, 0.00, 0.00], dtype = float)
                v__[0] = args[0]("cells")[i]("value")["u"][-1]
                v__[1] = args[0]("cells")[i]("value")["v"][-1]
                v__[2] = args[0]("cells")[i]("value")["w"][-1]
                v_val_ = np.sqrt(np.sum(np.array([map(lambda x: x^2, v__)])))
                Ret__ = args[0]("cells")[i]("prop")("rho")[-1] * pow(args[0]("cells")[i]("value")["k"][-1], 2) / \
                        (args[0]("cells")[i]("prop")("miu")[-1] * args[0]("cells")[i]("value")["e"][-1])
                cmiu__ = 0.09 * math.exp(-3.4 / pow(1 + Ret__/50, 2))
                gradCfluid__ = args[0]("cells")[i]("grad")["T"][-1]
                dperp__ = (np.sqrt(2 * args[0]("cells")[i]("value")["T"][-1]) - 1) * \
                            np.sqrt(gradCfluid__[0]**2 + gradCfluid__[1]**2 + gradCfluid__[2]**2)
                dCplus__ = dperp__ * pow(cmiu__, 0.25) * np.sqrt(args[0]("cells")[i]("value")["k"][-1]) * \
                            args[0]("cells")[i]("prop")("rho")[-1] / args[0]("cells")[i]("prop")("miu")[-1]
                dCplus__ = np.max(np.array([dCplus__, 11.06]))
                miutau__ = v_val_ * 0.41 / (np.log(dCplus__) + 5.25)
                dplusT__ = dperp__ * miutau__ * args[0]("cells")[i]("prop")("rho")[-1] / \
                            args[0]("cells")[i]("prop")("miu")[-1]
                Pr__ = args[0]("cells")[i]("prop")("miu")[-1] / (args[0]("cells")[i]("prop")("rho")[-1] \
                        * args[0]("cells")[i]("prop")("alpha")[-1])
                beta__ = pow(3.85 * pow(Pr__, 1/3) - 1.3, 2) + 2.12 * np.log(Pr__)
                Tplus__ = 2.12 * np.log(dplusT__) + beta__ * Pr__
                args[0]("cells")[i]("value")["T"][-1] = Tplus__
        return
    def __itersolve():
        
        return

class s2s(linear):
    def __init__(self, *args):
        super().__init__(args)
    def __call__(self, *args):
        super().__call__(args)
    def __calccoef(self, *args):
        # args mesh : mesh, user : user
        # what : str, time_step : int/double
        # s2s only
        prev_row = 0
        for i, j, k in itertools.izip(args[0]("templates")("neigh")["s2s"].row, \
                       args[0]("templates")("neigh")["s2s"].col, args[0]("templates")("neigh")["s2s"].data):
            if prev_row == i:
                Tclust_C__ = 0.00
                rho_clust_F__ = 0.00
                area_clust_C__ = 0.00
                area_clust_F__ = 0.00
                eps_clust_C__ = 0.0
                for l in args[0]("clusts")[i]("member"):
                    Tclust_C__ += args[0]("faces")[l]("value")["T"][-1] * args[0]("faces")[l]("area")
                    area_clust_C__ +=  args[0]("faces")[l]("area")
                    eps_clust_C__ = args[0]("faces")[l]("prop")("eps")[-1]
                for l in args[0]("clusts")[j]("member"):
                    rho_clust_F__ += args[0]("faces")[l]("prop")("rho")[-1] * args[0]("faces")[l]("area")
                    area_clust_F__ += args[0]("faces")[l]("area")
                Tclust_C__ = Tclust_C__ / area_clust_C__
                rho_clust_F__ = rho_clust_F__ / area_clust_F__
                self("lhs")[i][j] = rho_clust_F__ * args[0]("geoms")("view", i, j)
                prev_row = i
            else:
                self("lhs")[prev_rhs][prev_rhs] = 1
                self("rhs")[prev_rhs][0] = eps_clust_C__ * 5.67 * pow(10, -8) * pow(Tclust_C__, 4)
                Tclust_C__ = 0.00
                rho_clust_F__ = 0.00
                area_clust_C__ = 0.00
                area_clust_F__ = 0.00
                eps_clust_C__ = 0.0
                for l in args[0]("clusts")[i]("member"):
                    Tclust_C__ += args[0]("faces")[l]("value")["T"][-1] * args[0]("faces")[l]("area")
                    area_clust_C__ +=  args[0]("faces")[l]("area")
                    eps_clust_C__ = args[0]("faces")[l]("prop")("eps")[-1]
                for l in args[0]("clusts")[j]("member"):
                    rho_clust_F__ += args[0]("faces")[l]("prop")("rho")[-1] * args[0]("faces")[l]("area")
                    area_clust_F__ += args[0]("faces")[l]("area")
                Tclust_C__ = Tclust_C__ / area_clust_C__
                rho_clust_F__ = rho_clust_F__ / area_clust_F__
                self("lhs")[i][j] = rho_clust_F__ * args[0]("geoms")("view", i, j)
                prev_row = i
        self("lhs")[prev_rhs][prev_rhs] = 1
        self("rhs")[prev_rhs][0] = eps_clust_C__ * 5.67 * pow(10, -8) * pow(Tclust_C__, 4)
        return
    def __itersolve():
        
        return

def export_iterate():
    
    return

def iterate():
    
    return

if __name__ == "__main__":
    print("test module")
else:
