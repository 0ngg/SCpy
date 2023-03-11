from .mesh import *
from .interpolate import least_square_gradient
from CoolProp.HumidAirProp import HAPropsSI

# ------
# SCHEME

# SIMPLE: mass flow rate at face (m), velocity (v), temperature (T), pressure (P)
# SST: tke (k), omega (w)

# iter_obj: struct; prev, current, ltime -> floats and vectors
# value: iter_obj; unit, grad
# coef: iter_obj; aC, aF, bC

class iter_obj:
    __prev = None
    __current = None
    __new = None
    __ltime = None

    @property
    def prev(self):
        return self.__prev
    @property
    def current(self):
        return self.__current
    @property
    def new(self):
        pass
    @new.setter
    def new(self, val):
        self.__new = val
    @property
    def ltime(self):
        return self.__ltime
    @property
    def last(self):
        return self.__ltime[-1]

    def __init__(self, init):
        self.__prev = init
        self.__current = init
        self.__new = init
        self.__ltime = [init]
    def update(self):
        self.__prev = self.__current
        self.__current = self.__new
    def forward(self):
        self.__ltime.append(self.__new)
        self.__prev, self.__current = self.__new
class prop:
    __crho = None
    __frho = None

    @property
    def crho(self):
        return self.__crho
    @crho.setter
    def crho(self, crho_: dict):
        self.__crho = crho_
    @property
    def frho(self):
        return self.__frho
    @frho.setter
    def frho(self, frho_: dict):
        self.__frho = frho_

    def __init__(self, P_init: float, T_init: float, ncell: int, nface: int):
        crho_ = {}; frho_ = {}; rho_init = self.calc_rho(P_init, T_init)
        for it1 in range(0, ncell):
            crho_[it1] = iter_obj(rho_init)
        for it1 in range(0, nface):
            frho_[it1] = iter_obj(rho_init)
        self.crho(crho_); self.frho(frho_)
    def update(self):
        for it1, it2 in self.crho().items():
            it2.update()
        for it1, it2 in self.frho().items():
            it2.update()
    def forward(self):
        for it1, it2 in self.crho().items():
            it2.forward()
        for it1, it2 in self.frho().items():
            it2.forward()
    def calc_rho(P_: float, T_: float, W_ = 0.02) -> float:
        return HAPropsSI("Vha", "P", P_, "T", T_, "W", W_)
    def calc_mu(P_: float, T_: float, W_ = 0.02) -> float:
        return HAPropsSI("mu", "P", P_, "T", T_, "W", W_)
    def calc_K(P_: float, T_: float, W_ = 0.02) -> float:
        return HAPropsSI("k", "P", P_, "T", T_, "W", W_)
    def calc_cp(P_: float, T_: float, W_ = 0.02) -> float:
        return HAPropsSI("cp", "P", P_, "T", T_, "W", W_)
class value:
    __funit = None # dict(iter_obj)
    __cunit = None # dict(iter_obj)
    __fgrad = None # dict(iter_obj)
    __cgrad = None # dict(iter_obj)

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

    def __init__(self, init: float, ncell: int, nface: int):
        # zero grad at init
        funit_ = {}; cunit_ = {}
        fgrad_ = {}; cgrad_ = {}
        for it1 in range(0, ncell):
            cunit_[it1] = iter_obj(init)
            cgrad_[it1] = iter_obj(np.array([0, 0, 0], dtype = float))
        for it1 in range(0, nface):
            funit_[it1] = iter_obj(init)
            fgrad_[it1] = iter_obj(np.array([0, 0, 0], dtype = float))
    def update(self):
        for it1, it2 in self.funit().items():
            it2.update()
        for it1, it2 in self.cunit().items():
            it2.update()
        for it1, it2 in self.fgrad().items():
            it2.update()
        for it1, it2 in self.cgrad().items():
            it2.update()
    def forward(self):
        for it1, it2 in self.funit().items():
            it2.forward()
        for it1, it2 in self.cunit().items():
            it2.forward()
        for it1, it2 in self.fgrad().items():
            it2.forward()
        for it1, it2 in self.cgrad().items():
            it2.forward()
    def calc_gradient(self, mesh_: mesh, geom_: geom):
        least_square_gradient(mesh_, geom_, self)
