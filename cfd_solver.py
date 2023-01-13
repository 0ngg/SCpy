import numpy as np
import cfd_scheme
import cfd_linear

class solver:
    def __init__(self, what, *args):
        # args what : [] dtype = str, init_file : str, solid_prop_file : str, const_value_file : str, mesh_file : str, 
        # under_relax : double, tol : double, max_iter : int, time_step : float
        self.__user, self.__mesh = cfd_scheme.make_scheme(args[0], args[1], args[2], args[3])
        var_to_linear = dict({"Pcor": "pcorrect", "u": "momentum", "v": "momentum", "w": "momentum",\
                              "k": "turb_k", "e": "turb_e", "T": "energy", "q": "s2s"})
        coor_dict = dict({"u": 0, "v": 1, "w": 2})
        self.__eq = dict({})
        self.__res_loop = dict({})
        self.__res_time = dict({})
        for i in what:
            if var_to_linear[i] == "momentum":
                self.__eq[i] = cfd_linear.__dict__[var_to_linear[i]](self.__mesh, coor_dict[i])
            else:
                self.__eq[i] = cfd_linear.__dict__[var_to_linear[i]](self.__mesh)
            self.__res_loop[i] = []
            self.__res_time[i] = []
        self.__current_time = 0
        self.__under_relax = float(args[4])
        self.__tol = float(args[5])
        self.__max_iter = int(args[6])
        self.__time_step = float(args[7])
    def energys2sloop(self):
        # stop condition -> both equations return rmsr value < tol
        passes = 0
        check = [False, False]
        while all(check) is False:
            # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user
            rmsr_t__ = self.__eq["T"].itersolve(self.__eq["T"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__user, self.__current_time)
            rmsr_q__ = self.__eq["q"].itersolve(self.__eq["q"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__current_time)
            self.__res_loop["T"].extend(rmsr_t__); self.__res_loop["q"].extend(rmsr_q__)
            
            check = [i < self.__tol for i in [rmsr_t__, rmsr_q__]]
            passes += 1
        return passes
    def SIMPLEloop(self):
        # stop condition -> all equations return rmsr value < tol
        passes = 0
        check = [False, False, False, False]
        while all(check) is False:
            # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user
            rmsr_u__ = self.__eq["u"].itersolve(self.__eq["u"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__user, self.__current_time)
            rmsr_v__ = self.__eq["v"].itersolve(self.__eq["u"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__user, self.__current_time)
            rmsr_w__ = self.__eq["w"].itersolve(self.__eq["u"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__user, self.__current_time)
            # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, u :momentum, v : momentum, w : momentum
            rmsr_pcor__ = self.__eq["Pcor"].itersolve(self.__eq["Pcor"], self.__under_relax, self.__tol, self.__max_iter, \
                                        self.__time_step, self.__eq["u"], self.__eq["v"], self.__eq["w"], self.__current_time)
            self.__res_loop["u"].extend(rmsr_u__); self.__res_loop["v"].extend(rmsr_v__); self.__res_loop["w"].extend(rmsr_w__)
            self.__res_loop["Pcor"].extend(rmsr_pcor__)
            check = [i < self.__tol for i in [rmsr_u__, rmsr_v__, rmsr_w__, rmsr_pcor__]]
            passes += 1
        return passes
    def turbloop(self):
        # stop condition -> all equations return rmsr value < tol
        passes = 0
        check = [False, False]
        while all(check) is False:
            # args mesh : mesh, under_relax : double, tol : double, max_iter : int, time_step : float, user : user
            rmsr_k__ = self.__eq["k"].itersolve(self.__eq["k"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__user, self.__current_time)
            rmsr_e__ = self.__eq["e"].itersolve(self.__eq["e"], self.__under_relax, self.__tol, self.__max_iter, self.__time_step, self.__user, self.__current_time)
            self.__res_loop["k"].extend(rmsr_k__); self.__res_loop["e"].extend(rmsr_e__)
            check = [i < self.__tol for i in [rmsr_k__, rmsr_e__]]
            passes += 1
        return passes
    def sctimeloop(self, *args):
        # args max_passes_multip : int
        # stop condition per loop groups -> all loops return rmsr value < tol for a determined number of passes
        # loop 1 = energys2s, loop 2 = simple
        n_loops1 = 10 # energys2s - SIMPLE - turb
        n_loops2 = 10 # SIMPLE - turb
        while n_loops1 > args[0] * 3:
            n_loops1 = 0
            n_loops1 += self.energys2sloop()
            while n_loops2 > args[0] * 2:
                n_loops2 = 0
                n_loops2 += self.SIMPLEloop()
                n_loops2 += self.turbloop()
            n_loops1 += n_loops2
        # append time residuals
        check_loop = []
        for i in self.__res_time.keys():
            current_value = []; prev_value = []
            for j in self.__mesh.cells.keys():
                current_value.extend(self.__mesh.cells[j].value[i][-1])
                prev_value.extend(self.__mesh.cells[j].value[i][-2])
            self.__res_time[i].extend(cfd_linear.linear.calcrmsr(np.array(current_value), np.array(prev_value)))
            check_loop.extend(self.__res_time[i][-1] < self.__tol)
        # forward time step, add empty value/grad, move back props
        for i in self.__mesh.cells.dict():
            for j in self.__mesh.cells[i].value.dict():
                self.__mesh.cells[i].value[j] = np.concatenate((self.__mesh.cells[i].value[j], np.array([self.__mesh.cells[i].value[j][-1]])))
                self.__mesh.cells[i].grad[j] = np.concatenate((self.__mesh.cells[i].grad[j], self.__mesh.cells[i].grad[j][-1]))
                if "rho" in dir(self.__mesh.cells[i].prop):
                    self.__mesh.cells[i].prop.forwardtimeprop()
        for i in self.__mesh.faces.dict():
            for j in self.__mesh.faces[i].value.dict():
                self.__mesh.faces[i].value[j] = np.concatenate((self.__mesh.cells[i].value[j], np.array([self.__mesh.cells[i].value[j][-1]])))
                self.__mesh.faces[i].grad[j] = np.concatenate((self.__mesh.cells[i].grad[j], np.array([self.__mesh.cells[i].grad[j][-1]])))
                if "rho" in dir(self.__mesh.faces[i].prop):
                    self.__mesh.faces[i].prop.forwardtimeprop()
        return all(check_loop)
    def scsteadyloop(self, max_passes_multip : int, max_time_steps = 1000):
        # args max passes_multip : int, max_time_steps : int
        # stop condition if res_time [-1] < tol
        # args max_passes_multip : int
        pass_time = 0
        check_time = False
        while check_time is False:
            check_time = self.sctimeloop(self, max_passes_multip)
            pass_time += 1
            if pass_time >= max_time_steps:
                break
        return

if __name__ == "__main__":
    # args what : np.array([], dtype = str), init_file : str, solid_prop_file : str, const_value_file : str, mesh_file : str, 
    # under_relax : double, tol : double, max_iter : int, time_step : float
    filenames = [x for x in input().split(" ")]
    specnums = [x for x in input().split(" ")]
    eqnames = [x for x in input().split(" ")]
    solvargs = []; solvargs.extend(filenames); solvargs.extend(specnums)
    user_test, mesh_test = cfd_scheme.make_scheme(*filenames)
    solver_test = solver(eqnames, solvargs)
    # solver_test.scsteadyloop(5, 100)