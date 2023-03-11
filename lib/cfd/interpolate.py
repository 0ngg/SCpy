from .scheme import *

# --------
# GRADIENT

def least_square_gradient(mesh_: mesh, geom_: geom, value_: value):
    for it1, it2 in mesh_.cells().items():
        a_neigh_coef = np.zeros(shape=(3,3), dtype = float)
        a_bound_coef = np.zeros(shape=(3,1), dtype = float)
        b_neigh_coef = np.zeros(shape=(3,3), dtype = float)
        b_bound_coef = np.zeros(shape=(3,1), dtype = float)
        neigh_list = [it3[1] for it3 in it2.lcell()]
        bound_list = [it3 for it3 in it2.lface() if it3 not in neigh_list]
        for it3 in it2.lcell():
            w_ = 1 / geom_.dCF().scalar(it1, it3[0])
            dCF_ = geom_.dCF().vec(it1, it3[0])
            for it4 in range(0, 3):
                for it5 in range(0, 3):
                    a_neigh_coef[it4][it5] += w_ * dCF_[it4] * dCF_[it5]
                b_neigh_coef[it4][0] += w_ * dCF_[it4] * (value_.cunit()[it3[0]].current() - value_.cunit()[it1].current())
        for it3 in bound_list:
            w_ = 1 / geom_.dCf().scalar(it1, it3)
            dCf_ = geom_.dCf().vec(it1, it3)
            for it4 in range(0, 3):
                for it5 in range(0, 3):
                    a_bound_coef[it4][it5] += w_ * dCf_[it4] * dCf_[it5]
                b_bound_coef[it4][0] += w_ * dCf_[it4] * (value_.funit()[it3].current() - value_.cunit()[it1].current())
        a_coef = a_neigh_coef + a_bound_coef; b_coef = b_neigh_coef + b_bound_coef
        c, low = linalg.cho_factor(a_coef)
        grad_ = linalg.cho_solve((c, low), b_coef); grad_ = np.array(grad_).ravel()
        value_.cgrad()[it1].new(grad_); value_.cgrad()[it1].update()
    clear_of_interpolation = []
    for it1, it2 in mesh_.cells().items():
        for it3 in it2.lcell():
            if all([it3[1] in clear_of_interpolation]) is False:
                dCf_ = geom_.dCf().scalar(it1, it3[1]); dFf_ = geom_.dCf().scalar(it3[0], it3[1])
                gC_ = dFf_ / (dCf_ + dFf_)
                dCF_ = geom_.dCF().scalar(it1, it3[0])
                eCF_ = geom_.dCF().norm(it1, it3[0])
                gradC_ = value_.cgrad()[it1].current()
                gradF_ = value_.cgrad()[it1].current()
                gradfCF_ = (gC_ * gradC_) + ((1 - gC_) * gradF_)
                gradf_ = gradfCF_ + ( ((value_.cunit()[it3[0].current() - value_.cunit()[it1].current()]) / dCF_)
                                     - (np.dot(gradfCF_, eCF_))) * eCF_
                value_.fgrad()[it3[1]].new(gradf_); value_.fgrad()[it3[1]].update()
                clear_of_interpolation.append(it3[1])
        neigh_list = [it3[1] for it3 in it2.lcell()]
        bound_list = [it3 for it3 in it2.lface() if it3 not in neigh_list]
        for it3 in bound_list:
            gradC_ = value_.cgrad()[it1].current()
            eCf_ = geom_.dCf().norm(it1, it3)
            gradf_ = (np.dot(gradC_, eCf_)) * eCf_
            value_.fgrad()[it3].new(gradf_); value_.fgrad()[it3].update()