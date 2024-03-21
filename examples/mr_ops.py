from mrpro.operators import FourierOp
from mrpro.operators import SensitivityOp
import torch.nn as nn


class EncObj_Reco(nn.Module):

    def __init__(self, kdata, csm):
        super(EncObj_Reco, self).__init__()
    
        self.F = FourierOp(
            recon_matrix=kdata.header.recon_matrix, encoding_matrix=kdata.header.encoding_matrix, traj=kdata.traj
        )
        self.C = SensitivityOp(csm)

    def apply_A(self, x):
        k = (self.F @ self.C)(x) 
        return k[0]

    def apply_AH(self, k):
        x = (self.F @ self.C).adjoint(k)
        return x[0]

    def apply_AHA(self, x):
        k = self.apply_A(x)
        x = self.apply_AH(k)
        return x

    def apply_dcomp(self, k, dcomp):
        return dcomp * k

    def apply_Adag(self, k, dcomp):
        dcomp_k = self.apply_dcomp(k, dcomp)
        x = self.apply_AH(dcomp_k)
        return x

    def apply_AdagA(self, x, dcomp):
        k = self.apply_A(x)
        x = self.apply_Adag(k, dcomp)
        return x