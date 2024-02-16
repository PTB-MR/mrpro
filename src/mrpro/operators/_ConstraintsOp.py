"""Operator enforcing constraints by variable transformations."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import torch
import torch.nn.functional as F

from mrpro.operators import Operator


class ConstraintsOp(Operator):
    """Transformation to map real-valued tensors to certain ranges."""

    def __init__(
        self,
        bounds: tuple[tuple[float | None, float | None], ...],
        beta_sigmoid: float = 1.0,
        beta_softplus: int = 1,
    ) -> None:
        super().__init__()

        if beta_sigmoid <= 0:
            raise ValueError(f'parameter beta_sigmoid must be greater than zero; given {beta_sigmoid}')
        if beta_softplus <= 0:
            raise ValueError(f'parameter beta_softplus must be greater than zero; given {beta_softplus}')

        self.beta_sigmoid = beta_sigmoid
        self.beta_softplus = beta_softplus

        self.lower_bounds = [bound[0] for bound in bounds]
        self.upper_bounds = [bound[1] for bound in bounds]

        for lb, ub in bounds:
            if (ub is not None and lb is not None) and lb > ub:
                raise ValueError(
                    f'bounds should be ( (a1,b1), (a2,b2), ...) with ai<=bi if neither ai or bi is None;\
                        \nbound tuple {lb,ub} is invalid'
                )

    @staticmethod
    def sigmoid(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Constraint x to be in the range given by 'bounds'."""

        return F.sigmoid(beta * x)

    @staticmethod
    def sigmoid_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Constraint x to be in the range given by 'bounds'."""

        return torch.logit(x) / beta

    @staticmethod
    def softplus(x: torch.Tensor, beta: int = 1) -> torch.Tensor:
        """Constrain x to be in (bound,infty)."""

        return F.softplus(x, beta=beta)

    @staticmethod
    def softplus_inverse(x: torch.Tensor, beta: int = 1) -> torch.Tensor:
        """Inverse of 'softplus_transformation."""

        return beta * x + torch.log(-torch.expm1(-beta * x))

    def forward(self, x: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        # iterate over the tensors and constrain them if necessary according to the
        # chosen bounds
        xc = []
        for i in range(len(self.lower_bounds)):
            lb, ub = self.lower_bounds[i], self.upper_bounds[i]

            # distiguish cases
            if lb is not None and ub is not None:
                # case (a,b) with a<b and a,b \in R
                xc.append(lb + (ub - lb) * self.sigmoid(x[i], beta=self.beta_sigmoid))

            elif lb is not None and ub is None:
                # case (a,None); corresponds to (a, \infty)
                xc.append(lb + self.softplus(x[i], beta=self.beta_softplus))

            elif lb is None and ub is not None:
                # case (None,b); corresponds to (-\infty, b)
                xc.append(ub - self.softplus(-x[i], beta=self.beta_softplus))
            elif lb is None and ub is None:
                # case (None,None); corresponds to (-\infty, \infty), i.e. no transformation
                xc.append(x[i])

        return tuple(xc)

    def inverse(self, xc: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        # iterate over the tensors and constrain them if necessary according to the
        # chosen bounds
        x = []
        for i in range(len(self.lower_bounds)):
            lb, ub = self.lower_bounds[i], self.upper_bounds[i]

            # distiguish cases
            if lb is not None and ub is not None:
                # case (a,b) with a<b and a,b \in R
                x.append(self.sigmoid_inverse((xc[i] - lb) / (ub - lb), beta=self.beta_sigmoid))

            elif lb is not None and ub is None:
                # case (a,None); corresponds to (a, \infty)
                x.append(self.softplus_inverse(xc[i] - lb, beta=self.beta_softplus))

            elif lb is None and ub is not None:
                # case (None,b); corresponds to (-\infty, b)
                x.append(-self.softplus_inverse(-(xc[i] - ub), beta=self.beta_softplus))
            elif lb is None and ub is None:
                # case (None,None); corresponds to (-\infty, \infty), i.e. no transformation
                x.append(xc[i])

        return tuple(x)
