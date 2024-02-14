"""Operator enforcing constraints by variable transformations."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
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

from mrpro.operators import Operator


# TODO: inheriting from Operator throws "_backward_hooks"-error
class ConstraintsOp(Operator):
    """Transformation to map real-valued tensors to certain ranges."""

    def __init__(
        self,
        bounds: tuple[tuple[float | None, float | None], ...],
        beta_sigmoid: float = 1.0,
        beta_softplus: float = 1.0,
    ) -> None:
        super().__init__()
        self.lower_bounds = [bound[0] for bound in bounds]
        self.upper_bounds = [bound[1] for bound in bounds]

        self.beta_sigmoid = beta_sigmoid
        self.beta_softplus = beta_softplus

        for lb, ub in bounds:
            if (ub is not None and lb is not None) and lb > ub:
                raise ValueError(
                    f'bounds should be ( (a1,b1), (a2,b2), ...) with ai<=bi if neither ai or bi is None;\
                        \nbound tuple {lb,ub} is invalid'
                )

    @staticmethod
    def sigmoid_transf(x: torch.Tensor, bounds: tuple[float, float], beta: float = 1.0) -> torch.Tensor:
        """Constraint x to be in the range given by 'bounds'."""

        return bounds[0] + (bounds[1] - bounds[0]) * (1.0 / (1.0 + torch.exp(-beta * x)))

    @staticmethod
    def sigmoid_transf_inv(x: torch.Tensor, bounds: tuple[float, float], beta: float = 1.0) -> torch.Tensor:
        """Inverse of 'sigmoid_transf."""

        return -torch.log(1.0 / ((x - bounds[0]) / (bounds[1] - bounds[0])) - 1.0) / beta

    @staticmethod
    def softplus_transf(x: torch.Tensor, lbound: float, beta: float = 0.1) -> torch.Tensor:
        """Constrain x to be in (bound,infty)."""

        return lbound + 1.0 / beta * torch.log(1 + torch.exp(beta * x))

    @staticmethod
    def softplus_transf_inv(x: torch.Tensor, lbound: float, beta: float = 0.1) -> torch.Tensor:
        """Inverse of 'softplus_transf."""

        return torch.log(torch.exp(beta * (x - lbound)) - 1.0) / beta

    @staticmethod
    def neg_softplus_transf(x: torch.Tensor, ubound: float, beta: float = 0.1) -> torch.Tensor:
        """Contrain x to be in (-infty,bound)."""

        return ubound - 1.0 / beta * torch.log(1 + torch.exp(beta * x))

    @staticmethod
    def neg_softplus_transf_inv(x: torch.Tensor, ubound: float, beta: float = 0.1) -> torch.Tensor:
        """Constrain x to be in (-infty,bound)."""

        return torch.log(torch.exp(beta * (ubound - x)) - 1.0) / beta

    def forward(self, x: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        # iterate over the tensors and constrain them if necessary according to the
        # chosen bounds
        xc = []
        for i in range(len(self.lower_bounds)):
            lb, ub = self.lower_bounds[i], self.upper_bounds[i]

            # distiguish cases
            if lb is not None and ub is not None:
                # case (a,b) with a<b and a,b \in R
                xc.append(self.sigmoid_transf(x[i], bounds=(lb, ub), beta=self.beta_sigmoid))

            elif lb is not None and ub is None:
                # case (a,None); corresponds to (a, \infty)
                xc.append(self.softplus_transf(x[i], lbound=lb, beta=self.beta_softplus))

            elif lb is None and ub is not None:
                # case (None,b); corresponds to (-\infty, b)
                xc.append(self.neg_softplus_transf(x[i], ubound=ub, beta=self.beta_softplus))
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
                x.append(self.sigmoid_transf_inv(xc[i], bounds=(lb, ub), beta=self.beta_sigmoid))

            elif lb is not None and ub is None:
                # case (a,None); corresponds to (a, \infty)
                x.append(self.softplus_transf_inv(xc[i], lbound=lb, beta=self.beta_softplus))

            elif lb is None and ub is not None:
                # case (None,b); corresponds to (-\infty, b)
                x.append(self.neg_softplus_transf_inv(xc[i], ubound=ub, beta=self.beta_softplus))
            elif lb is None and ub is None:
                # case (None,None); corresponds to (-\infty, \infty), i.e. no transformation
                x.append(xc[i])

        return tuple(x)
