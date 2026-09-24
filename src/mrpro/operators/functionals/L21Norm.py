"""L2,1 Norm."""

from collections.abc import Sequence

import torch

from mrpro.operators.Functional import ElementaryProximableFunctional, throw_if_negative_or_complex


class L21Norm(ElementaryProximableFunctional):
    r"""Functional class for the (weighted, isotropic) L2,1 Norm.

    This implements the functional given by
    :math:`f: C^N \rightarrow [0, \infty), x \rightarrow \sum_{i=1}^N \sqrt{w_i} \, \| (x-b)_i \|_2`,

    where the inner L2-norm is taken along the dimensions given by `kernel_dim` at
    initialization (the "grouping" dimensions, e.g. the components of a vector field such
    as a gradient), and the outer sum (or mean, if `divide_by_n` is `True`) is taken along
    the dimensions given by `dim`, combining the resulting group-norms.

    This class only supports the case where the weight `W` is scalar *per group*, i.e.
    constant across the dimensions in `kernel_dim` (it may still vary across all other
    dimensions and/or be a single global scalar). This is the common case for edge-weighted,
    isotropic Total Variation, where :math:`w_i` weights the whole gradient vector at voxel
    :math:`i` uniformly across all spatial directions.

    If `weight` varies along `kernel_dim` (i.e. depends on the within-group index, such as
    a per-direction weight in a weighted TV), the closed-form prox implemented here does not
    apply, and a `ValueError` is raised.

    In most cases, consider setting `divide_by_n` to `true` to be independent of input size.
    """

    def __init__(
        self,
        target: torch.Tensor | complex | None = None,
        weight: torch.Tensor | complex = 1.0,
        dim: int | Sequence[int] | None = None,
        kernel_dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize L21Norm.

        We assume that the functional is given in the form
        :math:`f(x) = \sum_i \sqrt{w_i} \| (\mathrm{weight}(x - \mathrm{target}))_i \|_2`,
        where the inner L2-norm groups elements along `kernel_dim`, and the outer sum
        (or mean, if `divide_by_n`) is taken along `dim`.

        Parameters
        ----------
        target
            target element - often data tensor (see above)
        weight
            weight parameter (see above). Must be constant (broadcastable, size 1)
            along `kernel_dim` (Fall A); a weight that genuinely varies along
            `kernel_dim` is not supported and raises a `ValueError` when the
            operator is applied.
        dim
            dimension(s) over which the functional is reduced (summed or averaged).
            All other dimensions of `weight (x - target)` will be treated as batch
            dimensions. Must be disjoint from `kernel_dim`.
        kernel_dim
            dimension(s) over which the inner L2-norm is computed (the grouping
            dimension(s), e.g. the components of a vector field such as a gradient).
            Required; must be disjoint from `dim`.
        divide_by_n
            if true, the result is scaled by the number of elements of the dimensions
            indexed by `dim` in the tensor `weight (x - target)`. If true, the outer
            reduction is thus calculated as the mean, else the sum.
        keepdim
            if true, the dimension(s) of the input indexed by `dim` and `kernel_dim`
            are maintained and collapsed to singletons, else they are removed from
            the result.
        """
        super().__init__(target=target, weight=weight, dim=dim, divide_by_n=divide_by_n, keepdim=keepdim)

        if kernel_dim is None:
            raise ValueError('L21Norm requires `kernel_dim` to be set (the L2-grouping dimension(s)).')
        if isinstance(kernel_dim, int):
            kernel_dim = (kernel_dim,)
        else:
            kernel_dim = tuple(kernel_dim)
        if self.dim is not None and set(self.dim) & set(kernel_dim):
            raise ValueError(
                f'`dim` ({self.dim}) and `kernel_dim` ({kernel_dim}) must not overlap; '
                'they index disjoint reductions (outer L1/mean vs. inner L2).'
            )
        self.kernel_dim = kernel_dim

    def _check_weight_constant_along_kernel_dim(self, x_shape: torch.Size) -> None:
        """Ensure `weight` is constant along the group dimension(s) `self.kernel_dim`.

        This closed-form implementation only supports a scalar weight per group
        (Fall A). A weight that varies along `kernel_dim` (Fall B, e.g. a genuinely
        per-direction weight) would require a different prox (no closed form; an
        iterative ellipsoid projection) and is therefore rejected here.

        Parameters
        ----------
        x_shape
            Shape of the input tensor the operator is applied to. `weight` is
            right-aligned to this shape following standard broadcasting rules.

        Raises
        ------
        ValueError
            If `weight` has a size greater than 1 along any dimension in `kernel_dim`.
        """
        n = len(x_shape)
        kernel_dim_normalized = {d % n for d in self.kernel_dim}

        weight_ndim = self.weight.ndim
        offset = n - weight_ndim  # right-alignment offset for broadcasting

        for d in kernel_dim_normalized:
            w_dim = d - offset
            if 0 <= w_dim < weight_ndim and self.weight.shape[w_dim] > 1:
                raise ValueError(
                    'L21Norm (Fall A) requires `weight` to be constant (broadcastable, '
                    f'size 1) along kernel_dim {tuple(sorted(kernel_dim_normalized))}. '
                    f'Got weight.shape={tuple(self.weight.shape)} for input shape '
                    f'{tuple(x_shape)}. A weight that genuinely varies along kernel_dim '
                    '(Fall B, e.g. per-direction weighting) is not supported by this '
                    'closed-form implementation.'
                )

    def __call__(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Compute the L2,1 norm of the input tensor.

        Calculates :math:`\\sum_i \\sqrt{w_i} \\| (x - b)_i \\|_2`, where :math:`w` is `weight`
        and :math:`b` is `target`. The inner L2-norm is computed along `kernel_dim`; the outer
        sum (or mean, if `divide_by_n`) is computed along `dim`.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
            The L2,1 norm. If `keepdim` is true, the reduced dimensions (`dim` and
            `kernel_dim`) are retained with size 1; otherwise, they are removed.

        Raises
        ------
        ValueError
            If `weight` varies along `kernel_dim` (Fall B).
        """
        return super().__call__(x)
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Apply forward of L21Norm.

        .. note::
            Prefer calling the instance of the L21Norm as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        self._check_weight_constant_along_kernel_dim(x.shape)

        diff = x - self.target
        group_norm = torch.linalg.vector_norm(diff, ord=2, dim=self.kernel_dim, keepdim=True)

        
        weight = self.weight.clamp_min(0).sqrt()
        value = weight * group_norm

        l1_dim = self.dim if self.dim is not None else tuple(
            d for d in range(value.ndim) if d not in {k % value.ndim for k in self.kernel_dim}
        )

        if self.divide_by_n:
            value = torch.mean(value, dim=l1_dim, keepdim=True)
        else:
            value = torch.sum(value, dim=l1_dim, keepdim=True)

        if not self.keepdim:
            kernel_dim_normalized = {k % value.ndim for k in self.kernel_dim}
            l1_dim_normalized = {ld % value.ndim for ld in l1_dim}
            value = value.squeeze(dim=tuple(kernel_dim_normalized | l1_dim_normalized))
        return (value,)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal Mapping of the L2,1 Norm.

        Computes the proximal mapping of the (weighted, isotropic) L2,1 norm, i.e. group
        soft-thresholding: each group of elements along `self.kernel_dim` is shrunk
        isotropically (direction preserved) towards zero by an amount depending on
        `weight` and `sigma`.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal mapping applied to the input tensor

        Raises
        ------
        ValueError
            If `weight` varies along `kernel_dim` (Fall B).
        """
        throw_if_negative_or_complex(sigma)
        self._check_weight_constant_along_kernel_dim(x.shape)

        diff = x - self.target

        alpha = self.weight.clamp_min(0).sqrt() * sigma
        alpha = self._divide_by_n(alpha, torch.broadcast_shapes(diff.shape, alpha.shape))

        norm = torch.linalg.vector_norm(diff, ord=2, dim=self.kernel_dim, keepdim=True)
        scale = torch.clamp(1 - alpha / norm.clamp_min(torch.finfo(norm.dtype).tiny), min=0)

        x_out = diff * scale + self.target
        x_out = x_out.to(torch.result_type(alpha, x_out))
        return (x_out,)

    def prox_convex_conj(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor | float = 1.0,
    ) -> tuple[torch.Tensor]:
        """Convex conjugate of the L2,1 Norm.

        Compute the proximal mapping of the convex conjugate of the L2,1 norm, i.e. a
        group-wise (radial) projection onto the (weighted) L2-ball of radius `sqrt(weight)`,
        with groups formed along `self.kernel_dim`.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal of the convex conjugate applied to the input tensor

        Raises
        ------
        ValueError
            If `weight` varies along `kernel_dim` (Fall B).
        """
        throw_if_negative_or_complex(sigma)
        self._check_weight_constant_along_kernel_dim(x.shape)

        diff = x - sigma * self.target

        alpha = self.weight.clamp_min(0).sqrt()
        alpha = self._divide_by_n(alpha, torch.broadcast_shapes(diff.shape, alpha.shape))

        norm = torch.linalg.vector_norm(diff, ord=2, dim=self.kernel_dim, keepdim=True)
        scale = torch.clamp_max(alpha.abs() / norm.clamp_min(torch.finfo(norm.dtype).tiny), 1.0)

        x_out = diff * scale
        x_out = x_out.to(torch.result_type(alpha, x_out))
        return (x_out,)