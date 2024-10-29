import math
import logging
from typing import Callable, List

import torch
# https://github.com/kylematoba/freshstart/blob/main/test_hessian_comp.py

import utils.tensor

torch.set_printoptions(linewidth=2000, threshold=10000)
torch.manual_seed(11)

device = torch.device("cpu")
dtype = torch.float32

torch_inf = torch.tensor(float("Inf"))
torch_nan = torch.tensor(float("nan"))

logging_format = "%(asctime)s: %(message)s"
logging_level = 15
logging.basicConfig(level=logging_level,
                    format=logging_format)

logger = logging.getLogger(__name__)
init_beta = 1.1


def interleave_lists(list1: list,
                     list2: list) -> list:
    # https://stackoverflow.com/questions/7946798/interleave-multiple-lists-of-the-same-length-in-python
    interleaved_list = [val for pair in zip(list1, list2) for val in pair]
    return interleaved_list


def test_interleave_lists():
    list1 = [1, 3, 5, 7]
    list2 = [2, 4, 6, 8]
    assert [1, 2, 3, 4, 5, 6, 7, 8] == interleave_lists(list1, list2)


def build_softplus_model(layer_sizes: List[int]) -> torch.nn.Module:
    bias = True
    num_layers = len(layer_sizes)
    linear_layers = [torch.nn.Linear(layer_sizes[idx - 1],
                                     layer_sizes[idx],
                                     bias=bias) for idx in range(1, num_layers)]
    layers = _build_softplus_layer_list(linear_layers)
    model = torch.nn.Sequential(*layers)
    return model


def _build_softplus_layer_list(linear_layers: List[torch.nn.Module]) -> torch.nn.Module:
    half_network_len = len(linear_layers) - 1
    sp = torch.nn.Softplus()
    corresp_sp = [sp] * half_network_len

    list1 = corresp_sp
    list2 = linear_layers[1:]
    network = [linear_layers[0]] + interleave_lists(list1, list2)  # + [for ll in linear_layers[1:]]
    return network


def test_hessian_dim1():
    n = 10
    fun = lambda _: torch.exp(_ ** 2) / 2
    x = torch.randn(1, n)
    h1 = vectorized_hessian(fun, x).squeeze()
    vals = (1 + 2 * x ** 2) * torch.exp(x ** 2)
    h2 = torch.diag_embed(torch.diag(vals.flatten()))
    torch.testing.assert_close(h1, h2)


def bfun(before: torch.Tensor,
         center: torch.Tensor) -> torch.Tensor:
    es3d_2d = "ijk,kl->ijl"
    es2d_3d = "li,ijk->ljk"

    assert before.shape[1] == center.shape[0]
    assert before.shape[1] == center.shape[-1]

    val0 = torch.einsum(es2d_3d, before, center)
    val1 = torch.einsum(es3d_2d, val0, before.T)
    return val1


def vectorized_hessian(f: Callable, x: torch.Tensor) -> torch.Tensor:
    fx = f(x)
    rout = fx.numel()
    outs = [torch.autograd.functional.hessian(lambda x: f(x).flatten()[idx], x) for idx in range(rout)]
    vh = torch.stack(outs, 2)
    return vh


def dsoftplus(x: torch.Tensor) -> torch.Tensor:
    sp = torch.nn.Softplus()
    beta = sp.beta
    dsp_flat = 1 / (1 + torch.exp(-beta * x)).flatten()
    dsp = torch.diag(dsp_flat)
    return dsp


def ddsoftplus(x: torch.Tensor) -> torch.Tensor:
    sp = torch.nn.Softplus()
    beta = sp.beta

    dsp_flat = (beta * torch.exp(-beta * x) / (1 + torch.exp(-beta * x)) ** 2).flatten()
    dsp = utils.tensor.diag1d_to_diag3d(dsp_flat)
    assert not torch.isnan(dsp).any()
    return dsp


def test_first_second_derivative_relationship():
    n0 = 10
    sp = torch.nn.Softplus()
    beta = sp.beta

    x = torch.randn(1, n0)
    ddx = ddsoftplus(x)
    dx = dsoftplus(x)
    adj1 = (beta * torch.exp(-beta * x) / (1 + torch.exp(-beta * x))).flatten()
    adj2 = (beta) / (1 + torch.exp(beta * x)).flatten()
    torch.testing.assert_close(adj1, adj2)

    adj = adj1
    assert (adj <= beta).all()
    gg = utils.tensor.diag1d_to_diag3d(dx.diag() * adj)

    torch.testing.assert_close(gg, ddx)


def test_gradient_comp_nonlinear1():
    n0 = 20
    n1 = 15
    n2 = 10
    n3 = 5

    bias = True
    lin1 = torch.nn.Linear(n0, n1, bias=bias)
    lin2 = torch.nn.Linear(n1, n2, bias=bias)
    lin3 = torch.nn.Linear(n2, n3, bias=bias)

    sp = torch.nn.Softplus()
    nonlin_net = torch.nn.Sequential(lin1, sp, lin2, sp, lin3)

    layer = nonlin_net
    x = torch.randn(1, n0)
    dsp1 = dsoftplus(nonlin_net[:1](x))
    dsp2 = dsoftplus(nonlin_net[:3](x))

    g1 = torch.autograd.functional.jacobian(layer, x).squeeze()
    g2 = lin3.weight @ dsp2 @ lin2.weight @ dsp1 @ lin1.weight
    torch.testing.assert_close(g1, g2)


def test_dsp1():
    n0 = 15
    sp = torch.nn.Softplus()
    x = torch.randn(n0,)

    dsp1 = dsoftplus(x)
    dsp2 = torch.autograd.functional.jacobian(sp, x).squeeze()

    torch.testing.assert_close(dsp1, dsp2, rtol=0.0, atol=5e-4)


def test_ddsp1():
    n0 = 1
    sp = torch.nn.Softplus()
    x = torch.randn(n0,)

    ddsp1 = ddsoftplus(x)
    ddsp2 = torch.atleast_3d(torch.autograd.functional.hessian(sp, x))
    torch.testing.assert_close(ddsp1, ddsp2, rtol=0.0, atol=5e-4)


def test_ddsp2():
    n0 = 5
    sp = torch.nn.Softplus()
    model = lambda x: sp(x).sum()
    x = torch.randn(n0,)

    dsp1 = dsoftplus(x).sum(1)
    dsp2 = torch.autograd.functional.jacobian(model, x).squeeze()
    torch.testing.assert_close(dsp1, dsp2, rtol=0.0, atol=5e-4)


def test_ddsp3():
    n0 = 5
    sp = torch.nn.Softplus()
    x = torch.randn(n0,)

    ddsp1 = ddsoftplus(x)
    ddsp2 = vectorized_hessian(sp, x)

    torch.testing.assert_close(ddsp1, ddsp2, rtol=0.0, atol=.001)


def test_ddsp4():
    n0 = 10

    sp = torch.nn.Softplus()
    x = torch.randn(n0,)

    ddsp1 = ddsoftplus(x)
    ddsp2 = vectorized_hessian(sp, x)
    torch.testing.assert_close(ddsp1, ddsp2, rtol=0.0, atol=.001)


def test_hessian_nonlinear1():
    n0 = 20
    n1 = 15
    n2 = 1

    bias = True
    lin1 = torch.nn.Linear(n0, n1, bias=bias)
    lin2 = torch.nn.Linear(n1, n2, bias=bias)

    sp = torch.nn.Softplus()
    nonlin_net = torch.nn.Sequential(lin1, sp, lin2)

    x = torch.randn(1, n0)
    h1 = vectorized_hessian(nonlin_net, x).squeeze()

    w1 = lin1.weight
    w2 = lin2.weight
    ddsp1 = ddsoftplus(nonlin_net[:1](x))

    h2 = (w1.T @ ((ddsp1 @ w2.T).squeeze() @ w1))
    torch.testing.assert_close(h1, h2)


def test_hessian_nonlinear2():
    n0 = 20
    n1 = 15
    n2 = 5
    # n3 = 1

    # bias = False just to reconcile to the collapsed layer
    bias = True
    lin1 = torch.nn.Linear(n0, n1, bias=bias)
    lin2 = torch.nn.Linear(n1, n2, bias=bias)

    sp = torch.nn.Softplus()
    nonlin_net = torch.nn.Sequential(lin1, sp, lin2)

    x = torch.randn(1, n0)
    h1 = vectorized_hessian(nonlin_net, x).squeeze()

    w1 = lin1.weight
    w2 = lin2.weight
    ddsp1 = ddsoftplus(nonlin_net[:1](x))

    val0 = torch.einsum("ijk,lk->ijl", ddsp1, w2)
    val1 = torch.einsum("ji,jkl->ijl", w1, val0)
    val2 = torch.einsum("ijk,jl->ikl", val1, w1)
    h2 = val2
    torch.testing.assert_close(h1, h2)


def test_hessian_nonlinear3():
    n0 = 20
    n1 = 15
    n2 = 5

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        bias = True
        lin1 = torch.nn.Linear(n0, n1, bias=bias)
        lin2 = torch.nn.Linear(n1, n2, bias=bias)

        sp = torch.nn.Softplus()
        nonlin_net = torch.nn.Sequential(lin1, sp, lin2)

        x = torch.randn(1, n0)
        h1 = vectorized_hessian(nonlin_net, x).squeeze()

        w1 = lin1.weight
        w2 = lin2.weight

        ddsp1 = ddsoftplus(nonlin_net[:1](x))

        val0 = torch.einsum("ijk,lk->ijl", ddsp1, w2)
        val1 = torch.einsum("ijk,il->ljk", val0, w1)
        val2 = torch.einsum("ijk,jl->ikl", val1, w1)
        torch.testing.assert_close(h1, val2, **test_args)

        es3d_2d = "ijk,kl->ijl"
        es2d_3d = "li,ijk->ljk"
        after = w2.T
        before = w1.T
        val0 = torch.einsum(es3d_2d, ddsp1, after)
        val1 = torch.einsum(es2d_3d, before, val0)
        val2 = val1.swapaxes(2, 1)
        val3 = torch.einsum(es3d_2d, val2, before.T)
        torch.testing.assert_close(h1, val3, **test_args)

        es3d_2d = "ijk,kl->ijl"
        es2d_3d = "li,ijk->ljk"
        after = w2.T
        before = w1.T

        val0 = torch.einsum(es2d_3d, before, ddsp1)
        val1 = torch.einsum(es3d_2d, val0, after)
        val2 = val1.swapaxes(2, 1)
        val3 = torch.einsum(es3d_2d, val2, before.T)
        torch.testing.assert_close(h1, val3, **test_args)

        es3d_2d = "ijk,kl->ijl"
        es2d_3d = "li,ijk->ljk"
        after = w2.T
        before = w1.T

        val0 = torch.einsum(es2d_3d, before, ddsp1)
        val2 = torch.einsum(es3d_2d, val0, before.T)
        val3 = torch.einsum("ijk,jl->ilk", val2, after)
        torch.testing.assert_close(h1, val3, **test_args)

        val3 = utils.tensor.reduce(ddsp1, before, after)
        torch.testing.assert_close(h1, val3, **test_args)

        dsp = dsoftplus(nonlin_net[:1](x))
        deriv_seq = [lin2.weight, dsp, lin1.weight]

        after_ = prod_list(deriv_seq[:1]).T
        before_ = prod_list(deriv_seq[2:]).T
        torch.testing.assert_close(after, after_, **test_args)
        torch.testing.assert_close(before, before_, **test_args)


def test_hessian_nonlinear4():
    # three linear layers, but just one nonlinearity, to help line up the dims
    n0 = 20
    n1 = 15
    n2 = 10
    n3 = 1

    bias = True

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        lin1 = torch.nn.Linear(n0, n1, bias=bias)
        lin2 = torch.nn.Linear(n1, n2, bias=bias)
        lin3 = torch.nn.Linear(n2, n3, bias=bias)

        sp = torch.nn.Softplus()
        nonlin_net = torch.nn.Sequential(lin1, sp, lin2, lin3)

        x = torch.randn(1, n0)
        h1 = vectorized_hessian(nonlin_net, x).squeeze()

        w1 = lin1.weight
        w2 = lin2.weight
        w3 = lin3.weight

        ddsp1 = ddsoftplus(nonlin_net[:1](x))

        w32 = w3 @ w2
        val0 = torch.einsum("ijk,lk->ijl", ddsp1, w32)
        val1 = torch.einsum("ijk,il->ljk", val0, w1)
        val2 = torch.einsum("ijk,jl->ikl", val1, w1)
        h2 = val2
        torch.testing.assert_close(h1, h2.squeeze(), **test_args)


def prod_list(a: List[torch.Tensor],
              mode: str = "right") -> torch.Tensor:
    aa = a[0]
    for idx in range(1, len(a)):
        if mode == "right":
            aa = aa @ a[idx]
        elif mode == "left":
            aa = a[idx] @ aa
        else:
            raise ValueError(f"Do not know mode {mode}")
    return aa


def test_hessian_nonlinear5():
    n0 = 20
    n1 = 15
    n2 = 1

    bias = True
    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        lin1 = torch.nn.Linear(n0, n1, bias=bias)
        lin2 = torch.nn.Linear(n1, n2, bias=bias)

        sp = torch.nn.Softplus()
        nonlin_net = torch.nn.Sequential(lin1, sp, sp, lin2)

        x = torch.randn(1, n0)
        h1 = vectorized_hessian(nonlin_net, x).squeeze()

        ddsp1 = ddsoftplus(nonlin_net[:1](x))
        ddsp2 = ddsoftplus(nonlin_net[:2](x))

        dsp1 = dsoftplus(nonlin_net[:1](x))
        dsp2 = dsoftplus(nonlin_net[:2](x))

        g1 = torch.autograd.functional.jacobian(nonlin_net, x).squeeze()
        g2 = lin2.weight @ dsp2 @ dsp1 @ lin1.weight
        torch.testing.assert_close(g1.flatten(), g2.flatten(), **test_args)

        deriv_seq = [lin2.weight, dsp2, dsp1, lin1.weight]
        g3 = prod_list(deriv_seq, "right")
        torch.testing.assert_close(g1.flatten(), g3.flatten(), **test_args)
        idx1 = 1
        idx2 = 2

        after2 = prod_list(deriv_seq[:idx1], "right").T
        before2 = prod_list(deriv_seq[idx1+1:], "right").T

        after1 = prod_list(deriv_seq[:idx2], "right").T
        before1 = prod_list(deriv_seq[idx2+1:], "right").T

        vv1 = utils.tensor.reduce(ddsp1, before1, after1)
        vv2 = utils.tensor.reduce(ddsp2, before2, after2)
        vv = vv1 + vv2
        torch.testing.assert_close(vv.squeeze(), h1, **test_args)

        flattened = ddsp1.reshape(n1, -1)
        v1 = before1 @ flattened
        v2 = v1.reshape(-1, n1)
        v3 = v2 @ before1.T
        v4 = v3.reshape(n0, -1, n0)
        v5 = torch.einsum("ijk,jl->ilk", v4, after1)
        torch.testing.assert_close(vv1, v5, **test_args)


def test_hessian_nonlinear6():
    n0 = 20
    n1 = 15
    n2 = 10
    n3 = 5

    bias = True

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)
    for idx in range(10):
        lin1 = torch.nn.Linear(n0, n1, bias=bias)
        lin2 = torch.nn.Linear(n1, n2, bias=bias)
        lin3 = torch.nn.Linear(n2, n3, bias=bias)

        sp = torch.nn.Softplus()
        model = torch.nn.Sequential(lin1, sp, lin2, sp, lin3)

        x = torch.randn(1, n0)
        h1 = vectorized_hessian(model, x).squeeze()

        idx1 = 1
        idx2 = 3

        ddsp1 = ddsoftplus(model[:idx1](x))
        ddsp2 = ddsoftplus(model[:idx2](x))

        dsp1 = dsoftplus(model[:idx1](x))
        dsp2 = dsoftplus(model[:idx2](x))

        deriv_seq_rev = [lin3.weight, dsp2, lin2.weight, dsp1, lin1.weight]

        after2 = prod_list(deriv_seq_rev[:idx1], "right").T
        before2 = prod_list(deriv_seq_rev[idx1+1:], "right").T

        after1 = prod_list(deriv_seq_rev[:idx2], "right").T
        before1 = prod_list(deriv_seq_rev[idx2+1:], "right").T

        hess1 = utils.tensor.reduce(ddsp1, before1, after1)
        hess2 = utils.tensor.reduce(ddsp2, before2, after2)
        hess = (hess1 + hess2).squeeze()
        torch.testing.assert_close(h1, hess, **test_args)


def test_hessian_nonlinear7():
    n0 = 20
    n1 = 15
    n2 = 10
    n3 = 5

    bias = True

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):

        lin1 = torch.nn.Linear(n0, n1, bias=bias)
        lin2 = torch.nn.Linear(n1, n2, bias=bias)
        lin3 = torch.nn.Linear(n2, n3, bias=bias)

        assert lin1.weight.shape == (n1, n0)
        assert lin2.weight.shape == (n2, n1)
        assert lin3.weight.shape == (n3, n2)

        linear_layers = [lin1, lin2, lin3]
        layers = _build_softplus_layer_list(linear_layers)
        model = torch.nn.Sequential(*layers)

        x = torch.randn(1, n0)

        num_layers = len(layers)
        assert (torch.autograd.functional.jacobian(model[:1], x).squeeze() == lin1.weight).all()
        derivs = [None] * num_layers
        for idx in range(num_layers):
            # idx = 1
            # idx = 3
            if 0 == idx:
                layer_x = x
            else:
                layer_x = model[:idx](x)
            derivs[idx] = torch.autograd.functional.jacobian(model[idx], layer_x).squeeze()
        dsp1 = dsoftplus(model[:1](x))
        dsp2 = dsoftplus(model[:3](x))
        derivs_raw = [lin1.weight, dsp1, lin2.weight, dsp2, lin3.weight]
        norms = [(torch.norm(_1 - _2)) for _1, _2 in zip(derivs, derivs_raw)]
        # thresh = 1e-4
        thresh = 1e-3
        tf = [_ < thresh for _ in norms]
        if False:
            print(torch.norm(derivs[1] - derivs_raw[1]) < 5e-7)
            print(torch.norm(derivs[3] - derivs_raw[3]) < 5e-7)
        assert all(tf)


def compute_derivs(model: torch.nn.Sequential,
                   x: torch.Tensor) -> List[torch.Tensor]:
    num_layers = len(model)
    grad1 = torch.autograd.functional.jacobian(model[:1], x).squeeze()
    assert (grad1 == model[0].weight).all()
    derivs = [None] * num_layers
    for idx in range(num_layers):
        # idx = 1
        # idx = 3
        if 0 == idx:
            layer_x = x
        else:
            layer_x = model[:idx](x)
        derivs[idx] = torch.autograd.functional.jacobian(model[idx], layer_x).squeeze()
    return derivs


def test_hessian_nonlinear8():
    n0 = 20
    n1 = 15
    n2 = 10
    n3 = 5

    layer_sizes = [n0, n1, n2, n3]

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        model = build_softplus_model(layer_sizes)

        x = torch.randn(1, n0)
        derivs = compute_derivs(model, x)
        derivs_reversed = list(reversed(derivs))

        indices = [idx for idx, _ in enumerate(model)
                   if type(_) == torch.nn.Softplus]
        idx1 = indices[0]
        idx2 = indices[1]

        ddsp1 = ddsoftplus(model[:idx1](x))
        ddsp2 = ddsoftplus(model[:idx2](x))

        after2 = prod_list(derivs_reversed[:idx1], "right").T
        before2 = prod_list(derivs_reversed[idx1+1:], "right").T

        after1 = prod_list(derivs_reversed[:idx2], "right").T
        before1 = prod_list(derivs_reversed[idx2+1:], "right").T

        hess1 = utils.tensor.reduce(ddsp1, before1, after1)
        hess2 = utils.tensor.reduce(ddsp2, before2, after2)

        hess = (hess1 + hess2).squeeze()

        h1 = vectorized_hessian(model, x).squeeze()
        torch.testing.assert_close(h1, hess, **test_args)


def test_hessian_nonlinear9():
    n0 = 25
    n1 = 20
    n2 = 15
    n3 = 10
    n4 = 5

    layer_sizes = [n0, n1, n2, n3, n4]

    idx1 = 1
    idx2 = 3
    idx3 = 5

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        model = build_softplus_model(layer_sizes)
        x = torch.randn(1, layer_sizes[0])
        h1 = vectorized_hessian(model, x).squeeze()

        ddsp1 = ddsoftplus(model[:idx1](x))
        ddsp2 = ddsoftplus(model[:idx2](x))
        ddsp3 = ddsoftplus(model[:idx3](x))

        dsp1 = dsoftplus(model[:idx1](x))
        dsp2 = dsoftplus(model[:idx2](x))
        dsp3 = dsoftplus(model[:idx3](x))

        deriv_seq = [model[0].weight, dsp1, model[2].weight, dsp2, model[4].weight, dsp3, model[6].weight]
        # deriv_seq = [model[0].weight, dsp1, model[2].weight, dsp2, model[4].weight]
        deriv_seq_rev = list(reversed(deriv_seq))

        after1 = prod_list(deriv_seq_rev[:idx3], "right").T
        before1 = prod_list(deriv_seq_rev[idx3+1:], "right").T

        after2 = prod_list(deriv_seq_rev[:idx2], "right").T
        before2 = prod_list(deriv_seq_rev[idx2+1:], "right").T

        after3 = prod_list(deriv_seq_rev[:idx1], "right").T
        before3 = prod_list(deriv_seq_rev[idx1+1:], "right").T

        hess1 = utils.tensor.reduce(ddsp1, before1, after1)
        hess2 = utils.tensor.reduce(ddsp2, before2, after2)
        hess3 = utils.tensor.reduce(ddsp3, before3, after3)

        hess = (hess1 + hess2 + hess3).squeeze()
        torch.testing.assert_close(h1, hess, **test_args)


def test_hessian_nonlinear10():
    n0 = 30
    n1 = 25
    n2 = 20
    n3 = 15
    n4 = 10
    n5 = 5

    layer_sizes = [n0, n1, n2, n3, n4, n5]

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)
    for idx in range(10):

        model = build_softplus_model(layer_sizes)

        x = torch.randn(1, layer_sizes[0])
        h1 = vectorized_hessian(model, x).squeeze()

        indices = [idx for idx, _ in enumerate(model)
                   if type(_) == torch.nn.Softplus]

        idx1 = indices[0]
        idx2 = indices[1]
        idx3 = indices[2]
        idx4 = indices[3]

        ddsp1 = ddsoftplus(model[:idx1](x))
        ddsp2 = ddsoftplus(model[:idx2](x))
        ddsp3 = ddsoftplus(model[:idx3](x))
        ddsp4 = ddsoftplus(model[:idx4](x))

        dsp1 = dsoftplus(model[:idx1](x))
        dsp2 = dsoftplus(model[:idx2](x))
        dsp3 = dsoftplus(model[:idx3](x))
        dsp4 = dsoftplus(model[:idx4](x))

        deriv_seq = [model[0].weight, dsp1,
                     model[2].weight, dsp2,
                     model[4].weight, dsp3,
                     model[6].weight, dsp4,
                     model[8].weight]
        deriv_seq_rev = list(reversed(deriv_seq))

        after1 = prod_list(deriv_seq_rev[:indices[-1]], "right").T
        before1 = prod_list(deriv_seq_rev[indices[-1]+1:], "right").T

        after2 = prod_list(deriv_seq_rev[:indices[-2]], "right").T
        before2 = prod_list(deriv_seq_rev[indices[-2]+1:], "right").T

        after3 = prod_list(deriv_seq_rev[:indices[-3]], "right").T
        before3 = prod_list(deriv_seq_rev[indices[-3]+1:], "right").T

        after4 = prod_list(deriv_seq_rev[:indices[-4]], "right").T
        before4 = prod_list(deriv_seq_rev[indices[-4]+1:], "right").T

        hess1 = utils.tensor.reduce(ddsp1, before1, after1)
        hess2 = utils.tensor.reduce(ddsp2, before2, after2)
        hess3 = utils.tensor.reduce(ddsp3, before3, after3)
        hess4 = utils.tensor.reduce(ddsp4, before4, after4)

        hess = (hess1 + hess2 + hess3 + hess4).squeeze()
        torch.testing.assert_close(h1, hess, **test_args)


def test_hessian_nonlinear11():
    layer_sizes = [30, 25, 20, 15, 10, 5]
    atol = 1e-4
    for idx in range(10):
        model = build_softplus_model(layer_sizes)
        num_layers = len(model)

        nonlinear_indices = [idx for idx, _ in enumerate(model)
                   if type(_) == torch.nn.Softplus]
        linear_indices = sorted(set(range(num_layers)) - set(nonlinear_indices))
        num_nonlinearities = len(nonlinear_indices)

        x = torch.randn(1, layer_sizes[0])
        ddsps = [ddsoftplus(model[:i](x)) for i in nonlinear_indices]
        dsps = [dsoftplus(model[:i](x)) for i in nonlinear_indices]
        weights = [model[_].weight for _ in linear_indices]
        deriv_seq = interleave_lists(weights, dsps) + [weights[-1]]
        deriv_seq_rev = list(reversed(deriv_seq))

        model_g = torch.autograd.functional.jacobian(model, x).squeeze()
        hess_terms = [None] * num_nonlinearities

        for idx in range(num_nonlinearities):
            # idx = 0
            after = prod_list(deriv_seq_rev[:nonlinear_indices[-(idx + 1)]], "right").T
            before = prod_list(deriv_seq_rev[nonlinear_indices[-(idx + 1)]+1:], "right").T
            hess_terms[idx] = utils.tensor.reduce(ddsps[idx], before, after)

            torch.testing.assert_close((before @ dsps[idx] @ after).T, model_g, rtol=0.0, atol=atol)
            x_at = model[:nonlinear_indices[idx]+1](x)
            fun_at = model[nonlinear_indices[idx]+1:]
            g_at = torch.autograd.functional.jacobian(fun_at, x_at).squeeze().T

            torch.testing.assert_close(after, g_at, rtol=0.0, atol=atol)

        actual = sum(hess_terms).squeeze()
        expected = vectorized_hessian(model, x).squeeze()
        torch.testing.assert_close(expected, actual, rtol=0.0, atol=atol)


def test_hessian_nonlinear12():
    layer_sizes = [30, 25, 20, 15, 10, 5]
    atol = 1e-4

    test_args = dict(atol=atol, rtol=0.0)
    for idx in range(10):
        model = build_softplus_model(layer_sizes)
        num_layers = len(model)

        nonlinear_indices = [idx for idx, _ in enumerate(model)
                   if type(_) == torch.nn.Softplus]
        linear_indices = sorted(set(range(num_layers)) - set(nonlinear_indices))
        num_nonlinearities = len(nonlinear_indices)

        x = torch.randn(1, layer_sizes[0])
        ddsps = [ddsoftplus(model[:i](x)) for i in nonlinear_indices]
        dsps = [dsoftplus(model[:i](x)) for i in nonlinear_indices]
        weights = [model[_].weight for _ in linear_indices]
        deriv_seq = interleave_lists(weights, dsps) + [weights[-1]]

        model_g = torch.autograd.functional.jacobian(model, x).squeeze()
        hess_terms = [None] * num_nonlinearities

        for _ in range(num_nonlinearities):
            # _ = 2
            ind = nonlinear_indices[_]
            bef = deriv_seq[:ind]
            aft = deriv_seq[ind+1:]
            assert 1 + len(bef) + len(aft) == len(deriv_seq)
            after = prod_list(aft, "left").T
            before = prod_list(bef, "left").T
            hess_terms[_] = utils.tensor.reduce(ddsps[_], before, after)

            torch.testing.assert_close((before @ dsps[_] @ after).T, model_g, **test_args)
            x_at = model[:ind+1](x)
            fun_at = model[ind+1:]
            g_at = torch.autograd.functional.jacobian(fun_at, x_at).squeeze().T
            num_above = sum(__ > ind for __ in linear_indices)
            assert num_above == (len(deriv_seq) - ind) / 2
            torch.testing.assert_close(after, g_at, **test_args)

        actual = sum(hess_terms).squeeze()
        expected = vectorized_hessian(model, x).squeeze()
        torch.testing.assert_close(expected, actual, **test_args)

#
# def reduce_triple_sum(center: torch.Tensor,
#                       before: torch.Tensor,
#                       after: torch.Tensor) -> torch.Tensor:
#     m1 = before
#     m2 = after
#     m3 = before.T
#
#     d1, d2, d3 = center.shape
#     _1, s1 = m1.shape
#     assert _1 == d1
#     _2, s2 = m2.shape
#
#     n1, n2 = before.shape
#     _, n3 = after.shape
#     assert _ == n2
#     assert (_, _, _) == center.shape
#
#     reduced = torch.empty((n1, n3, n1))
#
#     for i in range(n1):
#         for j in range(n3):
#             for k in range(n1):
#                 reduced[i, j, k] = 0.0
#     return reduced


def test_hessian_nonlinear13():
    layer_sizes = [30, 25, 20, 15, 10, 5]

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        model = build_softplus_model(layer_sizes)
        num_layers = len(model)

        nonlinear_indices = [idx for idx, _ in enumerate(model)
                   if type(_) == torch.nn.Softplus]
        linear_indices = sorted(set(range(num_layers)) - set(nonlinear_indices))
        num_nonlinearities = len(nonlinear_indices)

        x = torch.randn(1, layer_sizes[0])
        # ddsps = [ddsoftplus(model[:i](x)) for i in nonlinear_indices]
        dsps = [dsoftplus(model[:i](x)) for i in nonlinear_indices]
        weights = [model[_].weight for _ in linear_indices]
        deriv_seq = interleave_lists(weights, dsps) + [weights[-1]]

        model_g = torch.autograd.functional.jacobian(model, x).squeeze()

        for _ in range(num_nonlinearities):
            # _ = 2
            ind = nonlinear_indices[_]
            bef = deriv_seq[:ind]
            aft = deriv_seq[ind+1:]
            assert 1 + len(bef) + len(aft) == len(deriv_seq)
            after = prod_list(aft, "left").T
            before = prod_list(bef, "left").T

            torch.testing.assert_close((before @ dsps[_] @ after).T, model_g, **test_args)

            x_at = model[:ind+1](x)
            fun_at = model[ind+1:]
            g_at = torch.autograd.functional.jacobian(fun_at, x_at).squeeze().T
            num_above = sum(__ > ind for __ in linear_indices)
            assert num_above == (len(deriv_seq) - ind) / 2
            torch.testing.assert_close(after, g_at, **test_args)


def test_network_loss_hessian():
    n0 = 30
    n1 = 25
    n2 = 20
    n3 = 15
    n4 = 10
    n5 = 5

    layer_sizes = [n0, n1, n2, n3, n4, n5]

    atol = 1e-4
    test_args = dict(atol=atol, rtol=0.0)

    for idx in range(10):
        model = build_softplus_model(layer_sizes)

        x = torch.randn(1, layer_sizes[0])
        y = torch.randint(layer_sizes[-1], (1,))

        model_h = vectorized_hessian(model, x).squeeze()
        full_model = lambda _: torch.nn.LogSoftmax(1)(model(_))
        loss = lambda _: torch.nn.LogSoftmax(1)(model(_)).flatten()[y]

        full_model_g = torch.autograd.functional.jacobian(full_model, x).squeeze()
        full_model_h = vectorized_hessian(full_model, x).squeeze()

        model_g = torch.autograd.functional.jacobian(model, x).squeeze()
        deriv1 = dlogsoftmax(model(x))
        deriv2 = ddlogsoftmax(model(x))

        actual_g = (deriv1 @ model_g)
        torch.testing.assert_close(full_model_g, actual_g, **test_args)

        eye_n5 = torch.eye(n5)
        eye_n0 = torch.eye(n0)

        after1 = eye_n5.T
        before1 = model_g.T
        center1 = deriv2

        after2 = deriv1.T
        before2 = eye_n0.T
        center2 = model_h

        term1 = utils.tensor.reduce(center1, before1, after1)
        term2 = utils.tensor.reduce(center2, before2, after2)
        term12 = bfun(before1, center1)
        torch.testing.assert_close(term1, term12, **test_args)

        term22 = torch.einsum("ijk,jl->ilk", center2, after2)
        torch.testing.assert_close(term2, term22, **test_args)

        actual_h = term1 + term2

        torch.testing.assert_close(actual_h, full_model_h, **test_args)

        loss_g = torch.autograd.functional.jacobian(loss, x)
        loss_h = vectorized_hessian(loss, x)

        torch.testing.assert_close(full_model_g[y, :].squeeze(), loss_g.squeeze(), **test_args)
        torch.testing.assert_close(full_model_h[:, y, :].squeeze(), loss_h.squeeze(), **test_args)


def nth_basis_vector(n: int, dim: int) -> torch.Tensor:
    out = torch.zeros((dim,))
    out[n] = 1
    return out


def ddlogsoftmax(x: torch.Tensor) -> torch.Tensor:
    sm = torch.exp(torch.nn.LogSoftmax(1)(x))
    n_out = sm.shape[1]
    actual_h = torch.stack([sm.T @ sm - torch.diag(sm.flatten())] * n_out, 1)
    return actual_h


def dlogsoftmax(x: torch.Tensor) -> torch.Tensor:
    exp_logsoftmax = torch.exp(torch.nn.LogSoftmax(1)(x))
    n_out = exp_logsoftmax.numel()
    dls_mat = torch.eye(n_out) - exp_logsoftmax
    return dls_mat


def test_logsoftmax_derivatives():
    nr = 1
    n_out = 10
    x = torch.randn((nr, n_out))

    which_ind = 0
    normalized_logits1 = lambda _: torch.nn.LogSoftmax(1)(_).flatten()[which_ind]

    expected_g = torch.autograd.functional.jacobian(normalized_logits1, x)
    actual_g = nth_basis_vector(which_ind, n_out) - torch.exp(torch.nn.LogSoftmax(1)(x))
    torch.testing.assert_close(expected_g, actual_g)

    # More generally, check the entire jacobian at once (all inputs, all outputs)
    expected_g = torch.autograd.functional.jacobian(torch.nn.LogSoftmax(1), x).squeeze()
    actual_g = dlogsoftmax(x)
    torch.testing.assert_close(expected_g, actual_g)

    actual_h = ddlogsoftmax(x)
    expected_h = vectorized_hessian(torch.nn.LogSoftmax(1), x).squeeze()
    assert (torch.diff(expected_h, dim=1) == 0).all()
    torch.testing.assert_close(actual_h, expected_h)


def dcrossentropyloss(x: torch.Tensor,
                      y: torch.Tensor) -> torch.Tensor:
    assert 1 == y.shape[0], "This function only works one row at a time"
    assert 1 == x.shape[0], "This function only works one row at a time"
    dlsm = dlogsoftmax(x)
    dxentl = -1 * dlsm[:, y]
    return dxentl


def test_loss_definitions():
    nr = 1
    n_out = 10
    logits = torch.randn((nr, n_out))
    logsoftmax_logits = torch.nn.LogSoftmax(1)(logits)

    target = torch.randint(n_out, (nr,))
    nl_loss = torch.nn.NLLLoss()(logsoftmax_logits, target)
    ce_loss = torch.nn.CrossEntropyLoss()(logits, target)
    torch.testing.assert_close(nl_loss, ce_loss)

    manual_loss = -1 * logsoftmax_logits.flatten()[target][0]
    torch.testing.assert_close(manual_loss, ce_loss)


def test_new_hessian_orientation():
    layer_sizes = [30, 25, 20, 15, 10, 5]
    model = build_softplus_model(layer_sizes)
    num_layers = len(model)

    nonlinear_indices = [idx for idx, _ in enumerate(model)
               if type(_) == torch.nn.Softplus]
    linear_indices = sorted(set(range(num_layers)) - set(nonlinear_indices))
    num_nonlinearities = len(nonlinear_indices)

    x = torch.randn(1, layer_sizes[0])
    ddsps = [ddsoftplus(model[:i](x)) for i in nonlinear_indices]
    dsps = [dsoftplus(model[:i](x)) for i in nonlinear_indices]
    weights = [model[_].weight for _ in linear_indices]
    deriv_seq = interleave_lists(weights, dsps) + [weights[-1]]

    model_g = torch.autograd.functional.jacobian(model, x).squeeze()
    hess_terms = [None] * num_nonlinearities

    for _ in range(num_nonlinearities):
        # _ = 2
        ind = nonlinear_indices[_]
        bef = deriv_seq[:ind]
        aft = deriv_seq[ind+1:]
        assert 1 + len(bef) + len(aft) == len(deriv_seq)
        after = prod_list(aft, "left").T
        before = prod_list(bef, "left").T
        hess = ddsps[_]
        hess2 = hess.swapaxes(1, 2)
        # mlist = [before, before.T, after]
        # mlist = [before.T, torch.eye(25), after.T]
        # m1 = torch.ones(25, 40)
        m1 = before.T
        m2 = before.T
        m3 = after
        mlist = [m1, m2, m3]
        kk = utils.tensor.covariant_multilinear_matrix_multiplication(hess2, mlist).swapaxes(1, 2)
        hess_terms[_] = utils.tensor.reduce(hess, before, after)
        torch.testing.assert_close(kk, hess_terms[_])
        torch.testing.assert_close((before @ dsps[_] @ after).T, model_g)

        x_at = model[:ind+1](x)
        fun_at = model[ind+1:]
        g_at = torch.autograd.functional.jacobian(fun_at, x_at).squeeze().T
        num_above = sum(__ > ind for __ in linear_indices)
        assert num_above == (len(deriv_seq) - ind) / 2
        torch.testing.assert_close(after, g_at, atol=5e-5, rtol=0.0)


if __name__ == "__main__":
    test_hessian_nonlinear13()
    test_hessian_nonlinear12()
    test_hessian_nonlinear11()

    test_first_second_derivative_relationship()

    test_new_hessian_orientation()

    # test_network_partial_gradient()

    test_loss_definitions()
    test_logsoftmax_derivatives()

    test_network_loss_hessian()
    test_hessian_nonlinear9()
    test_hessian_nonlinear8()
    test_hessian_nonlinear7()
    test_hessian_nonlinear6()
    test_hessian_nonlinear5()
    test_hessian_nonlinear4()
    test_hessian_nonlinear3()
    test_hessian_nonlinear2()
    test_hessian_nonlinear1()
    test_ddsp2()

    test_ddsp1()
    test_dsp1()

    print("Done")
