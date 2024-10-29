# https://github.com/stanford-crfm/levanter/blob/main/tests/test_sophia.py
import functools
import os
#
import equinox as eqx
import equinox.nn
import torch.nn
import jax
import jax.numpy as jnp
import numpy as np
from chex import assert_trees_all_close

import levanter
import levanter.optim.sophia

import sophia_orig


def test_sophia_h():
    key = jax.random.PRNGKey(0)
    model = equinox.nn.Linear(4, 4, use_bias=False, key=key)
    data = np.load(f"{os.path.dirname(__file__)}/data/hero_data.npy").astype("float32")
    optimizer = levanter.optim.sophia.sophia_h(
        lr=1,
        b1=0,
        b2=0.99,
        gamma=2,
        weight_decay=0.0,
        clip_threshold=1,
        key=key,
        update_interval=1,
    )
    model = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), model)
    zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), model)
    opt_state = optimizer.init(model)

    def loss_fn(model, data):
        out = eqx.filter_vmap(model)(data)
        return jnp.mean(out**2) * 4

    jit_update = eqx.filter_jit(optimizer.update)
    obj_fn = functools.partial(loss_fn, data=data)
    for i in range(1000):
        _, opt_state = jit_update(zero_grad, opt_state, params=model, obj_fn=obj_fn)
    assert_trees_all_close(opt_state[0].h.weight, 2, rtol=0.2, atol=0.3)  # this is very approximate

    grad_loss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))

    loss, grad = grad_loss_fn(model, data)
    model_updates, opt_state = optimizer.update(grad, opt_state, params=model, obj_fn=obj_fn)
    model = eqx.apply_updates(model, model_updates)

    assert_trees_all_close(loss, 15.74834156036377, rtol=1e-3, atol=1e-3)
    assert_trees_all_close(model.weight, 0.5, rtol=0.2, atol=0.1)

    for i in range(10):
        loss, grad = grad_loss_fn(model, data)
        model_updates, opt_state = optimizer.update(grad, opt_state, params=model, obj_fn=obj_fn)
        model = eqx.apply_updates(model, model_updates)
        assert loss < 15.74834156036377 * 0.75 ** (i + 1)
    print('Done')


def test_sophia_kyle_h1():
    key = jax.random.PRNGKey(0)

    model_jx = equinox.nn.Linear(4, 4, use_bias=False, key=key)
    data = np.load(f"{os.path.dirname(__file__)}/data/hero_data.npy").astype("float32")
    optimizer = levanter.optim.sophia.sophia_h(
        lr=1,
        b1=0,
        b2=0.99,
        gamma=2,
        weight_decay=0.0,
        clip_threshold=1,
        key=key,
        update_interval=1,
    )
    model = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), model_jx)
    zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), model_jx)

    opt_state = optimizer.init(model_jx)

    def loss_fn(model, data):
        out = eqx.filter_vmap(model)(data)
        return jnp.mean(out**2) * 4

    jit_update = eqx.filter_jit(optimizer.update)

    obj_fn = functools.partial(loss_fn, data=data)
    for i in range(1000):
        _, opt_state = jit_update(zero_grad, opt_state, params=model_jx, obj_fn=obj_fn)

    # print('Test-estimated hessian: most coordinates should be approximately 2')
    # print('Estimated hessian:', opt_state[0].h.weight)
    assert_trees_all_close(opt_state[0].h.weight, 2, rtol=0.2, atol=0.3)  # this is very approximate

    grad_loss_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))

    loss, grad = grad_loss_fn(model, data)
    model_updates, opt_state = optimizer.update(grad, opt_state, params=model, obj_fn=obj_fn)
    model = eqx.apply_updates(model, model_updates)

    assert_trees_all_close(loss, 15.74834156036377, rtol=1e-3, atol=1e-3)

    # print("Test-model param after 1 step: most coordinates should be very loosely 0.5")
    assert_trees_all_close(model.weight, 0.5, rtol=0.2, atol=0.1)  # this is very approximate

    for i in range(10):
        loss, grad = grad_loss_fn(model, data)
        model_updates, opt_state = optimizer.update(grad, opt_state, params=model, obj_fn=obj_fn)
        model = eqx.apply_updates(model, model_updates)

        # print('Step:', i , "Loss:", loss.item())
        assert loss < 15.74834156036377 * 0.75 ** (i + 1)


def test_sophia_kyle_g1():
    key = jax.random.PRNGKey(0)
    model_jx = equinox.nn.Linear(4, 4, use_bias=False, key=key)
    model_pt = torch.nn.Linear(4, 4, bias=False)
    data = np.load(f"{os.path.dirname(__file__)}/data/hero_data.npy").astype("float32")

    args = dict(lr=1,
        b1=0,
        b2=0.99,
        gamma=2,
        weight_decay=0.0,
        clip_threshold=1,
    )
    optimizer_jax = levanter.optim.sophia.sophia_g(
        **args,
        key=key,
        update_interval=1,
    )
    optimizer_torch = sophia_orig.SophiaG(model_pt.parameters(),
                                          lr=args["lr"],
                                          betas=(args["b1"], args["b2"]),
                                          rho=args["gamma"],
                                          weight_decay=args["weight_decay"])
    # https://github.com/stanford-crfm/levanter/blob/331c0aa02eec635fa220fc44267cede455b1bca2/tests/test_sophia.py#L32
    model_jx = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), model_jx)
    zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), model_jx)

    opt_state = optimizer_jax.init(model_jx)

    def loss_fn_jax(model, data):
        out = eqx.filter_vmap(model)(data)
        return jnp.mean(out**2) * 4

    jit_update = eqx.filter_jit(optimizer_jax.update)

    # obj_fn = functools.partial(loss_fn_jax, data=data.T)
    obj_fn = functools.partial(loss_fn_jax, data=data)
    if False:
        obj_fn(model_jx)

    # stochastic_diag_gauss_newton(fn=obj_fn, model=model_jx)
    _, opt_state = jit_update(zero_grad, opt_state, params=model_jx, obj_fn=obj_fn)
    print("done")


if __name__ == "__main__":
    # test_sophia_h()
    test_sophia_kyle_g1()
    test_sophia_kyle_h1()

    print("done")
