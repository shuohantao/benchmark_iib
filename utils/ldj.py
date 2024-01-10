# Taken from: https://github.com/rtqichen/residual-flows 
import torch
import torch.nn as nn
import numpy as np


class MemoryEfficientLogDetEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters())
    )


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def get_ldj(nnet, x, n_ldj_iter, n_exact_terms=2, est_name='neumann', mem_eff=True, is_training=True):
    est = {
        'basic': basic_logdet_estimator,
        'neumann': neumann_logdet_estimator,
    }

    if est_name not in est:
        raise ValueError(f'Unknown estimator name: {est_name}')

    vareps = torch.randn_like(x)
    n_ldj_iter, coeff_fn = get_n_ldj_iter_and_coeff_fn(n_ldj_iter, n_exact_terms)

    if mem_eff:
        z, ldj = mem_eff_wrapper(est[est_name], nnet, x, n_ldj_iter, vareps, coeff_fn, is_training)
    else:
        z = nnet(x)
        ldj = est[est_name](z, x, n_ldj_iter, vareps, coeff_fn, is_training)

    return z, ldj


def get_n_ldj_iter_and_coeff_fn(n_ldj_iter, n_exact_terms):
    if n_ldj_iter is None:
        geom_p = 0.5
        sample_fn = lambda m: geometric_sample(geom_p, m)
        rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)

        n_samples = sample_fn(1)
        n_ldj_iter = max(n_samples) + n_exact_terms
        coeff_fn = lambda k: 1 / rcdf_fn(k, n_exact_terms) * \
            sum(n_samples >= k - n_exact_terms) / len(n_samples)
    else:
        coeff_fn = lambda k: 1.

    return n_ldj_iter, coeff_fn


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def basic_logdet_estimator(g, x, n_ldj_iter, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_ldj_iter + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_ldj_iter, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_ldj_iter + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad