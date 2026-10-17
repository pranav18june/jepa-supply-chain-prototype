"""Engineering checks for decision_v3 dynamics comparators (not evidence of learned dynamics)."""
import numpy as np
import pytest
import torch
from prototype_new.src.models.controlled import (DeltaDynamics, LatentDynamics, RidgeDynamics, ConstantDelta, Ensemble)
from prototype_new.src.planning.cem import CEMPlanner, Goal
from prototype_new.src.env.isomorph import DispatchSpace

D, A = 6, 2


def data(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(n, D, generator=g) * 10; a = torch.rand(n, A, generator=g) * 100
    nxt = s.clone(); nxt[:, 0] += 0.05 * a[:, 0]; nxt[:, 1] -= 0.02 * a[:, 1]   # feature 5 never changes
    nxt[:, 2] += 1.0; t = a.sum(1) * 0.5
    return s, a, nxt, t


def fitted(m):
    s, a, n, t = data(); m.norm.fit(s, a, n, t, 100, [1, 1, 0, 0, 0, 1]); return m


@pytest.mark.parametrize('make', [lambda: DeltaDynamics(D, A, 32), lambda: LatentDynamics(D, A, 'ema', 8, 32),
                                  lambda: LatentDynamics(D, A, 'sigreg', 8, 32), lambda: LatentDynamics(D, A, 'jepa_only', 8, 32),
                                  lambda: ConstantDelta(D, A), lambda: ConstantDelta(D, A, True)])
def test_rollout_shapes_mask_and_nonnegativity(make):
    m = fitted(make()); s = torch.zeros(4, D); acts = torch.rand(4, 3, A) * 100
    st, tr = m.rollout(s, acts)
    assert st.shape == (4, 3, D) and tr.shape == (4, 3)
    assert torch.all(st[..., 5] == s[:, None, 5])            # zero-variance delta feature held fixed
    assert torch.all(st[..., [0, 1, 5]] >= 0) and torch.all(tr >= 0)


def test_ridge_recovers_linear_one_step_effects():
    s, a, n, t = data(); m = fitted(RidgeDynamics(D, A)).fit(s, a, n, t, 1e-4)
    pred, tp = m.step(s[:50], a[:50])
    torch.testing.assert_close(pred[:, :3], n[:50, :3].clamp_min(0), atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(tp, t[:50], atol=1e-2, rtol=1e-3)


def test_delta_mlp_learns_action_effect_beyond_persistence():
    torch.manual_seed(0); s, a, n, t = data(800); m = fitted(DeltaDynamics(D, A, 64))
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    for _ in range(300):
        loss, _ = m.loss(s, a[:, None], n[:, None], t[:, None]); opt.zero_grad(); loss.backward(); opt.step()
    s0 = s[:1].expand(2, -1); acts = torch.tensor([[[0., 0.]], [[100., 0.]]])
    st, _ = m.rollout(s0, acts)
    assert abs(float(st[1, 0, 0] - st[0, 0, 0]) - 5.0) < 1.0   # true paired effect 0.05*100


def test_jepa_only_decoder_does_not_train_encoder_and_ema_target_frozen():
    s, a, n, t = data(64)
    m = fitted(LatentDynamics(D, A, 'jepa_only', 8, 32)); m.latent_weight = m.sigreg_weight = 0.
    loss, _ = m.loss(s, a[:, None], n[:, None], t[:, None]); loss.backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.decoder.parameters())
    e = fitted(LatentDynamics(D, A, 'ema', 8, 32)); loss, _ = e.loss(s, a[:, None], n[:, None], t[:, None]); loss.backward()
    assert all(p.grad is None for p in e.target.parameters())


def test_ensemble_mean_and_members_and_cem_warm_start():
    ms = [fitted(DeltaDynamics(D, A, 16)) for _ in range(3)]; ens = Ensemble(ms)
    s = torch.zeros(2, D); acts = torch.rand(2, 2, A)
    mean, tmean, st, tr = ens.rollout(s, acts, return_members=True)
    torch.testing.assert_close(mean, st.mean(0)); assert st.shape[0] == 3
    planner = CEMPlanner(DispatchSpace(1, 10), horizon=1, samples=16, elites=4, iterations=1, init_std=1.)
    seen = []
    def predict(_, c):
        seen.append(c.copy()); x = c[..., 0]
        return np.stack([np.zeros_like(x), (x - 3.) ** 2, np.zeros_like(x)], -1)[None]
    planner.plan(None, predict, Goal(), seed=0, init=[[3]], extra=[[[7]]])
    assert seen[0][2].tolist() == [[7]] and seen[0][0].tolist() == [[3]]
