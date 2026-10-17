"""Structured-state action-conditioned latent rollout, trained from scratch.

EMA JEPA, LeWM-inspired SIGReg, and supervised-only objectives are separate.
Inputs: states [B,D], actions [B,H,A]; outputs latents [B,H,L], states
[B,H,D], nonnegative cost components [B,H,3]. Normalization is train-only.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F


class ControlledWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=64, objective='ema', use_actions=True):
        super().__init__()
        if objective not in ('ema', 'sigreg', 'supervised'):
            raise ValueError('Unknown objective')
        self.objective, self.use_actions = objective, use_actions
        self.encoder = nn.Sequential(nn.Linear(state_dim, latent_dim), nn.SiLU(),
                                     nn.Linear(latent_dim, latent_dim))
        self.target = copy.deepcopy(self.encoder) if objective == 'ema' else None
        if self.target is not None:
            self.target.requires_grad_(False)
        self.predictor = nn.Sequential(nn.Linear(latent_dim+action_dim, latent_dim),
                                       nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.state_head = nn.Linear(latent_dim, state_dim)
        self.cost_head = nn.Sequential(nn.Linear(latent_dim, 32), nn.SiLU(), nn.Linear(32, 3), nn.Softplus())
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_scale', torch.ones(state_dim))
        self.register_buffer('cost_scale', torch.ones(3))
        self.register_buffer('action_scale', torch.ones(action_dim))

    def fit_preprocessing(self, states, costs, maximum):
        self.state_mean.copy_(states.mean(0))
        self.state_scale.copy_(states.std(0, unbiased=False).clamp_min(1.))
        self.cost_scale.copy_(costs.square().mean(0).sqrt().clamp_min(1.))
        self.action_scale.fill_(maximum)

    def normalize(self, states):
        return (states-self.state_mean)/self.state_scale

    def forward(self, states, actions):
        if states.ndim != 2 or actions.ndim != 3 or states.shape[0] != actions.shape[0] or actions.shape[1] < 1:
            raise ValueError('Expected aligned [B,D], [B,H,A] with positive horizon')
        z = self.encoder(self.normalize(states))
        rollout = []
        for action in actions.unbind(1):
            a = action / self.action_scale if self.use_actions else torch.zeros_like(action)
            z = self.predictor(torch.cat((z, a), -1))
            rollout.append(z)
        latent = torch.stack(rollout, 1)
        return latent, self.state_head(latent), self.cost_head(latent)

    def loss(self, states, actions, futures, costs, sigreg_weight=0.1):
        latent, state_pred, cost_pred = self(states, actions)
        state_loss = F.mse_loss(state_pred, self.normalize(futures))
        cost_loss = F.mse_loss(cost_pred, costs/self.cost_scale)
        latent_loss = latent.new_zeros(())
        regularizer = latent.new_zeros(())
        if self.objective == 'ema':
            with torch.no_grad():
                target = self.target(self.normalize(futures))
            latent_loss = F.mse_loss(latent, target)
        elif self.objective == 'sigreg':
            target = self.encoder(self.normalize(futures))
            latent_loss = F.mse_loss(latent, target)
            regularizer = sigreg(target.transpose(0,1))
        loss = state_loss + cost_loss + latent_loss + sigreg_weight * regularizer
        return loss, dict(state=float(state_loss.detach()), cost=float(cost_loss.detach()),
                          latent=float(latent_loss.detach()), regularizer=float(regularizer.detach()),
                          latent_std=float(latent.detach().flatten(0,1).std(0,unbiased=False).mean()))

    @torch.no_grad()
    def update_target(self, momentum=.99):
        if not 0 <= momentum <= 1:
            raise ValueError('Invalid EMA momentum')
        if self.target is not None:
            for target, online in zip(self.target.parameters(), self.encoder.parameters()):
                target.lerp_(online, 1-momentum)

    @torch.no_grad()
    def predict_components(self, state, actions):
        self.eval()
        device = self.state_mean.device
        a = torch.as_tensor(actions, dtype=torch.float32, device=device)
        s = torch.as_tensor(state, dtype=torch.float32, device=device).expand(len(a), -1)
        return (self(s,a)[2]*self.cost_scale).cpu().numpy()


def sigreg(z, projections=128, knots=17):
    """LeWM Epps–Pulley sketch statistic; [time, independent batch, latent].

    Reimplemented from the MIT reference; no EMA or stop-gradient in this variant.
    """
    if z.ndim != 3 or z.shape[1] < 2 or knots < 2 or projections < 1:
        raise ValueError('SIGReg requires [T,B>=2,D], knots>=2, projections>=1')
    direction = torch.randn(z.shape[-1], projections, device=z.device, dtype=z.dtype)
    direction = F.normalize(direction, dim=0)
    t = torch.linspace(0,3,knots,device=z.device,dtype=z.dtype)
    phi = torch.exp(-t.square()/2)
    weights = torch.full_like(t, 6/(knots-1)); weights[[0,-1]] /= 2
    phase = (z@direction).unsqueeze(-1)*t
    error = (phase.cos().mean(1)-phi).square()+phase.sin().mean(1).square()
    return (error@(weights*phi)).mean()*z.shape[1]


# ---------------------------------------------------------------------------
# decision_v3 dynamics comparators (round_claude, 2026-10-17)
#
# All models expose rollout(states [B,D], actions [B,H,A]) -> (states [B,H,D],
# transport [B,H]) in physical units, so evaluation and planning treat them
# identically. Targets are per-feature normalized one-step deltas; features
# whose delta never varies in training are held fixed (predicted change 0).
# Known physical constraint: inventory/backlog/order/pipeline features are
# projected to be nonnegative after every predicted step.
# ---------------------------------------------------------------------------

class Normalizer(nn.Module):
    """Train-only statistics, persisted inside every checkpoint."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        for name, size in (('x_mean', state_dim), ('x_std', state_dim), ('d_mean', state_dim),
                           ('d_std', state_dim), ('d_mask', state_dim), ('nonneg', state_dim),
                           ('a_scale', action_dim), ('t_mean', 1), ('t_std', 1)):
            self.register_buffer(name, torch.ones(size) if 'std' in name or 'scale' in name else torch.zeros(size))

    def fit(self, states, actions, next_states, transport, maximum, nonneg):
        delta = next_states - states
        self.x_mean.copy_(states.mean(0))
        self.x_std.copy_(torch.where(states.std(0) > 1e-6, states.std(0), torch.ones_like(states[0])))
        varies = delta.std(0) > 1e-6
        self.d_mask.copy_((varies | (delta.mean(0).abs() > 1e-6)).float())   # constant nonzero drift kept
        self.d_mean.copy_(delta.mean(0) * self.d_mask)
        self.d_std.copy_(torch.where(varies, delta.std(0), torch.ones_like(states[0])))
        self.a_scale.fill_(float(maximum))
        self.t_mean.fill_(float(transport.mean())); self.t_std.fill_(float(transport.std().clamp_min(1e-6)))
        self.nonneg.copy_(torch.as_tensor(nonneg, dtype=torch.float32))
        return self

    def x(self, s):
        return (s - self.x_mean) / self.x_std

    def apply(self, s, out):
        """Normalized network output [.., D+1] -> next physical state, transport."""
        delta = out[..., :-1] * self.d_std * self.d_mask
        nxt = s + delta
        nxt = torch.where(self.nonneg > 0, nxt.clamp_min(0), nxt)
        transport = (out[..., -1] * self.t_std + self.t_mean).clamp_min(0)
        return nxt, transport

    def step_loss(self, s_prev_true, pred, true, t_pred, t_true):
        """Per-feature normalized error of the predicted state; masked features excluded."""
        err = ((pred - true) / self.d_std).square() * self.d_mask
        return err.sum(-1).mean() / self.d_mask.sum() + ((t_pred - t_true) / self.t_std).square().mean()


def mlp(i, o, width, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]; d = width
    return nn.Sequential(*layers, nn.Linear(d, o))


class DeltaDynamics(nn.Module):
    """Supervised direct physical-state dynamics; trained by open-loop rollout loss."""
    kind = 'delta_mlp'

    def __init__(self, state_dim, action_dim, width=256, depth=3):
        super().__init__()
        self.norm = Normalizer(state_dim, action_dim)
        self.net = mlp(state_dim + action_dim, state_dim + 1, width, depth)

    def step(self, s, a):
        return self.norm.apply(s, self.net(torch.cat((self.norm.x(s), a / self.norm.a_scale), -1)))

    def rollout(self, s, actions):
        states, transport = [], []
        for a in actions.unbind(1):
            s, t = self.step(s, a); states.append(s); transport.append(t)
        return torch.stack(states, 1), torch.stack(transport, 1)

    def loss(self, s0, actions, futures, transport):
        pred, tp = self.rollout(s0, actions)
        loss = sum(self.norm.step_loss(None, pred[:, k], futures[:, k], tp[:, k], transport[:, k])
                   for k in range(actions.shape[1])) / actions.shape[1]
        return loss, dict(state=float(loss.detach()))


class LatentDynamics(nn.Module):
    """Encoder / latent predictor / residual decoder with selectable objective.

    latent_sup: decoder loss only (architecture control).
    ema:        + MSE(predicted latent, stop-grad EMA-target encoding of true future)  [V-JEPA 2-AC style target]
    sigreg:     + MSE(predicted latent, online encoding) + SIGReg on encodings, no EMA  [LeWM style]
    jepa_only:  latent + SIGReg losses train encoder/predictor; decoder sees detached latents
                (tests what the JEPA objective alone retains; decoder acts as a probe).
    Physical predictions decode consecutive latents into a state delta, so
    multi-step rollout stays in latent space (no re-encoding of predictions).
    """
    kind = 'latent'

    def __init__(self, state_dim, action_dim, objective='latent_sup', latent=64, width=256,
                 latent_weight=1.0, sigreg_weight=0.1):
        super().__init__()
        if objective not in ('latent_sup', 'ema', 'sigreg', 'jepa_only'):
            raise ValueError('Unknown objective')
        self.objective, self.latent_weight, self.sigreg_weight = objective, latent_weight, sigreg_weight
        self.norm = Normalizer(state_dim, action_dim)
        self.encoder = mlp(state_dim, latent, width)
        self.predictor = mlp(latent + action_dim, latent, width)
        self.decoder = mlp(2 * latent, state_dim + 1, width)
        self.target = copy.deepcopy(self.encoder).requires_grad_(False) if objective == 'ema' else None

    def encode(self, s):
        return self.encoder(self.norm.x(s))

    def latent_rollout(self, s, actions):
        z = self.encode(s); zs = [z]
        for a in actions.unbind(1):
            z = z + self.predictor(torch.cat((z, a / self.norm.a_scale), -1)); zs.append(z)
        return torch.stack(zs, 1)

    def rollout(self, s, actions, zs=None):
        zs = self.latent_rollout(s, actions) if zs is None else zs
        states, transport = [], []
        for k in range(actions.shape[1]):
            z0, z1 = zs[:, k], zs[:, k + 1]
            if self.objective == 'jepa_only':
                z0, z1 = z0.detach(), z1.detach()
            s, t = self.norm.apply(s, self.decoder(torch.cat((z0, z1), -1)))
            states.append(s); transport.append(t)
        return torch.stack(states, 1), torch.stack(transport, 1)

    def loss(self, s0, actions, futures, transport):
        zs = self.latent_rollout(s0, actions)
        pred, tp = self.rollout(s0, actions, zs)
        H = actions.shape[1]
        dec = sum(self.norm.step_loss(None, pred[:, k], futures[:, k], tp[:, k], transport[:, k]) for k in range(H)) / H
        lat = reg = dec.new_zeros(())
        if self.objective == 'ema':
            with torch.no_grad():
                tgt = self.target(self.norm.x(futures))
            lat = F.mse_loss(zs[:, 1:], tgt)
        elif self.objective in ('sigreg', 'jepa_only'):
            enc = self.encode(torch.cat((s0[:, None], futures), 1))
            lat = F.mse_loss(zs[:, 1:], enc[:, 1:])
            reg = sigreg(enc.transpose(0, 1))
        loss = dec + self.latent_weight * lat + self.sigreg_weight * reg
        return loss, dict(state=float(dec.detach()), latent=float(lat.detach()), regularizer=float(reg.detach()),
                          latent_std=float(zs.detach().flatten(0, 1).std(0).mean()))

    @torch.no_grad()
    def update_target(self, momentum=.99):
        if self.target is not None:
            for t, o in zip(self.target.parameters(), self.encoder.parameters()):
                t.lerp_(o, 1 - momentum)


class RidgeDynamics(nn.Module):
    """Linear one-step delta model on normalized [state, action, 1]; alpha chosen on validation."""
    kind = 'ridge'

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.norm = Normalizer(state_dim, action_dim)
        self.register_buffer('W', torch.zeros(state_dim + action_dim + 1, state_dim + 1))

    def features(self, s, a):
        return torch.cat((self.norm.x(s), a / self.norm.a_scale, torch.ones_like(s[..., :1])), -1)

    def fit(self, s, a, n, t, alpha):
        X = self.features(s, a).double()
        Y = torch.cat(((n - s - self.norm.d_mean) / self.norm.d_std * self.norm.d_mask,
                       ((t - self.norm.t_mean) / self.norm.t_std)[:, None]), -1).double()
        A = X.T @ X + alpha * torch.eye(X.shape[1], dtype=torch.float64)
        self.W.copy_(torch.linalg.solve(A, X.T @ Y).float())
        return self

    def step(self, s, a):
        out = self.features(s, a) @ self.W
        out = torch.cat((out[..., :-1] + self.norm.d_mean / self.norm.d_std * self.norm.d_mask, out[..., -1:]), -1)
        return self.norm.apply(s, out)

    rollout = DeltaDynamics.rollout


class ConstantDelta(nn.Module):
    """Persistence (zero delta) or train-mean delta; transport predicted by its train mean."""

    def __init__(self, state_dim, action_dim, use_mean=False):
        super().__init__()
        self.kind = 'train_mean_delta' if use_mean else 'persistence'
        self.use_mean = use_mean
        self.norm = Normalizer(state_dim, action_dim)

    def step(self, s, a):
        out = torch.zeros(*s.shape[:-1], s.shape[-1] + 1, dtype=s.dtype, device=s.device)
        if self.use_mean:
            out[..., :-1] = self.norm.d_mean / self.norm.d_std
        return self.norm.apply(s, out)

    rollout = DeltaDynamics.rollout


class Ensemble(nn.Module):
    """Mean prediction plus member dispersion (heuristic, not calibrated, unless shown)."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members); self.kind = 'ensemble'

    def rollout(self, s, actions, return_members=False):
        outs = [m.rollout(s, actions) for m in self.members]
        st = torch.stack([o[0] for o in outs]); tr = torch.stack([o[1] for o in outs])
        return (st.mean(0), tr.mean(0), st, tr) if return_members else (st.mean(0), tr.mean(0))
