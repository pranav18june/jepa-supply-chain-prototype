import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from prototype_new.src.env.isomorph import IsomorphEnv, build_simulation, DispatchSpace
from prototype_new.src.planning.cem import CEMPlanner, Goal, RecedingHorizonController
from prototype_new.src.models.controlled import ControlledWorldModel, sigreg
from prototype_new.src.data.protocols import validate_rollout_splits


@pytest.mark.parametrize('seed,n_items,days',[(5,50,8),(19,3,32)])
def test_upstream_default_parity(seed,n_items,days):
    path=Path('prototype_new/results/execution_audit/upstream_Supplychaingeo_item50.py')
    if not path.exists():
        pytest.skip('Run fetch_audit_sources.py to obtain pinned independent upstream fixture')
    spec=importlib.util.spec_from_file_location('_parity_original',path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    scenario={'ss_scale':.1,'base_lambda_lo':32.,'base_lambda_hi':48.}
    a=build_simulation(seed,days,scenario,n_items,module)
    env=IsomorphEnv(days=days,n_items=n_items,scenario=scenario); env.reset(seed)
    for day in range(days):
        a.step(day); env.step(None)
        for field in ('daily_records','shipments_log','inventory_history','backlog_history',
                      'intransit_history','demand_ema','dest_in_transit'):
            assert getattr(a,field)==getattr(env.sim,field), (day,field)
        assert a.rng.getstate()==env.sim.rng.getstate()
        for nid in a.network.nodes:
            assert vars(a.network.nodes[nid])==vars(env.sim.network.nodes[nid])


def test_action_effect_and_logging():
    a,b=IsomorphEnv(days=12),IsomorphEnv(days=12)
    np.testing.assert_array_equal(a.reset(13),b.reset(13))
    sa,_,_,ia=a.step(np.zeros(3,dtype=int))
    sb,_,_,ib=b.step(np.full(3,500))
    assert not np.array_equal(sa,sb)
    assert ia['realized_dispatch']==[0,0,0]
    assert sum(ib['realized_dispatch'])>0
    assert ia['demand']==ib['demand']
    assert b.records[0]['action']==[500,500,500]
    assert b.records[0]['next_state']==sb.tolist()
    assert all(v>=0 for n in b.sim.network.nodes.values() for v in n.inventory.values())
    assert all(v>=0 for e in b.sim.network.edges.values() for v in e.daily_containers)
    assert all(s['arrival_day']>s['day'] for s in ib['shipments'])


def test_projection_and_rejection():
    s=DispatchSpace(2,10)
    assert s.project([-1,15]).tolist()==[0,10]
    for a in ([np.nan,0],[.5,2],[-1,2],[1,11],[1]):
        with pytest.raises(ValueError): s.validate(a)


def test_cem_tiny_optimum_goal_determinism_and_fallback():
    planner=CEMPlanner(DispatchSpace(1,10),horizon=1,samples=100,elites=10,iterations=5)
    def predict(_,actions):
        x=actions[...,0]
        return np.stack([x*x,(x-8)**2,np.zeros_like(x)],-1)[None]
    result=planner.plan(np.zeros(1),predict,Goal(holding=0),seed=42)
    assert result.actions.tolist()==[[8]] # agrees with enumeration over 0..10
    assert result.evaluations==500 and not result.fallback
    assert planner.plan(None,predict,Goal(backlog=0),42).actions.tolist()==[[0]]
    assert np.array_equal(planner.plan(None,predict,Goal(holding=0),42).actions,result.actions)
    failure=planner.plan(None,lambda s,a: np.full((1,len(a),1,3),np.nan))
    assert failure.fallback and failure.actions.tolist()==[[0]]


def test_execute_observe_replan():
    env=IsomorphEnv(days=3,n_items=1); obs=env.reset(9)
    calls=[]
    def predict(s,a):
        calls.append(s.copy())
        return np.zeros((1,len(a),a.shape[1],3))
    controller=RecedingHorizonController(CEMPlanner(env.space,samples=8,elites=2,iterations=1),predict)
    records=controller.run(env,obs)
    assert len(records)==len(calls)==3
    assert not np.array_equal(calls[0],calls[1])


@pytest.mark.parametrize('objective',['ema','sigreg','supervised'])
def test_model_gradient_shapes_and_checkpoint(objective,tmp_path):
    torch.manual_seed(1)
    model=ControlledWorldModel(8,2,objective=objective)
    s=torch.randn(6,8); a=torch.rand(6,3,2); f=torch.randn(6,3,8); c=torch.rand(6,3,3)
    model.fit_preprocessing(s,c.flatten(0,1),10)
    loss,stats=model.loss(s,a,f,c); loss.backward()
    assert torch.isfinite(loss)
    assert model.predictor[0].weight.grad.abs().sum()>0
    if objective=='ema':
        assert all(p.grad is None for p in model.target.parameters())
        with torch.no_grad(): next(model.encoder.parameters()).add_(1)
        model.update_target(0)
        for p,q in zip(model.target.parameters(),model.encoder.parameters()): assert torch.equal(p,q)
    assert model(s,a)[0].shape==(6,3,64)
    assert not torch.equal(model(s,a)[0],model(s,a+10)[0])
    torch.save(model.state_dict(),tmp_path/'m.pt')
    other=ControlledWorldModel(8,2,objective=objective)
    other.load_state_dict(torch.load(tmp_path/'m.pt',weights_only=True))
    torch.testing.assert_close(model(s,a)[0],other(s,a)[0])


def test_sigreg_collapse_and_split_guards():
    torch.manual_seed(9)
    normal=sigreg(torch.randn(2,256,16))
    collapsed=sigreg(torch.zeros(2,256,16))
    assert collapsed>normal
    rows=[dict(rollout='a',seed=1,config_hash='x',split='train'),
          dict(rollout='b',seed=2,config_hash='y',split='test')]
    validate_rollout_splits(rows)
    for key in ('rollout','seed','config_hash'):
        altered=[rows[0],dict(rows[1],**{key:rows[0][key]})]
        with pytest.raises(ValueError): validate_rollout_splits(altered)


def test_inventory_conservation_over_arrivals_and_replenishment():
    env=IsomorphEnv(days=40,n_items=3,scenario={'ss_scale':.03})
    env.reset(47); rng=np.random.default_rng(89)
    def total():
        return (sum(sum(n.inventory.values()) for n in env.sim.network.nodes.values())+
                sum(o[1] for n in env.sim.network.nodes.values() for o in n.outstanding_orders.values() if o)+
                sum(sum(v.values()) for v in env.sim.dest_in_transit.values()))
    for _ in range(40):
        before=total(); backlog=sum(env.sim.network.nodes[env.sim.destination_id].backlog.values())
        old={(nid,i):o for nid,n in env.sim.network.nodes.items() if n.is_source for i,o in n.outstanding_orders.items()}
        _,_,_,info=env.step(env.space.project(rng.uniform(0,600,3)))
        arrivals=sum(o[1] for nid,n in env.sim.network.nodes.items() if n.is_source
                     for i,o in n.outstanding_orders.items() if o and o!=old.get((nid,i)))
        unfilled=info['demand']-info['served']
        new_backlog=sum(env.sim.network.nodes[env.sim.destination_id].backlog.values())
        backlog_served=backlog+unfilled-new_backlog
        assert total()==before+arrivals-info['served']-backlog_served
        assert info['cost']==sum(info['components'].values())


def test_sigreg_matches_reference_formula():
    # Independent scalar integration reference with the same random projection stream.
    torch.manual_seed(91); z=torch.randn(2,20,5)
    torch.manual_seed(12); actual=sigreg(z,projections=7)
    torch.manual_seed(12); direction=torch.randn(5,7); direction/=direction.norm(dim=0)
    t=torch.linspace(0,3,17); phi=torch.exp(-t*t/2)
    phase=(z@direction)[...,None]*t
    err=(phase.cos().mean(1)-phi).square()+phase.sin().mean(1).square()
    expected=2*torch.trapezoid(err*phi,t,dim=-1).mean()*20
    torch.testing.assert_close(actual,expected)


def test_no_action_ablation_is_invariant():
    model=ControlledWorldModel(5,2,use_actions=False)
    s=torch.randn(4,5)
    torch.testing.assert_close(model(s,torch.zeros(4,3,2))[0],model(s,torch.ones(4,3,2)*100)[0])
