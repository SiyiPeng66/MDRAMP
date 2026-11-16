import json
from types import SimpleNamespace
from mdramp import release_train


def test_train_forwards_cuda_to_feature_preparation_and_all_stages(tmp_path, monkeypatch):
    public = tmp_path/'data/public'
    public.mkdir(parents=True)
    stages = [{'name': n, 'module': m, 'options': {'device': 'cpu'}} for n, m in
              [('experts', 'src.train.pretrain_public'), ('interaction', 'src.train.train_target_affinity'), ('generator', 'src.generation.cli')]]
    (public/'training_plan.json').write_text(json.dumps({'stages': stages}))
    monkeypatch.setattr(release_train, 'ROOT', tmp_path)
    monkeypatch.setattr(release_train, 'resolve_model', lambda _: {'config': {'esm_checkpoint': str(tmp_path/'weights')}})
    prep = []
    def prepare(plan, root, weights, device='cpu'):
        prep.append(device)
        return plan
    monkeypatch.setattr(release_train, 'prepare_public_features', prepare)
    commands = []
    monkeypatch.setattr(release_train.subprocess, 'run', lambda cmd, **kw: commands.append(cmd))
    args = SimpleNamespace(epochs=1, batch_size=2, learning_rate=.001, seed=11,
                           threads=2, device='cuda', output_dir=str(tmp_path/'trained'))
    release_train.run(args)
    assert prep == ['cuda']
    assert len(commands) == 3
    assert all(cmd[cmd.index('--device')+1] == 'cuda' for cmd in commands)
