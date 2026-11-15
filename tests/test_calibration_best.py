import json
import math
import pandas as pd
import pytest
from mdramp.calibration_best import STRAINS, aggregate, minimum_interval, prepare, sha256


def interval(lo, hi, li=True, ui=True):
    return dict(lower_log2=lo, upper_log2=hi, lower_inclusive=li, upper_inclusive=ui)


def test_minimum_strict_endpoints_and_exact():
    assert minimum_interval([interval(1, 5, False), interval(1, 6)]) == (1, 5, True, True)
    assert minimum_interval([interval(1, 5), interval(2, 5, True, False)]) == (1, 5, True, False)
    assert minimum_interval([interval(4, 4), interval(5, math.inf)]) == (4, 4, True, True)
    assert minimum_interval([interval(-math.inf, 5), interval(5, 8, False)]) == (-math.inf, 5, False, True)
    with pytest.raises(ValueError): minimum_interval([interval(4, 4, False)])


def fixture():
    sequences = pd.DataFrame([dict(peptide_id='a', sequence='AAAA', cluster_id='c', fold=0)])
    observations = pd.DataFrame([dict(peptide_id='a', strain_id=s, status='usable', cluster_id='c', fold=0,
                                     provenance='[]', **interval(-math.inf, 5)) for s in STRAINS])
    return sequences, observations


def test_panel_missing_or_invalid_rejected_and_screen_not_exact():
    seq, obs = fixture()
    result = aggregate(seq, obs)
    assert result.observation_type.item() == 'left_censored'
    assert result.exact_log2.isna().all()
    assert len(json.loads(result.provenance.item())) == 6
    with pytest.raises(ValueError): aggregate(seq, obs.iloc[:-1])
    obs.loc[0, 'status'] = 'quarantined'
    with pytest.raises(ValueError): aggregate(seq, obs)


def test_atomic_versioned_provenance(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    seq, obs = fixture()
    seq.to_csv(source / 'sequences.csv', index=False); obs.to_csv(source / 'observations.csv', index=False)
    manifest = {'output_sha256':{name:sha256(source / name) for name in ('sequences.csv','observations.csv')}}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    output = tmp_path / 'result'
    produced = prepare(source, output)
    assert sha256(output / 'sequences.csv') == sha256(source / 'sequences.csv')
    for name, digest in produced['output_sha256'].items(): assert sha256(output / name) == digest
    with pytest.raises(ValueError): prepare(source, output)
    (source / 'observations.csv').write_text('tampered')
    with pytest.raises(ValueError, match='hash mismatch'): prepare(source, tmp_path / 'failed')
    assert not (tmp_path / 'failed').exists()


def test_derived_targets_cannot_be_relabelled_as_measurements():
    seq, obs = fixture()
    for column, value in [('target_semantics', 'design_target_not_measured'), ('target_divisor', 5), ('measured_lower_log2', 0)]:
        invalid = obs.copy(); invalid[column] = value
        with pytest.raises(ValueError, match='measured|Measured'):
            aggregate(seq, invalid)
