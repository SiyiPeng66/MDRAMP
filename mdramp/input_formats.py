"""Public file contracts for peptide inference and experimental calibration."""
from pathlib import Path
import math
import re
import numpy as np
import pandas as pd

AA = frozenset('ACDEFGHIKLMNPQRSTVWY')
MIC = re.compile(r'^(>?)([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$')


def _table(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Input file does not exist: {path}')
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        if suffix == '.xls':
            raise ValueError('Use .xlsx for Excel input; legacy .xls is not supported')
        frame = pd.read_excel(path, sheet_name=0, dtype=str, keep_default_na=False)
    elif suffix in {'.csv', '.tsv'}:
        frame = pd.read_csv(path, sep='\t' if suffix == '.tsv' else ',', dtype=str,
                            keep_default_na=False, encoding='utf-8-sig')
    else:
        raise ValueError('Expected CSV, TSV or Excel (.xlsx) input')
    if frame.empty:
        raise ValueError('Input contains no data rows')
    return frame


def _normalize(frame, lines=None):
    if 'sequence' not in frame:
        raise ValueError('Required column missing: sequence (column names are case-sensitive)')
    frame = frame.copy()
    lines = lines or list(range(2, len(frame) + 2))
    seqs, ids = [], []
    for pos, (_, row) in enumerate(frame.iterrows()):
        raw = row['sequence']
        seq = re.sub(r'\s+', '', str(raw)).upper() if pd.notna(raw) else ''
        bad = sorted(set(seq) - AA)
        if not seq or bad:
            raise ValueError(f'Row {lines[pos]}: empty sequence or invalid amino acid characters {bad}')
        identifier = str(row.get('id', '')).strip()
        if not identifier or identifier.lower() == 'nan':
            identifier = 'peptide_' + str(pos + 1).zfill(4)
        seqs.append(seq); ids.append(identifier)
    frame['sequence'], frame['id'] = seqs, ids
    seen = {}
    for pos, (identifier, seq) in enumerate(zip(ids, seqs)):
        if identifier in seen and seen[identifier] != seq:
            raise ValueError(f'Row {lines[pos]}: id {identifier!r} maps to conflicting sequences')
        seen[identifier] = seq
    return frame


def read_sequences(path):
    """Read normalized id/sequence columns, preserving input order and repetitions."""
    path = Path(path)
    if path.suffix.lower() in {'.fa', '.faa', '.fasta', '.fas'}:
        rows, lines, identifier, parts, start = [], [], None, [], None
        def append_record():
            if identifier is not None:
                rows.append({'id': identifier, 'sequence': ''.join(parts)})
                lines.append(start)
        for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                append_record(); identifier = line[1:].strip(); parts = []; start = n
                if not identifier:
                    raise ValueError(f'Row {n}: FASTA header must contain an identifier')
            else:
                if identifier is None:
                    raise ValueError(f'Row {n}: FASTA sequence must follow a >identifier header')
                parts.append(line)
        append_record()
        if not rows:
            raise ValueError('Input contains no FASTA records')
        frame = _normalize(pd.DataFrame(rows), lines)
    else:
        frame = _normalize(_table(path))
    return frame[['id', 'sequence']].reset_index(drop=True)


def read_experiments(path):
    """Read original-scale MIC observations; > values remain right-censored."""
    frame = _normalize(_table(path))
    if 'mic' not in frame:
        raise ValueError('Required column missing: mic (unit: μg/mL)')
    lows, highs, inclusive = [], [], []
    for pos, value in enumerate(frame.mic):
        token = re.sub(r'\s+', '', str(value))
        match = MIC.fullmatch(token)
        if match is None:
            raise ValueError(f'Row {pos + 2}: mic must be a positive number or >number in μg/mL')
        number = float(match.group(2))
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f'Row {pos + 2}: mic must be positive and finite')
        lower = math.log2(number)
        lows.append(lower); highs.append(math.inf if match.group(1) else lower)
        inclusive.append(not bool(match.group(1)))
    frame['lower_log2'], frame['upper_log2'] = lows, highs
    frame['lower_inclusive'] = inclusive
    frame['upper_inclusive'] = True
    # Same sequence/strain/round is the same observation context. Repeated
    # discordant values require explicit adjudication rather than taking a min.
    keys = ['sequence'] + [c for c in ['strain', 'round'] if c in frame]
    for _, group in frame.groupby(keys, dropna=False, sort=False):
        bounds = group[['lower_log2', 'upper_log2']].drop_duplicates()
        if len(bounds) > 1:
            line = int(group.index[-1]) + 2
            raise ValueError(f'Row {line}: conflicting MIC observations for the same sequence/strain/round')
    return frame.reset_index(drop=True)


def aggregate_best(frame, divisor=1):
    """Return one best-strain interval per sequence, with original bounds retained.

    For observations X_i in [L_i,U_i], min(X_i) is bounded by
    [min(L_i), min(U_i)]. A right-censored minimum stays right-censored;
    mixed exact/censored minima can form an interval, never an invented point.
    """
    if not math.isfinite(divisor) or divisor <= 0:
        raise ValueError('MIC divisor must be positive and finite')
    if frame.empty:
        raise ValueError('No experimental observations')
    rows = []
    for seq, group in frame.groupby('sequence', sort=False):
        lo, hi = float(group.lower_log2.min()), float(group.upper_log2.min())
        lower_rows = group[group.lower_log2.eq(lo)]
        lower_inclusive = bool(lower_rows.lower_inclusive.any()) if 'lower_inclusive' in group else True
        rows.append({'id': group.id.iloc[0], 'sequence': seq,
                     'original_lower_log2': lo, 'original_upper_log2': hi,
                     'lower_log2': lo - math.log2(divisor),
                     'upper_log2': hi - math.log2(divisor),
                     'lower_inclusive': lower_inclusive, 'upper_inclusive': True,
                     'observation_type': 'exact' if lo == hi else 'right_censored' if math.isinf(hi) else 'interval',
                     'observation_count': len(group), 'target_divisor': divisor})
    return pd.DataFrame(rows)
