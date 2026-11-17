# -*- coding: utf-8 -*-
"""
make_sample_data.py  -  synthetic LAS wells to test the workflow
================================================================
Creates fake (but physically consistent) evaporite wells so the code can be
run without the real data:

    sample_data/train/     5 wells with GR, sonic (DT or VP), shear (DTS or VS),
                           RHOB and a LITO code  -> train_rhob_model.py
    sample_data/predict/   3 wells                -> predict_rhob.py
        SYN-PRED-1  all logs (full blind test)
        SYN-PRED-2  RHOB only in the upper part, VS missing in the lower part
        SYN-PRED-3  no RHOB and no shear log (pure prediction, VS from VP)
    sample_data/sample_config.json   well names + Salt top/base per well

How the logs are built
  * a bed sequence (halite-dominated, with anhydrite, K-salt, marl and
    carbonate beds) is drawn per well; each bed gets a mineral mix
  * DT, DTS, GR and RHOB are mixed linearly from mineral end-members
    (slowness mixes ~linearly; density and GR mix by volume)
  * logs are smoothed (tool vertical resolution) and noise is added; each well
    has its own GR offset (tests GR normalisation), a few bad RHOB readings
    (tests VALID_RANGES) and a few null gaps
  * some wells use another sampling step / other mnemonics (tests resampling
    and curve detection)

The numbers are synthetic - do not use them for any interpretation.
Run:  python make_sample_data.py   (overwrites sample_data/)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lasio

BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
OUT = BASE_DIR / "sample_data"
SEED = 7
NULL = -999.25

# Mineral end-members: DT / DTS (us/ft), GR (API), RHOB (g/cc)
MIN = {
    'Halite':       dict(DT=67.0, DTS=116.0, GR=8.0,   RHOB=2.04),
    'Anhydrite':    dict(DT=50.0, DTS=92.0,  GR=10.0,  RHOB=2.98),
    'Sylvite':      dict(DT=74.0, DTS=130.0, GR=500.0, RHOB=1.86),
    'Carnallite':   dict(DT=83.0, DTS=150.0, GR=220.0, RHOB=1.57),
    'Tachyhydrite': dict(DT=92.0, DTS=165.0, GR=5.0,   RHOB=1.66),
    'Calcite':      dict(DT=49.0, DTS=92.0,  GR=15.0,  RHOB=2.71),
    'Dolomite':     dict(DT=44.0, DTS=82.0,  GR=15.0,  RHOB=2.87),
    'Clay':         dict(DT=95.0, DTS=180.0, GR=110.0, RHOB=2.45),
}
# LITO codes of rhob_common.LITO_TABLE
CODE = {'HALITA': 17, 'ANIDRITA': 15, 'SILVITA': 18, 'CARNALITITA': 19, 'TAQUIDRITA': 20,
        'CALCARIO': 0, 'DOLOMITO': 4, 'MARGA': 11, 'FOLHELHO': 10}

#        name           top    thickness  step     sonic  shear  GR offset  folder
WELLS = [('SYN-TRAIN-1', 2950, 420, 0.1524, 'DT',  'DTS', +0.0, 'train'),
         ('SYN-TRAIN-2', 3100, 380, 0.1524, 'VP',  'VS',  +6.0, 'train'),
         ('SYN-TRAIN-3', 2800, 450, 0.1000, 'DT',  'DTS', -4.0, 'train'),   # finer step
         ('SYN-TRAIN-4', 3300, 360, 0.2000, 'DTCO', 'DTSM', +3.0, 'train'),  # coarser step
         ('SYN-TRAIN-5', 3000, 400, 0.1524, 'DT',  'VS',  -7.0, 'train'),
         ('SYN-PRED-1',  3050, 400, 0.1524, 'DT',  'DTS', +5.0, 'predict'),
         ('SYN-PRED-2',  2900, 430, 0.1524, 'VP',  'VS',  -3.0, 'predict'),
         ('SYN-PRED-3',  3200, 370, 0.1524, 'DT',  None,  +2.0, 'predict')]
MARGIN_TOP, MARGIN_BASE = 40.0, 25.0     # logged above / below the salt interval (m)


def bed_sequence(rng, top, base):
    """List of (z_top, z_base, LITO name, {mineral: fraction})."""
    beds = []
    z = top - MARGIN_TOP
    while z < top:                                  # post-salt marls / shales above
        h = rng.uniform(2, 8)
        name = rng.choice(['MARGA', 'FOLHELHO', 'CALCARIO'], p=[.5, .3, .2])
        beds.append((z, min(z + h, top), name, _mix(rng, name)))
        z += h
    z = top
    while z < base:
        frac = (z - top) / (base - top)             # K-salts more frequent in the lower half
        p_k = 0.08 + 0.17 * frac
        name = rng.choice(['HALITA', 'ANIDRITA', 'K', 'MARGA', 'CALCARIO'],
                          p=[0.62 - p_k, 0.25, p_k, 0.08, 0.05])
        if name == 'K':
            name = rng.choice(['CARNALITITA', 'SILVITA', 'TAQUIDRITA'], p=[.5, .3, .2])
        h = {'HALITA': rng.exponential(9) + 1.0, 'ANIDRITA': rng.uniform(0.4, 4),
             'CARNALITITA': rng.uniform(1, 7), 'SILVITA': rng.uniform(0.8, 4),
             'TAQUIDRITA': rng.uniform(1, 5), 'MARGA': rng.uniform(0.4, 2),
             'CALCARIO': rng.uniform(0.5, 3)}[name]
        beds.append((z, min(z + h, base), name, _mix(rng, name)))
        z += h
    z = base
    while z < base + MARGIN_BASE:                   # carbonates below the salt
        h = rng.uniform(2, 6)
        name = rng.choice(['CALCARIO', 'DOLOMITO', 'MARGA'], p=[.5, .3, .2])
        beds.append((z, z + h, name, _mix(rng, name)))
        z += h
    return beds


def _mix(rng, name):
    u = rng.uniform
    if name == 'HALITA':
        a = u(0, 0.08); c = u(0, 0.03) if rng.random() < 0.3 else 0.0
        return {'Halite': 1 - a - c, 'Anhydrite': a, 'Carnallite': c}
    if name == 'ANIDRITA':
        a = u(0.65, 0.97)
        return {'Anhydrite': a, 'Halite': 1 - a}
    if name in ('CARNALITITA', 'SILVITA', 'TAQUIDRITA'):
        m = {'CARNALITITA': 'Carnallite', 'SILVITA': 'Sylvite', 'TAQUIDRITA': 'Tachyhydrite'}[name]
        k = u(0.45, 0.9)
        return {m: k, 'Halite': 1 - k}
    if name in ('MARGA', 'FOLHELHO'):
        c = u(0.45, 0.85) if name == 'MARGA' else u(0.75, 0.95)
        return {'Clay': c, 'Calcite': 1 - c}
    if name == 'DOLOMITO':
        d = u(0.7, 0.95)
        return {'Dolomite': d, 'Clay': 1 - d}
    c = u(0.7, 0.95)                                 # CALCARIO
    return {'Calcite': c, 'Clay': 1 - c}


def smooth(v, step, res_m=0.6):
    """Gaussian smoothing ~ tool vertical resolution."""
    s = res_m / 2.355 / step
    k = np.arange(-int(4 * s) - 1, int(4 * s) + 2)
    w = np.exp(-0.5 * (k / s) ** 2)
    w /= w.sum()
    pad = len(k) // 2
    return np.convolve(np.pad(v, pad, mode='edge'), w, mode='valid')


def make_well(rng, name, top, thick, step, sonic, shear, gr_off):
    base = top + thick
    beds = bed_sequence(rng, top, base)
    md = np.arange(top - MARGIN_TOP, base + MARGIN_BASE, step)
    n = len(md)
    logs = {k: np.zeros(n) for k in ('DT', 'DTS', 'GR', 'RHOB')}
    lito = np.full(n, np.nan)
    for z1, z2, lname, mix in beds:
        m = (md >= z1) & (md < z2)
        if not m.any():
            continue
        lito[m] = CODE[lname]
        # small within-bed variation of the mix
        jitter = 1 + rng.normal(0, 0.04, m.sum())
        for k in logs:
            main = max(mix, key=mix.get)
            vals = sum(f * MIN[mi][k] for mi, f in mix.items())
            alt = MIN[main][k]
            logs[k][m] = alt + (vals - alt) * jitter
    for k in logs:
        logs[k] = smooth(logs[k], step)
    logs['DT'] += rng.normal(0, 0.8, n)
    logs['DTS'] += rng.normal(0, 2.0, n)
    logs['GR'] += rng.normal(0, 4.0, n) + gr_off
    logs['GR'] = np.clip(logs['GR'], 0, None)
    logs['RHOB'] += rng.normal(0, 0.015, n)

    # a few washout-like bad density readings (removed by VALID_RANGES)
    for _ in range(3):
        i = rng.integers(0, n - 10)
        logs['RHOB'][i:i + rng.integers(2, 8)] = rng.uniform(1.0, 1.4)
    # a couple of null gaps in the sonic and GR
    for k in ('DT', 'GR'):
        i = rng.integers(0, n - 60)
        logs[k][i:i + rng.integers(10, 50)] = np.nan

    df = pd.DataFrame({'DEPT': md, 'LITO': lito, 'GR': logs['GR']})
    df[sonic] = logs['DT'] if sonic.startswith('DT') else 304800.0 / logs['DT']   # VP in m/s
    if shear:
        df[shear] = logs['DTS'] if shear.startswith('DT') else 304800.0 / logs['DTS']
    df['RHOB'] = logs['RHOB']
    return df


UNITS = {'DEPT': 'm', 'LITO': '', 'GR': 'gAPI', 'DT': 'us/ft', 'DTCO': 'us/ft', 'DTS': 'us/ft',
         'DTSM': 'us/ft', 'VP': 'm/s', 'VS': 'm/s', 'RHOB': 'g/cm3'}
DESCR = {'LITO': 'Lithology code (see rhob_common.LITO_TABLE)', 'GR': 'Gamma ray',
         'DT': 'Compressional slowness', 'DTCO': 'Compressional slowness',
         'DTS': 'Shear slowness', 'DTSM': 'Shear slowness', 'VP': 'P-wave velocity',
         'VS': 'S-wave velocity', 'RHOB': 'Bulk density'}


def write_las(df, path, well):
    las = lasio.LASFile()
    las.well['WELL'].value = well
    las.well['NULL'].value = NULL
    las.well['COMP'].value = 'SYNTHETIC DATA - make_sample_data.py'
    for c in df.columns:
        las.append_curve(c, df[c].fillna(NULL).to_numpy(float), unit=UNITS.get(c, ''),
                         descr=DESCR.get(c, 'Measured depth' if c == 'DEPT' else ''))
    las.write(str(path), version=2.0, fmt='%.4f')


def main():
    rng = np.random.default_rng(SEED)
    cfg = {'train_wells': [], 'predict_wells': [], 'salt_markers': {}}
    for sub in ('train', 'predict'):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for f in (OUT / sub).glob('*.las'):
            f.unlink()
    for name, top, thick, step, sonic, shear, gr_off, sub in WELLS:
        df = make_well(rng, name, top, thick, step, sonic, shear, gr_off)
        if name == 'SYN-PRED-2':                      # partial RHOB and partial VS
            md = df['DEPT']
            df.loc[md > top + 0.4 * thick, 'RHOB'] = np.nan
            df.loc[md > top + 0.5 * thick, shear] = np.nan
        if name == 'SYN-PRED-3':                      # no density at all
            df = df.drop(columns='RHOB')
        write_las(df, OUT / sub / f"{name}.las", name)
        cfg[f'{sub}_wells'].append(name)
        cfg['salt_markers'][name] = {'top': float(top), 'base': float(top + thick)}
        print(f"  {sub:7s} {name}: {len(df)} samples, step {step} m, curves {list(df.columns[1:])}")
    with open(OUT / 'sample_config.json', 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, indent=2)
    print(f"Written to {OUT}")


if __name__ == '__main__':
    main()
