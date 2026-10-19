#!/usr/bin/env python3
"""Stage the two source collections into work/ for 01_curate_and_split.py.

BBBP is shipped with this repository as data/BBBP_source.csv, the exact file used
for the published analysis (MD5 f6017522e99b22697469c4d94af9a919); it originates
from the MoleculeNet benchmark suite and is copied into work/ unchanged.

B3DB is downloaded from the authors' repository. Pass --b3db-path to use a local
copy instead if this machine has no network access.
"""
import argparse, hashlib, os, shutil, sys, urllib.request

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(_ROOT, 'work') + os.sep
DATA = os.path.join(_ROOT, 'data') + os.sep

BBBP_MD5 = 'f6017522e99b22697469c4d94af9a919'
B3DB_URL = ('https://raw.githubusercontent.com/theochem/B3DB/main/'
            'B3DB/B3DB_classification.tsv')


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--b3db-path', help='local B3DB_classification.tsv instead of downloading')
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)

    src = DATA + 'BBBP_source.csv'
    if not os.path.exists(src):
        sys.exit('missing %s — re-clone the repository' % src)
    shutil.copyfile(src, WORK + 'BBBP.csv')
    got = md5(WORK + 'BBBP.csv')
    print('BBBP.csv  -> work/  md5 %s  %s' % (got, 'ok' if got == BBBP_MD5 else 'MISMATCH'))
    if got != BBBP_MD5:
        sys.exit('BBBP.csv does not match the published input; results will not reproduce')

    dst = WORK + 'b3.tsv'
    if args.b3db_path:
        shutil.copyfile(args.b3db_path, dst)
        print('b3.tsv    -> work/  copied from %s' % args.b3db_path)
    else:
        print('downloading B3DB from %s' % B3DB_URL)
        urllib.request.urlretrieve(B3DB_URL, dst)
        print('b3.tsv    -> work/  %d bytes' % os.path.getsize(dst))
    print('\nB3DB is a living database: if the upstream file has changed since '
          'publication, partition sizes may differ. The external set exactly as used '
          'is shipped at data/external_B3DB_filtered.csv.')


if __name__ == '__main__':
    main()
