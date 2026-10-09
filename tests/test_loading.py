import pathlib

import pytest

import el_hd_plasticity.two_photon as tp
from el_hd_plasticity.two_photon import loading

ROW = {'date': '20240101', 'fly': 'f1', 'sess': 's1', 'session': 'sess1', 'condition': 'cond', 'rnai': 0,
       'basedir': '/other/base'}
BASE = pathlib.Path('/data/base')


@pytest.mark.parametrize('kwargs, expected', [
    ({}, '/data/base/20240101/f1/s1/preprocess.pkl'),
    ({'session_col': 'session'}, '/data/base/20240101/f1/sess1/preprocess.pkl'),
    ({'folder_cols': ('condition',)}, '/data/base/cond/20240101/f1/s1/preprocess.pkl'),
    ({'folder_cols': ('rnai',), 'value_maps': {'rnai': {0: 'control', 1: 'rnai'}}},
     '/data/base/control/20240101/f1/s1/preprocess.pkl'),
])
def test_session_path_layouts(kwargs, expected):
    assert str(loading.session_path(ROW, basedir=BASE, **kwargs)) == expected


def test_session_path_basedir_from_row():
    p = loading.session_path(ROW, basedir_col='basedir', folder_cols=('condition',))
    assert str(p) == '/other/base/cond/20240101/f1/s1/preprocess.pkl'


def test_make_load_row_loads_the_expected_file(monkeypatch):
    seen = []
    monkeypatch.setattr(loading.st2p.preprocessing.EBImagingSession, 'from_file',
                        classmethod(lambda cls, p: seen.append(str(p)) or 'session'))
    assert tp.loading.make_load_row(BASE, folder_cols=('condition',))(ROW) == 'session'
    assert seen == ['/data/base/cond/20240101/f1/s1/preprocess.pkl']


def test_sessmaker_filters_unusable_sessions_and_builds_ids(tmp_path):
    (tmp_path / 'sessions.csv').write_text('date,fly,session,usable,closed_loop\n'
                                           '2024_01_01,f1,s1,1,1\n2024_01_01,f2,s1,0,1\n')
    sm = loading.SessMaker(tmp_path, tmp_path / 'figs', id_suffix='da')
    assert list(sm.fly_ids) == ['2024_01_01_f1_da']
    assert (tmp_path / 'figs').is_dir()
    assert str(loading.session_path(sm.sess_df.iloc[0], basedir=sm.basedir, session_col='session')) \
        == f'{tmp_path}/2024_01_01/f1/s1/preprocess.pkl'
