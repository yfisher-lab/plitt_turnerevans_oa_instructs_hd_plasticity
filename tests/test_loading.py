import pathlib

import pytest

import el_hd_plasticity.two_photon as tp
from el_hd_plasticity.two_photon import loading

ROW = {'date': '20240101', 'fly': 'f1', 'session': 's1', 'condition': 'cond', 'genotype': 'ctrl', 'rnai': 0,
       'basedir': '/other/base'}
BASE = pathlib.Path('/data/base')


@pytest.mark.parametrize('kwargs, expected', [
    ({}, '/data/base/20240101/f1/s1/preprocess.pkl'),
    ({'folder_cols': ('condition',)}, '/data/base/cond/20240101/f1/s1/preprocess.pkl'),
    ({'folder_cols': ('condition', 'genotype')}, '/data/base/cond/ctrl/20240101/f1/s1/preprocess.pkl'),
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


CSV = ('date,fly,session,usable,closed_loop,condition,genotype\n'
       '2024_01_01,f1,s1,1,1,remap,ctrl\n'
       '2024_01_01,f2,s1,0,1,remap,exp\n'
       '2024_01_02,f3,s1,1,0,no_chrimson,exp\n')


def make_sessmaker(tmp_path, **kwargs):
    (tmp_path / 'sessions.csv').write_text(CSV)
    return loading.SessionManager(tmp_path, figfolder=tmp_path / 'figs', **kwargs)


def test_sessmaker_filters_unusable_sessions_and_creates_figfolder(tmp_path):
    sm = make_sessmaker(tmp_path)
    assert list(sm.fly_ids) == ['2024_01_01_f1', '2024_01_02_f3']
    assert (tmp_path / 'figs').is_dir()
    assert str(loading.session_path(sm.sess_df.iloc[0], basedir=sm.basedir)) \
        == f'{tmp_path}/2024_01_01/f1/s1/preprocess.pkl'


def test_id_cols_append_column_name_and_value_in_order(tmp_path):
    sm = make_sessmaker(tmp_path, id_cols=['condition', 'genotype'])
    assert list(sm.fly_ids) == ['2024_01_01_f1_condition_remap_genotype_ctrl',
                                '2024_01_02_f3_condition_no_chrimson_genotype_exp']


def test_id_cols_keep_flies_with_the_same_date_and_name_apart(tmp_path):
    (tmp_path / 'sessions.csv').write_text('date,fly,session,usable,condition\n'
                                           '2024_01_01,f1,s1,1,a\n2024_01_01,f1,s2,1,b\n')
    sm = loading.SessionManager(tmp_path, figfolder=tmp_path / 'figs', id_cols=['condition'])
    assert sorted(sm.fly_ids) == ['2024_01_01_f1_condition_a', '2024_01_01_f1_condition_b']


def test_extra_cols_add_constant_columns_usable_in_ids(tmp_path):
    sm = make_sessmaker(tmp_path, id_cols=['grab'], extra_cols={'grab': 'DA'})
    assert (sm.sess_df['grab'] == 'DA').all()
    assert list(sm.fly_ids) == ['2024_01_01_f1_grab_DA', '2024_01_02_f3_grab_DA']


def test_missing_id_column_raises_a_clear_error(tmp_path):
    with pytest.raises(KeyError, match='nonexistent'):
        make_sessmaker(tmp_path, id_cols=['nonexistent'])


def test_numeric_dates_and_values_are_stringified(tmp_path):
    (tmp_path / 'sessions.csv').write_text('date,fly,session,usable,dark\n20240101,7,s1,1,0\n')
    sm = loading.SessionManager(tmp_path, figfolder=tmp_path / 'figs', id_cols=['dark'])
    assert list(sm.fly_ids) == ['20240101_7_dark_0']


def test_session_manager_aliases_a_sess_column_to_session(tmp_path):
    import pandas as pd
    pd.DataFrame({'date': ['d'], 'fly': ['f'], 'sess': ['s-000'], 'closed_loop': [1], 'usable': [1]}).to_csv(
        tmp_path / 'sessions.csv', index=False)
    manager = tp.loading.SessionManager(basedir=tmp_path)
    assert manager.sess_df['session'].tolist() == ['s-000']
