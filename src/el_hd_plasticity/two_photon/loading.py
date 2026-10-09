"""Locating and loading preprocessed imaging sessions from disk.

Sessions are stored as ``<basedir>/[<folder columns>/]<date>/<fly>/<session>/preprocess.pkl`` and
described by a ``sessions.csv`` with one row per session. The layout differs slightly between
experiments (extra condition/genotype folders, ``sess`` vs ``session`` column), which
:func:`make_load_row` captures in a few arguments so every notebook can build its own
``load_row(row)`` without redefining it.
"""

import pathlib

import pandas as pd
import SessionTools.two_photon as st2p

PREPROCESS_FILENAME = 'preprocess.pkl'


def session_path(row, basedir=None, folder_cols=(), session_col='sess', basedir_col=None, value_maps=None):
    """Path to the preprocessed pickle for one session row.

    Args:
        row: Row (Series or dict) of a sessions DataFrame.
        basedir: Base directory containing all sessions. Ignored if `basedir_col` is given.
        folder_cols: Row columns whose values are intermediate folders between `basedir`
            and the date folder, in order (e.g. ``('condition',)``).
        session_col: Row column holding the session folder name.
        basedir_col: If given, take the base directory from this row column instead of `basedir`
            (for tables that pool sessions from several base directories).
        value_maps: Optional ``{column: {value: folder_name}}`` for columns whose values are not
            folder names themselves (e.g. ``{'rnai': {0: 'control', 1: 'rnai'}}``).

    Returns:
        pathlib.Path to ``preprocess.pkl``.
    """
    value_maps = value_maps or {}
    root = pathlib.Path(row[basedir_col]) if basedir_col is not None else pathlib.Path(basedir)
    for col in folder_cols:
        value = row[col]
        root = root / value_maps[col][value] if col in value_maps else root / value
    return root / row['date'] / row['fly'] / row[session_col] / PREPROCESS_FILENAME


def load_session(path):
    """Load a preprocessed session pickle as an ``EBImagingSession``."""
    return st2p.preprocessing.EBImagingSession.from_file(pathlib.PurePath(path))


def make_load_row(basedir=None, **path_kwargs):
    """Build a ``load_row(row)`` function for a sessions table.

    Args:
        basedir: Base directory containing the sessions (see :func:`session_path`).
        **path_kwargs: ``folder_cols``, ``session_col``, ``basedir_col`` and ``value_maps``,
            forwarded to :func:`session_path`.

    Returns:
        Function mapping a session row to its loaded ``EBImagingSession``.

    Example:
        >>> load_row = make_load_row(basedir, folder_cols=('condition',))
        >>> ts = GetTS(load_row(sess_df.iloc[0]))
    """
    def load_row(row):
        """Load the preprocessed session of one row of the sessions table."""
        return load_session(session_path(row, basedir=basedir, **path_kwargs))
    return load_row


class SessMaker:
    """Manage imaging sessions listed in ``<basedir>/sessions.csv`` and provide data access.

    Loads session metadata, filters usable sessions, and provides methods to
    access preprocessed imaging data and session properties.

    Attributes:
        basedir: Path to base directory containing session data.
        sess_df: DataFrame with session metadata.
        fig_folder: Path where figures will be saved.
    """

    def __init__(self, basedir=None, figfolder=None, mkdir=True, id_suffix=None):
        """Initialize session manager.

        Args:
            basedir: Path to base directory containing sessions.csv.
            figfolder: Path where output figures will be saved.
            mkdir: If True, create figfolder if it doesn't exist.
            id_suffix: Optional suffix to append to fly_id.
        """
        self.basedir = pathlib.Path(basedir)
        self.sess_df = pd.read_csv(self.basedir / 'sessions.csv')
        self._id_suffix = id_suffix
        self.fig_folder = pathlib.Path(figfolder)
        if mkdir:
            self.fig_folder.mkdir(parents=True, exist_ok=True)

        self.add_fly_id()
        self.filter_usable()

    def add_fly_id(self):
        """Create unique fly identifiers from date, fly, and optional suffix."""
        fly_id = []
        if self._id_suffix is None:
            for _, row in self.sess_df.iterrows():
                fly_id.append(row['date'] + '_' + row['fly'])
        else:
            for _, row in self.sess_df.iterrows():
                fly_id.append(row['date'] + '_' + row['fly'] + '_' + self._id_suffix)
            self.sess_df['grab'] = self._id_suffix

        self.sess_df['fly_id'] = fly_id

    def filter_usable(self):
        """Filter to only usable sessions (usable==1)."""
        self.sess_df = self.sess_df.loc[self.sess_df['usable'] == 1]

    def load_row(self, row):
        """Load preprocessed imaging data for a session row.

        Args:
            row: Session row from sess_df (needs ``date``, ``fly`` and ``session``).

        Returns:
            EBImagingSession object with preprocessed data.
        """
        return load_session(session_path(row, basedir=self.basedir, session_col='session'))

    @property
    def fly_ids(self):
        """Return array of unique fly identifiers."""
        return self.sess_df['fly_id'].unique()
