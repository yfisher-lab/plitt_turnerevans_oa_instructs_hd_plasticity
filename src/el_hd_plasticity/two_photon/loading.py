"""Locating and loading preprocessed imaging sessions from disk.

Sessions are stored as ``<basedir>/[<folder columns>/]<date>/<fly>/<session>/preprocess.pkl`` and
described by a ``sessions.csv`` with one row per session. Every ``sessions.csv`` must have the columns
``date``, ``fly`` and ``session`` (the folder names at those levels). Experiments differ only in extra
intermediate folders (condition, genotype, ...), which :func:`make_load_row` captures in a few arguments
so every notebook can build its own ``load_row(row)`` without redefining it.
"""

import pathlib

import pandas as pd
import SessionTools.two_photon as st2p

PREPROCESS_FILENAME = 'preprocess.pkl'


def session_path(row, basedir=None, folder_cols=(), basedir_col=None):
    """Path to the preprocessed pickle for one session row.

    Args:
        row: Row (Series or dict) of a sessions DataFrame with ``date``, ``fly`` and ``session`` columns.
        basedir: Base directory containing all sessions. Ignored if `basedir_col` is given.
        folder_cols: Row columns whose values are intermediate folders between `basedir`
            and the date folder, in order (e.g. ``('condition',)``).
        basedir_col: If given, take the base directory from this row column instead of `basedir`
            (for tables that pool sessions from several base directories).

    Returns:
        pathlib.Path to ``preprocess.pkl``.
    """
    
    root = pathlib.Path(row[basedir_col]) if basedir_col is not None else pathlib.Path(basedir)
    for col in folder_cols:
        value = row[col]
        root = root / value
    return root / row['date'] / row['fly'] / row['session'] / PREPROCESS_FILENAME


def load_session(path):
    """Load a preprocessed session pickle as an ``EBImagingSession``."""
    return st2p.preprocessing.EBImagingSession.from_file(pathlib.PurePath(path))


def make_load_row(basedir=None, **path_kwargs):
    """Build a ``load_row(row)`` function for a sessions table.

    Args:
        basedir: Base directory containing the sessions (see :func:`session_path`).
        **path_kwargs: ``folder_cols`` and ``basedir_col``,
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


class SessionManager:
    """Manage imaging sessions listed in ``<basedir>/sessions.csv`` and provide data access.

    Loads session metadata, filters usable sessions, and provides methods to
    access preprocessed imaging data and session properties.

    Attributes:
        basedir: Path to base directory containing session data.
        sess_df: DataFrame with session metadata.
        fig_folder: Path where figures will be saved.
    """

    def __init__(
            self, 
            basedir=None, 
            folder_cols=(),
            basedir_col=None,
            figfolder=None, 
            mkdir=True, 
            id_cols=None, 
            extra_cols=None):
        """Initialize session manager.

        Args:
            basedir: Path to base directory containing sessions.csv.
            folder_cols: Optional tuple of session-table columns whose values are intermediate folders
                between `basedir` and the date folder, in order (e.g. ``('condition',)``).
            basedir_col: Optional session-table column whose value is the base directory for that row, instead of `basedir` (for tables that pool sessions from several base directories
            figfolder: Path where output figures will be saved.
            mkdir: If True, create figfolder if it doesn't exist.
            id_cols: Optional list of columns of sessions.csv (or of `extra_cols`) whose name and
                value are appended to ``fly_id``, in order. With ``id_cols=['condition']`` a fly
                recorded on 2024_01_01 with condition 'remap' gets the id
                ``2024_01_01_f1_condition_remap``. Use this to keep flies from different
                conditions or experiments distinct when their tables are pooled.
            extra_cols: Optional dict of constant columns to add to the session table, e.g.
                ``{'grab': 'DA'}`` to label every session of a dataset; they can be used in `id_cols`.
        """
        self.basedir = pathlib.Path(basedir)
        self._sess_df = pd.read_csv(self.basedir / 'sessions.csv')
        for col, value in (extra_cols or {}).items():
            self.sess_df[col] = value

        self._folder_cols = folder_cols
        self._basedir_col = basedir_col
        self._id_cols = list(id_cols) if id_cols is not None else []

        if figfolder is not None:
            self.fig_folder = pathlib.Path(figfolder)
            if mkdir:
                self.fig_folder.mkdir(parents=True, exist_ok=True)

        self.add_fly_id()
        self.filter_usable()
        self.load_row = make_load_row(basedir=self.basedir, folder_cols=folder_cols)
        

    def add_fly_id(self):
        """Create unique fly identifiers: ``<date>_<fly>`` plus ``_<column>_<value>`` for each of `id_cols`."""
        missing = [col for col in self._id_cols if col not in self.sess_df.columns]
        if missing:
            raise KeyError(f'id_cols {missing} not in sessions.csv columns {list(self.sess_df.columns)}')

        fly_id = self.sess_df['date'].astype(str) + '_' + self.sess_df['fly'].astype(str)
        for col in self._id_cols:
            fly_id = fly_id + f'_{col}_' + self.sess_df[col].astype(str)
        self.sess_df['fly_id'] = fly_id

    def filter_usable(self):
        """Filter to only usable sessions (usable==1)."""
        self._sess_df = self._sess_df.loc[self._sess_df['usable'] == 1]

    def filter_rows(self, filter_mask):
        """Filter sessions by a boolean mask.

        Args:
            filter_mask: Boolean mask for filtering sessions.
        """
        self._sess_df = self._sess_df.loc[filter_mask]

    @property
    def sess_df(self):
        """Return the session DataFrame"""
        return self._sess_df

    @property
    def fly_ids(self):
        """Return array of unique fly identifiers."""
        return self._sess_df['fly_id'].unique()

    def iterrows(self):
        """Iterate over sessions as (index, row) pairs; alias for ``sess_df.iterrows()``."""
        return self.sess_df.iterrows()


