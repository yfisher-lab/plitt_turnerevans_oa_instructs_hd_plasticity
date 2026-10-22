"""Defaults for connecting to neuPrint (server dataset and authentication token).

The token is read from the ``NEUPRINT_APPLICATION_CREDENTIALS`` environment variable (the one the neuPrint
tools use) when it is set.
"""

import os

NPT_TOKEN = os.environ.get('NEUPRINT_APPLICATION_CREDENTIALS')
NPT_DATASET = 'hemibrain:v1.2.1'

CODEX_TOKEN = os.environ.get('CODEX_API_TOKEN')
