"""Read-only prerequisite manifest for a single candidate, never an adapter.

Only five explicitly named reference files are read, as bounded inert bytes.
Matching their recorded hashes does not authenticate the complete source tree.
Optional distribution metadata is inspected without importing solver modules.
"""
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import re
import sys


_MAX_FILE_BYTES = 512 * 1024
_PINNED_FILES = {
    'pyproject.toml': 'bb53147d4a84362b10844296c47e495e7b236f4ac06f6392535670ab166ca5fe',
    'LICENSE': 'a8cb02bae136d4e96c55f1693e4d714a8c90c1b5091569bda43faa6988cdb368',
    'quantecon/__init__.py': 'e50be547efabb6e9e4304e75715bff2ae12c4d32db9ff5172113668eb14c1f0d',
    'quantecon/_kalman.py': 'fad7f054056ff5500580a72b34fbfd04a7610578dd5ec6a3eff4c8c621af7d4f',
    'quantecon/_lss.py': '6565d47bf9fa565b0f9ef1acae6b526dd9be733718d68315c8f4d3242b27d541',
}


def _reference_files(root):
    if root is None:
        return {'status': 'NOT_INSPECTED', 'files': []}
    rows = []
    for relative, expected in _PINNED_FILES.items():
        digest = None
        try:
            path = Path(root) / relative
            with path.open('rb') as stream:
                data = stream.read(_MAX_FILE_BYTES + 1)
            if len(data) > _MAX_FILE_BYTES:
                status = 'SIZE_LIMIT'
            else:
                digest = hashlib.sha256(data).hexdigest()
                status = 'MATCH' if digest == expected else 'HASH_MISMATCH'
        except (OSError, ValueError, TypeError):
            status = 'MISSING_OR_UNREADABLE'
        rows.append({'file': relative, 'expected_sha256': expected,
                     'sha256': digest, 'status': status})
    return {'status': 'PINNED_FILE_HASHES_MATCH' if all(
        row['status'] == 'MATCH' for row in rows) else 'BLOCKED', 'files': rows}


def time_series_prerequisites(reference_root=None):
    """Describe blockers for pinned QuantEcon Kalman work without execution.

No root means no reference reads. A supplied root is never added to sys.path.
There is deliberately no caller-provided approval/isolation/parity assertion.
"""
    reference = _reference_files(reference_root)
    blockers = []
    python_minimum_met = sys.version_info[:2] >= (3, 12)
    if not python_minimum_met:
        blockers.append('REFERENCE_PYTHON_MINIMUM_NOT_MET')
    if reference['status'] != 'PINNED_FILE_HASHES_MATCH':
        blockers.append('REFERENCE_FILES_NOT_VERIFIED')
    rows = []
    for name in ('quantecon', 'numpy', 'scipy', 'numba', 'llvmlite'):
        version = None
        try:
            value = metadata.distribution(name).metadata.get('Version')
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.!+_-]{0,127}', value):
                installation = 'UNKNOWN'
                blockers.append('DISTRIBUTION_METADATA_INVALID:' + name)
            else:
                version = value
                installation = 'METADATA_FOUND'
        except metadata.PackageNotFoundError:
            installation = 'NOT_INSTALLED'
            blockers.append('DISTRIBUTION_NOT_INSTALLED:' + name)
        except (OSError, UnicodeError, ValueError):
            installation = 'UNKNOWN'
            blockers.append('DISTRIBUTION_METADATA_UNREADABLE:' + name)
        row = {'distribution': name, 'installation': installation, 'version': version}
        if name == 'quantecon':
            row['version_matches_reference'] = None if version is None else version == '0.13.1'
            if version is not None and version != '0.13.1':
                blockers.append('REFERENCE_VERSION_MISMATCH:quantecon')
        rows.append(row)
    blockers.extend(['DEPENDENCY_COMPATIBILITY_NOT_ESTABLISHED', 'LICENSE_REVIEW_REQUIRED',
                     'OS_ISOLATION_NOT_ESTABLISHED', 'ADAPTER_NOT_IMPLEMENTED',
                     'NATIVE_REFERENCE_PARITY_NOT_RUN'])
    return {
        'schema': 'societybox-time-series-prerequisites-v1',
        'scope': 'READ_ONLY_PREREQUISITE_MANIFEST_ONLY', 'status': 'BLOCKED',
        'candidate': {
            'source_id': 'U-QE', 'distribution': 'quantecon', 'reference_version': '0.13.1',
            'recorded_commit': '8c85f1b2bdb8410b50093cc9ed0bc2c2155334e9',
            'commit_authentication': 'NOT_ESTABLISHED',
            'entrypoints': ['Kalman.prior_to_filtered', 'Kalman.filtered_to_forecast', 'Kalman.update'],
            'timing': 'prior_to_filtered conditions on current y; update also advances to the next prior',
        },
        'reference': reference,
        'runtime': {
            'python': platform.python_version(), 'implementation': platform.python_implementation(),
            'reference_requires_python': '>=3.12',
            'python_minimum_met': python_minimum_met,
            'reference_direct_requirements': ['numba>=0.56.0', 'numpy>=1.17.0', 'scipy>=1.5.0'],
            'additional_dependency_probe': 'llvmlite (numba dependency; not a full environment lock)',
            'distributions': rows, 'dependency_compatibility': 'NOT_ESTABLISHED',
        },
        'licensing': {'reference_license_label': 'MIT',
                      'label_basis': 'Recorded local LICENSE; only bound when its file hash matches',
                      'dependency_license_review': 'NOT_ESTABLISHED',
                      'redistribution_review': 'NOT_ESTABLISHED'},
        'adapter': 'NOT_IMPLEMENTED', 'isolation': 'NOT_ESTABLISHED',
        'execution': 'NOT_RUN', 'parity': 'NOT_RUN', 'authorization': 'NOT_GRANTED',
        'blockers': blockers,
        'limitations': [
            'File hashes bind five inspected files, not the whole source, origin authenticity or rights.',
            'Distribution metadata does not prove importability or binary/dependency compatibility.',
            'No solver import, source evaluation, installation, network, model or reference execution.',
            'No estimation, empirical forecast skill, historical-as-of or causal identification established.',
            'One candidate does not replace the eight-upstream capability inventory.',
        ],
    }
