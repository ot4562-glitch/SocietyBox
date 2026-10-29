"""Read installed distribution metadata; never import optional solver code.

This is a prerequisite diagnostic, not a dependency resolver, license audit,
adapter, isolation boundary, or solver execution/parity check.
"""
from importlib import metadata


def macro_prerequisites():
    """Common optional contract; no solver imports, executable probes or rights."""
    import shutil

    candidates = []
    definitions = (
        ('HARK', 'econ-ark', '4c285e863cd1a60eb2a9c4ab335c7f86ea5aa767',
         'Apache-2.0', 'HARK.ConsumptionSaving.ConsIndShockModel.solve_one_period_ConsPF',
         ('numpy', 'scipy', 'numba', 'econ-ark')),
        ('Dolo', 'dolo', '33f873eef42d9644b5a0705a63fb3a34de5c2825',
         'BSD-2-Clause', 'dolo.yaml_import; dolo.algos.commands',
         ('numpy', 'scipy', 'numba', 'dolang', 'dynopy', 'interpolation', 'jax', 'dolo')),
        ('Dynare', None, '7.2:88204354739547117b315e8fa3f8780a7570c576efd12f6a292f2a2468bb7a3c',
         'GPL-3.0-or-later_WITH_FILE_EXCEPTIONS', 'dynare fixed-fixture.mod via Octave', ()),
    )
    for name, distribution, pin, license_id, api, packages in definitions:
        dependencies, blockers = [], []
        for package in packages:
            try:
                version = metadata.version(package)
                if type(version) is not str or not version.strip():
                    raise ValueError('invalid version')
                state = 'METADATA_FOUND'
            except metadata.PackageNotFoundError:
                version, state = None, 'NOT_INSTALLED'
            except (OSError, ValueError, UnicodeError):
                version, state = None, 'UNKNOWN'
            dependencies.append({'distribution': package, 'version': version, 'state': state})
            if state != 'METADATA_FOUND':
                blockers.append('DEPENDENCY_' + state + ':' + package)
        backend = None
        if name == 'Dynare':
            backend = 'PATH_CANDIDATE_UNVERIFIED' if (shutil.which('octave-cli') or
                       shutil.which('octave')) else 'NOT_FOUND_ON_PATH'
            blockers.extend(['MATH_BACKEND_NOT_VERIFIED', 'DYNARE_RUNTIME_NOT_VERIFIED',
                             'GPL_FILE_LEVEL_AND_PROCESS_BOUNDARY_REVIEW_REQUIRED'])
        blockers.extend(['DEPENDENCY_PROVENANCE_ABI_NOT_ATTESTED',
                         'TRANSITIVE_LICENSE_REVIEW_REQUIRED',
                         'MODEL_MAPPING_NOT_IMPLEMENTED',
                         'TRUSTED_BOUNDED_WORKER_NOT_IMPLEMENTED'])
        candidates.append({'engine': name, 'distribution': distribution, 'source_pin': pin,
                           'reference_license': license_id, 'original_api_candidate': api,
                           'dependencies': dependencies, 'math_backend': backend,
                           'status': 'BLOCKED', 'adapter': 'PREPARED_BLOCKED',
                           'native_execution': 'NOT_RUN', 'native_parity': 'NOT_RUN',
                           'redistribution': 'NOT_GRANTED', 'blockers': blockers})
    return {'status': 'BLOCKED', 'candidates': candidates,
            'discovery_scope': 'METADATA_AND_PATH_ONLY_NO_IMPORT_OR_EXECUTION',
            'dependency_scope': 'SELECTED_REQUIRED_PACKAGES_NOT_COMPLETE_CLOSURE',
            'generated_code': 'BLOCKED', 'os_isolation': 'NOT_ATTESTED',
            'license_basis': 'Local pinned root texts; not redistribution clearance',
            'process_boundary_note': 'Out-of-process alone does not settle GPL obligations or establish OS isolation.'}


def quantecon_stationary_prerequisites():
    """Optional stationary API contract; metadata discovery grants no import rights."""
    from importlib.util import find_spec

    dependencies = []
    blockers = []
    for name in ('numpy', 'scipy', 'numba', 'llvmlite', 'quantecon'):
        try:
            spec = find_spec(name)
            origin = None if spec is None else spec.origin
        except (ImportError, ValueError, OSError):
            origin = None
        try:
            version = metadata.version(name)
        except (metadata.PackageNotFoundError, ValueError, OSError):
            version = None
        dependencies.append({'name': name, 'origin': origin, 'version': version})
        if origin is None:
            blockers.append('DEPENDENCY_NOT_IMPORTABLE:' + name)
    blockers.extend(['DEPENDENCY_PROVENANCE_NOT_ATTESTED',
                     'TRUSTED_QUANTECON_WORKER_NOT_IMPLEMENTED'])
    return {
        'status': 'BLOCKED', 'adapter': 'PREPARED_BLOCKED',
        'original_api': 'quantecon.MarkovChain(P).stationary_distributions',
        'reference_commit': '8c85f1b2bdb8410b50093cc9ed0bc2c2155334e9',
        'reference_version': '0.13.1', 'reference_license': 'MIT',
        'license_basis': 'Pinned local reference LICENSE; not redistribution clearance',
        'dependencies': dependencies, 'blockers': blockers,
        'native_execution': 'NOT_RUN', 'native_parity': 'NOT_RUN',
        'os_isolation': 'NOT_ATTESTED',
        'discovery_scope': 'MODULE_SPEC_AND_METADATA_ONLY_NO_IMPORT_TEST',
    }


def isolation_prerequisites():
    """Static discovery only: a canary receipt is not an execution credential."""
    return {
        "status": "BLOCKED", "generated_code": "BLOCKED",
        "trusted_native": "SEPARATE_LICENSE_AND_PARITY_REVIEW_REQUIRED",
        "canary_evidence": "NOT_ATTESTED",
        "network_enforcement": "NOT_ESTABLISHED",
        "blockers": ["OS_ISOLATION_NOT_ESTABLISHED",
                     "STRICT_READ_WRITE_NETWORK_RESOURCE_BOUNDARIES_REQUIRED"],
        "limitations": [
            "Codex restricted-token diagnostics do not establish MXC equivalence or accreditation.",
            "Unelevated read restrictions and subprocess timeouts are not a complete OS sandbox.",
            "Fixed synthetic canaries never authorize generated or researcher-supplied code.",
        ],
    }


def native_prerequisites():
    candidates = []
    for name, distribution in (("Mesa", "mesa"), ("QuantEcon.py", "quantecon"),
                               ("HARK", "econ-ark")):
        blockers = []
        version = None
        try:
            version = metadata.distribution(distribution).metadata.get("Version")
            if not isinstance(version, str) or not version.strip():
                version = None
                installation = "UNKNOWN"
                blockers.append("DISTRIBUTION_METADATA_INVALID")
            else:
                installation = "METADATA_FOUND"
        except metadata.PackageNotFoundError:
            installation = "NOT_INSTALLED"
            blockers.append("DISTRIBUTION_NOT_INSTALLED")
        except (OSError, UnicodeError, ValueError):
            installation = "UNKNOWN"
            blockers.append("DISTRIBUTION_METADATA_UNREADABLE")
        blockers.extend(["LICENSE_REVIEW_REQUIRED", "ADAPTER_NOT_IMPLEMENTED",
                         "NATIVE_REFERENCE_PARITY_NOT_RUN"])
        candidates.append({
            "name": name, "distribution": distribution, "version": version,
            "installation": installation, "adapter": "NOT_IMPLEMENTED",
            "execution": "NOT_RUN", "parity": "NOT_RUN", "blockers": blockers,
        })
    return {
        "scope": "ACTIVE_INTERPRETER_DISTRIBUTION_METADATA_ONLY",
        "status": "BLOCKED", "candidates": candidates,
        "limitations": [
            "Package metadata does not prove importability, dependency compatibility or license rights.",
            "No candidate solver was imported or executed; no upstream parity was measured.",
            "Reference-only repositories are not installed adapters; no fallback model is selected.",
            "Arbitrary generated code still requires a separately proven OS isolation boundary.",
        ],
    }
