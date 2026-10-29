"""Fail-closed parser for researcher-declared JSON contracts.

JSON permits ambiguous duplicate object members in some decoders. Silent
last-key-wins parsing can change the economic model actually executed; refuse
such inputs consistently in independent CLI boundaries.
"""
import json


def loads_strict(raw):
    """Read a JSON value while rejecting repeated keys at every nesting level."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("RESEARCH_JSON_DUPLICATE_KEY")
            result[key] = value
        return result

    def finite_only(_token):
        raise ValueError("RESEARCH_JSON_NONFINITE_NUMBER")

    return json.loads(raw, object_pairs_hook=unique, parse_constant=finite_only)
