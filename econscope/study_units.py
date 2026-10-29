"""Small *symbolic* dimension checker for the restricted scalar expression DSL.

Unit labels are supplied by users; this checker enforces their algebraic
consistency. It DOES NOT authenticate declared physical meanings, perform
conversions, estimate parameters, or validate the underlying theory.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import re

from .workspace import ScopeError

SYNTAX = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:[*/][A-Za-z][A-Za-z0-9_]*)*\Z")
BOOL = object()
ZERO = object()


def dimensions(unit):
    if unit in ("1", "dimensionless"):
        return ()
    if not isinstance(unit,str) or not SYNTAX.fullmatch(unit):
        raise ScopeError("STUDY_UNIT_SYNTAX_INVALID")
    out=defaultdict(int)
    pieces=re.split(r"([*/])",unit)
    sign=1
    for chunk in pieces:
        if chunk=="*": sign=1
        elif chunk=="/": sign=-1
        else: out[chunk]+=sign
    return tuple(sorted((k,v) for k,v in out.items() if v))


def combine(a,b,divide=False):
    if a is BOOL or b is BOOL:
        raise ScopeError("STUDY_UNIT_BOOLEAN_ARITHMETIC")
    # Bare zero can adopt a unit in additive contexts, but multiplication
    # and division must retain the other operand's symbolic dimensions.
    # Treat its multiplicative dimension as the dimensionless literal it is.
    if a is ZERO: a=()
    if b is ZERO: b=()
    res=defaultdict(int)
    for k,v in a: res[k]+=v
    for k,v in b: res[k]+=(-v if divide else v)
    return tuple(sorted((k,v) for k,v in res.items() if v))


def compatible(a,b):
    if a is BOOL or b is BOOL:
        raise ScopeError("STUDY_UNIT_BOOLEAN_NUMBER_MIXED")
    if a is ZERO: return b
    if b is ZERO: return a
    if a!=b:
        raise ScopeError("STUDY_DIMENSION_MISMATCH")
    return a


def infer(node,labels):
    if isinstance(node,ast.Expression):
        return infer(node.body,labels)
    if isinstance(node,ast.Constant):
        return ZERO if node.value==0 else ()
    if isinstance(node,ast.Name):
        if node.id not in labels:
            raise ScopeError("STUDY_UNIT_SYMBOL_UNKNOWN")
        return labels[node.id]
    if isinstance(node,ast.UnaryOp):
        val=infer(node.operand,labels)
        if val is BOOL:
            raise ScopeError("STUDY_UNIT_BOOLEAN_NUMBER_MIXED")
        return val
    if isinstance(node,ast.BinOp):
        a,b=infer(node.left,labels),infer(node.right,labels)
        if isinstance(node.op,(ast.Add,ast.Sub)):
            return compatible(a,b)
        if isinstance(node.op,ast.Mult): return combine(a,b)
        if isinstance(node.op,ast.Div):
            if b is ZERO:
                raise ScopeError("STUDY_DIVISION_BY_ZERO")
            return combine(a,b,True)
    if isinstance(node,ast.Compare):
        a=infer(node.left,labels)
        b=infer(node.comparators[0],labels)
        compatible(a,b)
        return BOOL
    if isinstance(node,ast.BoolOp):
        for v in node.values:
            if infer(v,labels) is not BOOL:
                raise ScopeError("STUDY_UNIT_EXPECTED_BOOLEAN")
        return BOOL
    if isinstance(node,ast.Call):
        dims=[infer(arg,labels) for arg in node.args]
        # All supported scalar functions require numbers, including unary
        # min/max calls. Do not let a comparison masquerade as a valid event
        # through a function that the evaluator would reject at runtime.
        if any(d is BOOL for d in dims):
            raise ScopeError("STUDY_UNIT_BOOLEAN_NUMBER_MIXED")
        if node.func.id=="abs":
            return dims[0]
        ref=dims[0]
        for d in dims[1:]:
            ref=compatible(ref,d)
        return ref
    raise ScopeError("STUDY_UNIT_EXPRESSION_UNSUPPORTED")


def audit_expression(parsed_tree, unit_map, expected_unit=None, event=False):
    labels={k:dimensions(v) for k,v in unit_map.items()}
    actual=infer(parsed_tree,labels)
    if event:
        if actual is not BOOL:
            raise ScopeError("STUDY_EVENT_UNIT_MUST_BE_BOOLEAN")
        return
    if actual is BOOL:
        raise ScopeError("STUDY_RULE_UNIT_MUST_BE_NUMERIC")
    expected=dimensions(expected_unit)
    if actual is ZERO:
        return
    if actual!=expected:
        raise ScopeError("STUDY_TRANSITION_UNIT_MISMATCH")
