"""Bounded comma/quote CSV input independent of process-global csv settings."""
import re

from .workspace import ScopeError


# Match the standard Python parser's default, but never inherit or mutate it.
FIELD_CHAR_LIMIT = 131072
# 4 MiB of comma-only text must not allocate millions of empty columns.
MAX_RECORD_FIELDS = 1024
_UNQUOTED = re.compile(r'[^,\r\n]*')


def rows(text):
    """Yield strict comma/quote records; caller enforces file and row bounds.

    Preserve embedded CR/LF and doubled quotes. As with the standard dialect,
    quotes in an unquoted field are ordinary text; only a leading quote opens
    a quoted field. Empty physical lines yield empty records.
    """
    cursor = 0
    size = len(text)
    while cursor < size:
        record = []
        if text[cursor] in '\r\n':
            end = cursor
        else:
            while True:
                # Bound construction of the current row, including a trailing
                # empty field. The user-visible error is intentionally distinct
                # from malformed quoting and does not expose input cell data.
                if len(record) >= MAX_RECORD_FIELDS:
                    raise ScopeError('STUDY_CSV_COLUMN_LIMIT')
                if cursor < size and text[cursor] == '"':
                    cursor += 1
                    pieces = []
                    length = 0
                    while True:
                        end = text.find('"', cursor)
                        if end < 0:
                            raise ScopeError('STUDY_CSV_PARSE_ERROR')
                        length += end - cursor
                        if length > FIELD_CHAR_LIMIT:
                            raise ScopeError('STUDY_CSV_PARSE_ERROR')
                        pieces.append(text[cursor:end])
                        cursor = end + 1
                        if cursor < size and text[cursor] == '"':
                            pieces.append('"')
                            length += 1
                            if length > FIELD_CHAR_LIMIT:
                                raise ScopeError('STUDY_CSV_PARSE_ERROR')
                            cursor += 1
                        else:
                            break
                    value = ''.join(pieces)
                    end = cursor
                    if end < size and text[end] not in ',\r\n':
                        raise ScopeError('STUDY_CSV_PARSE_ERROR')
                else:
                    match = _UNQUOTED.match(text, cursor)
                    end = match.end()
                    if end - cursor > FIELD_CHAR_LIMIT:
                        raise ScopeError('STUDY_CSV_PARSE_ERROR')
                    value = match.group()
                record.append(value)
                if end == size or text[end] != ',':
                    break
                cursor = end + 1
        yield record
        cursor = end
        if cursor < size:
            cursor += 2 if text.startswith('\r\n', cursor) else 1
