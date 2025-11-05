from __future__ import annotations
from pathlib import Path
import textwrap

import pytest

from solaris_pointing.offsets.io import write_offsets_tsv, SchemaMismatchError


def test_header_with_extra_blank_lines(tmp_path, md, sample_row):
    p = tmp_path / "out.tsv"
    # create a valid file first
    write_offsets_tsv(str(p), md, [sample_row], append=False)
    # inject additional blank lines before header
    txt = p.read_text(encoding="utf-8")
    parts = txt.splitlines()
    # Find index of header (first non-comment, non-blank)
    idx = 0
    while idx < len(parts) and (not parts[idx] or parts[idx].startswith("#")):
        idx += 1
    # Insert extra blank lines just before header
    parts.insert(idx, "")
    parts.insert(idx, "   ")
    p.write_text("\n".join(parts) + "\n", encoding="utf-8")

    # Should still append fine
    write_offsets_tsv(str(p), md, [sample_row], append=True)


def test_header_with_crlf_lines(tmp_path, md, sample_row):
    p = tmp_path / "out.tsv"
    write_offsets_tsv(str(p), md, [sample_row], append=False)
    txt = p.read_text(encoding="utf-8")
    # rewrite file with CRLF endings
    p.write_bytes(txt.replace("\n", "\r\n").encode("utf-8"))

    # Append should still pass (strip() handles \r\n)
    write_offsets_tsv(str(p), md, [sample_row], append=True)


def test_schema_mismatch_raises(tmp_path, md, sample_row):
    p = tmp_path / "out.tsv"
    write_offsets_tsv(str(p), md, [sample_row], append=False)

    # Corrupt header: reorder columns
    lines = p.read_text(encoding="utf-8").splitlines()
    # Find header line index
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("#")):
        i += 1
    assert i < len(lines)
    lines[i] = (
        "timestamp\televation\tazimuth\toffset_az\toffset_el\ttemperature\tpressure\thumidity"
    )
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SchemaMismatchError) as ei:
        write_offsets_tsv(str(p), md, [sample_row], append=True)

    msg = str(ei.value)
    assert "Path:" in msg and "Expected:" in msg and "Found:" in msg and "Hint:" in msg
    assert str(p) in msg


def test_no_header_value_error(tmp_path, md):
    p = tmp_path / "out.tsv"
    # Only comments, no header
    p.write_text("# only comments\n# still comments\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_offsets_tsv(str(p), md, [], append=True)
