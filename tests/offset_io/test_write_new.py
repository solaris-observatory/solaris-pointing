from __future__ import annotations
from pathlib import Path

from solaris_pointing.offset_io.tsv import write_offsets_tsv


def test_overwrite_new_file(
    tmp_path, md, sample_row, parse_noncomment_header_and_rows, ANGLE_4DEC_RE
):
    p = tmp_path / "out.tsv"
    write_offsets_tsv(str(p), md, [sample_row], append=False)
    text = p.read_text(encoding="utf-8")
    columns = (
        "timestamp\tazimuth\televation\toffset_az\toffset_el\t"
        + "temperature\tpressure\thumidity"
    )

    assert "# Location: MZS, Antarctica" in text
    assert "# offset_az: solar azimuth - observed azimuth, deg" in text
    assert columns in text

    # Use the injected parser fixture
    header, rows = parse_noncomment_header_and_rows(text)
    assert header == columns
    assert len(rows) == 1
    row = rows[0].split("\t")
    assert row[0] == "2025-08-01T10:00:00Z"
    # azimuth/elevation/offsets must be exactly 4 decimals
    assert ANGLE_4DEC_RE.match(row[1])
    assert ANGLE_4DEC_RE.match(row[2])
    assert ANGLE_4DEC_RE.match(row[3])
    assert ANGLE_4DEC_RE.match(row[4])
    # env fields None -> "NaN"
    assert row[5] == "NaN" and row[6] == "NaN" and row[7] == "NaN"


def test_first_write_append_true_equivalent(tmp_path, md, sample_row):
    p = tmp_path / "out.tsv"
    write_offsets_tsv(str(p), md, [sample_row], append=True)
    text = p.read_text(encoding="utf-8")
    assert "timestamp\tazimuth\televation" in text
