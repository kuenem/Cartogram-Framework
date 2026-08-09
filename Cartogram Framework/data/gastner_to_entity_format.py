#!/usr/bin/env python3
"""
Convert Gastner-format cartogram CSVs (e.g. "NAME_1,Data,Color" or
"Region,Data") into the long "Entity,Code,Year,<value>" format.

Gastner format, one row per region, no year column:
    NAME_1,Data,Color
    AndhraPradesh,604228.62,
    ArunachalPradesh,18509.16,

Target format:
    Entity,Code,Year,gdp
    AndhraPradesh,,2015-16,604228.62
    ArunachalPradesh,,2015-16,18509.16

Since the Gastner file has no Year column, you supply the year (or year
range, e.g. "2015-16") for each file. If you have one Gastner CSV per
year, you can merge them all into a single long-format file in one call.

------------------------------------------------------------------
Usage as a library
------------------------------------------------------------------
    from gastner_to_entity_format import convert_gastner_csv, merge_gastner_csvs

    # Single file, single year
    convert_gastner_csv("sdp_2015-16.csv", "out.csv", year="2015-16",
                         value_column_name="gdp")

    # Multiple files (one per year) merged into one long-format file
    merge_gastner_csvs(
        [("abbeville_2020.csv", 2020), ("abbeville_2021.csv", 2021)],
        "out.csv",
        value_column_name="all years",
    )

------------------------------------------------------------------
Usage from the command line
------------------------------------------------------------------
    # Single file
    python gastner_to_entity_format.py single input.csv output.csv 2015-16 --value-column-name gdp

    # Merge several files (one per year) into one output
    python gastner_to_entity_format.py merge sdp_2020.csv:2020 sdp_2021.csv:2021 \
        -o combined.csv --value-column-name "all years"
"""
import argparse
import csv
from pathlib import Path


def _detect_data_column(header, first_row, entity_idx, data_col_name=None):
    """Figure out which column index holds the numeric data."""
    if data_col_name is not None:
        if data_col_name not in header:
            raise ValueError(f"Column '{data_col_name}' not found in header {header}")
        return header.index(data_col_name)

    if "Data" in header:
        return header.index("Data")

    # Fall back: first column (other than the entity column) that looks numeric
    for i, val in enumerate(first_row):
        if i == entity_idx:
            continue
        try:
            float(val)
            return i
        except (ValueError, TypeError):
            continue

    raise ValueError(
        "Could not auto-detect the data column; pass data_col_name explicitly."
    )


def _read_gastner_rows(input_path, entity_col_index=0, data_col_name=None):
    """Read a Gastner-format CSV, return list of (entity, value) tuples."""
    input_path = Path(input_path)
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]

    if not rows:
        return []

    data_idx = _detect_data_column(header, rows[0], entity_col_index, data_col_name)

    result = []
    for row in rows:
        entity = row[entity_col_index].strip()
        if not entity:
            continue
        value = row[data_idx].strip()
        result.append((entity, value))
    return result


def convert_gastner_csv(
    input_path,
    output_path,
    year,
    value_column_name="value",
    entity_col_index=0,
    data_col_name=None,
):
    """
    Convert one Gastner-format CSV into Entity,Code,Year,<value_column_name>.

    Parameters
    ----------
    input_path : str or Path
        Path to the Gastner-format CSV.
    output_path : str or Path
        Where to write the converted CSV.
    year : str or int
        Value for the Year column on every row (e.g. 2020 or "2015-16").
    value_column_name : str
        Header for the 4th output column (e.g. "gdp", "all years").
    entity_col_index : int
        Index of the entity/name column in the input file (default 0).
    data_col_name : str or None
        Name of the input column holding the numeric data. If None,
        prefers a column literally named "Data", else auto-detects the
        first numeric-looking column.
    """
    rows = _read_gastner_rows(input_path, entity_col_index, data_col_name)
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity", "Code", "Year", value_column_name])
        for entity, value in rows:
            writer.writerow([entity, "", year, value])
    return output_path


def merge_gastner_csvs(
    files_with_years,
    output_path,
    value_column_name="value",
    entity_col_index=0,
    data_col_name=None,
):
    """
    Merge several Gastner-format CSVs (one per year) into a single
    Entity,Code,Year,<value_column_name> file.

    Parameters
    ----------
    files_with_years : list of (path, year) tuples
        e.g. [("sdp_2020.csv", 2020), ("sdp_2021.csv", 2021)]
    output_path : str or Path
        Where to write the merged CSV.
    value_column_name, entity_col_index, data_col_name :
        Same meaning as in convert_gastner_csv.
    """
    output_path = Path(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Entity", "Code", "Year", value_column_name])
        for path, year in files_with_years:
            rows = _read_gastner_rows(path, entity_col_index, data_col_name)
            for entity, value in rows:
                writer.writerow([entity, "", year, value])
    return output_path


def _parse_pair(spec):
    """Parse 'path:year' into (path, year), tolerant of Windows drive letters like C:\\..."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"Expected format path:year, got '{spec}'")
    path, year = spec.rsplit(":", 1)
    return path, year


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_single = sub.add_parser("single", help="Convert one Gastner CSV")
    p_single.add_argument("input_csv")
    p_single.add_argument("output_csv")
    p_single.add_argument("year")
    p_single.add_argument("--value-column-name", default="value")
    p_single.add_argument("--data-col-name", default=None)
    p_single.add_argument("--entity-col-index", type=int, default=0)

    p_merge = sub.add_parser("merge", help="Merge several Gastner CSVs (one per year) into one file")
    p_merge.add_argument("pairs", nargs="+", type=_parse_pair, help="path:year, e.g. sdp_2020.csv:2020")
    p_merge.add_argument("-o", "--output", required=True)
    p_merge.add_argument("--value-column-name", default="value")
    p_merge.add_argument("--data-col-name", default=None)
    p_merge.add_argument("--entity-col-index", type=int, default=0)

    args = parser.parse_args()

    if args.mode == "single":
        out = convert_gastner_csv(
            args.input_csv,
            args.output_csv,
            args.year,
            value_column_name=args.value_column_name,
            entity_col_index=args.entity_col_index,
            data_col_name=args.data_col_name,
        )
        print(f"Wrote {out}")
    else:
        out = merge_gastner_csvs(
            args.pairs,
            args.output,
            value_column_name=args.value_column_name,
            entity_col_index=args.entity_col_index,
            data_col_name=args.data_col_name,
        )
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
