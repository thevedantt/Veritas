"""Run: python test_ingest.py"""
from ingest import compact, with_sections

assert compact("| S755 | S765 | $O | %0 | Sales |") == "| $755 | $765 | $0 | 0% | Sales |"

rows = ["| $0 | 10% |\n", "| Single | Single |\n", "| $648,100 | 37% |\n", "| Head of Household | Head of Household |\n",
        "| $15,550 | 0% |\n"]
assert with_sections(rows) == ["| $0 | 10% |\n", "| Single | Single |\n", "| Single: $648,100 | 37% |\n",
                               "| Head of Household | Head of Household |\n", "| Head of Household: $15,550 | 0% |\n"]
assert with_sections(["| 2020 | 2020 |\n", "| a | b |\n"]) == ["| 2020 | 2020 |\n", "| a | b |\n"]  # numbers aren't labels

# scanned pages are split out of a batch and parsed one by one; text pages stay together
from ingest import segments  # noqa: E402

assert list(segments(1, 4, {2, 3})) == [((1, 1), False), ((2, 2), True), ((3, 3), True), ((4, 4), False)]
assert list(segments(5, 8, set())) == [((5, 8), False)]
assert list(segments(1, 3, {1})) == [((1, 1), True), ((2, 3), False)]
print("ok")

from ingest import square_tables  # noqa: E402

assert square_tables("Title\n| a | b |\n|---|---|---|\n| 1 | 2 | 3 |\ntext") == \
    "Title\n| a | b |  |\n|---|---|---|\n| 1 | 2 | 3 |\ntext"
print("ok")

# a section row may be bold and only partly filled (vision-model transcriptions)
assert with_sections(["| **Single** | **Single** | | |\n", "| $1 | $2 | $3 | $4 |\n"])[1] == "| Single: $1 | $2 | $3 | $4 |\n"
print("ok")

from ingest import trim  # noqa: E402

assert trim("If the Adjusted Annual Wage Amount on Worksheet 1A is: At least") == "If the Adjuste…1A is: At least"
assert trim("At least") == "At least" and len(trim("x" * 80)) == 30
print("ok")
