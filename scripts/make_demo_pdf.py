"""Build data/kestrel_annual_report_2025.pdf, a 20-page FICTIONAL annual report for demoing Veritas, and
DEMO_QUESTIONS.md with questions and expected answers taken from the same data (so they cannot disagree).

It mixes prose, KPI and financial tables, a multi-page product table, range-lookup tables (discount and SLA
brackets) and one image-only "scanned" page (warranty claims) that exercises the vision transcription path.

pip install reportlab   (dev-only; not needed to run Veritas)
python scripts/make_demo_pdf.py
"""
import random
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

OUT = Path("data/kestrel_annual_report_2025.pdf")
QUESTIONS = Path("DEMO_QUESTIONS.md")
CO = "Kestrel Robotics Ltd."

# ---------------- single source of truth ----------------
SEGMENTS = [("Warehouse Automation", 231.4, 21.0), ("Medical Robotics", 118.7, 14.5), ("Agricultural Robotics", 62.5, 19.0)]
REVENUE = round(sum(r for _, r, _ in SEGMENTS), 1)            # 412.6
QUARTERS = [("Q1", 88.9, 41.2, 6.1), ("Q2", 97.4, 42.0, 9.8), ("Q3", 104.6, 42.9, 12.4), ("Q4", 121.7, 43.8, 16.9)]
assert round(sum(q[1] for q in QUARTERS), 1) == REVENUE
NET_INCOME, PREV_REVENUE, HEADCOUNT, RD_SPEND, PATENTS = 38.9, 349.7, 1240, 61.2, 57
GROWTH = round((REVENUE / PREV_REVENUE - 1) * 100, 1)      # 18.0
REGIONS = [("North America", 171.2), ("Europe", 118.3), ("Asia-Pacific", 98.4), ("Latin America", 24.7)]
assert round(sum(r for _, r in REGIONS), 1) == REVENUE
FUNCTIONS = [("Engineering", 512), ("Manufacturing", 318), ("Sales & Marketing", 176), ("Customer Support", 121),
             ("Operations & Finance", 79), ("People & Legal", 34)]
assert sum(n for _, n in FUNCTIONS) == HEADCOUNT
random.seed(7)
FAMILIES = [("Atlas", "Warehouse Automation", "AMR"), ("Heron", "Medical Robotics", "Surgical assist"),
            ("Sparrow", "Agricultural Robotics", "Field rover"), ("Talon", "Warehouse Automation", "Picking arm")]
PRODUCTS = []
for fam, seg, kind in FAMILIES:
    for i, gen in enumerate(["S1", "S2", "X1", "X2", "Pro", "Max", "Lite", "Mini", "Plus", "Ultra"]):
        PRODUCTS.append((f"{fam} {gen}", seg, kind, 15 + 25 * i + random.randint(0, 9), 4 + i % 5 + random.randint(0, 3),
                         round(18_500 + 6_250 * i + random.randint(0, 40) * 100, -2), 2019 + i % 7))
PRODUCTS = sorted(PRODUCTS)
ATLAS_X2 = next(p for p in PRODUCTS if p[0] == "Atlas X2")
DISCOUNTS = [(1, 4, "0%"), (5, 19, "4%"), (20, 49, "7.5%"), (50, 99, "10%"), (100, 249, "12.5%"), (250, None, "15%")]
SLA = [("P1 - production stopped", "1 hour", "8 hours", "24/7"), ("P2 - degraded", "4 hours", "2 business days", "24/7"),
       ("P3 - minor issue", "1 business day", "5 business days", "Business hours"),
       ("P4 - question", "2 business days", "10 business days", "Business hours")]
EMISSIONS = [("Scope 1", 3_120, 3_480), ("Scope 2 (market-based)", 4_870, 6_210), ("Scope 3", 41_600, 44_900)]
RENEWABLE = 64
WARRANTY = [("Atlas", 4_210, 97, "Battery connector"), ("Heron", 612, 9, "Arm calibration drift"),
            ("Sparrow", 1_380, 58, "Wheel hub seal"), ("Talon", 2_045, 41, "Gripper sensor")]
BALANCE = [("Cash and equivalents", 96.4, 71.8), ("Receivables", 74.9, 63.2), ("Inventory", 58.1, 52.7),
           ("Property, plant and equipment", 132.6, 118.0), ("Total assets", 401.3, 352.5),
           ("Borrowings", 64.0, 72.5), ("Total liabilities", 168.9, 160.4), ("Shareholders' equity", 232.4, 192.1)]
CASES = [("FreshMart (grocery, US)", "FreshMart deployed 120 Atlas X2 robots across three distribution centres "
           "and cut average order picking time by 35% within six months."),
          ("St. Aurel Hospital (Switzerland)", "St. Aurel Hospital performed 410 procedures with Heron Pro surgical "
           "assist arms in 2025, its first full year of use."),
          ("Verdant Farms (Brazil)", "Verdant Farms runs a fleet of 64 Sparrow field rovers for weeding, reducing "
           "herbicide use by 28% compared with 2024."),
          ("Northgate Logistics (UK)", "Northgate Logistics added 45 Talon picking arms and now processes 18,000 "
           "parcels per hour at peak.")]
GLOSSARY = [("AMR", "Autonomous mobile robot: navigates without fixed tracks."),
            ("ARR", "Annual recurring revenue from software and service contracts."),
            ("MTBF", "Mean time between failures, in operating hours."),
            ("Payload", "Maximum load a robot carries or lifts in normal operation."),
            ("SLA", "Service level agreement: committed response and resolution times."),
            ("Scope 3", "Indirect emissions in the value chain, e.g. purchased parts and shipping.")]

# ---------------- styles ----------------
ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=18, spaceAfter=10,
                    textColor=colors.HexColor("#1f6f5c"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=13, spaceBefore=8, spaceAfter=6)
P = ParagraphStyle("p", parent=ss["BodyText"], fontName="Helvetica", fontSize=10.2, leading=15, spaceAfter=7)
SMALL = ParagraphStyle("small", parent=P, fontSize=8.5, textColor=colors.HexColor("#666666"))


def table(rows, widths=None, header=True):
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9) if header else ("FONT", (0, 0), (0, 0), "Helvetica", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f2ee") if header else colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9c9c9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def money(m):
    return f"${m:,.1f}M"


def lorem_ops(topic, n):
    """Plausible filler prose about a topic, varied but containing no facts the questions rely on."""
    bits = [f"The {topic} team worked closely with customers throughout the year, running quarterly reviews and "
            "publishing roadmap updates so that buyers could plan deployments with confidence.",
            "Supplier diversification continued: no single supplier accounts for more than a fifth of component "
            "spend, and dual sourcing is now in place for motors, lidar units and battery cells.",
            f"Field feedback from {topic} deployments fed directly into firmware releases, which moved from a "
            "quarterly to a monthly cadence in the second half of the year.",
            "Training programmes for integrators were expanded, with new certification tracks covering safety "
            "configuration, fleet orchestration and preventive maintenance.",
            "Management regards these investments as foundational for the next planning cycle and expects them to "
            "reduce deployment time and support costs over the medium term."]
    return [Paragraph(bits[(i + len(topic)) % len(bits)], P) for i in range(n)]


def scanned_page_image() -> Image.Image:
    """Image-only page: the warranty table drawn as a slightly rotated, noisy 'scan' with no text layer."""
    w, h = 1240, 1754
    img = Image.new("L", (w, h), 250)
    d = ImageDraw.Draw(img)
    try:
        bold, reg = ImageFont.truetype("arialbd.ttf", 34), ImageFont.truetype("arial.ttf", 28)
    except OSError:
        bold = reg = ImageFont.load_default()
    d.text((110, 140), "Appendix C - Warranty Claims by Product Family, 2025", font=bold, fill=20)
    d.text((110, 200), f"{CO} - internal quality report (scanned copy)", font=reg, fill=60)
    cols = [110, 380, 640, 860, 1130]
    heads = ["Product family", "Units shipped", "Claims", "Top failure cause"]
    y = 300
    d.rectangle((100, y - 15, 1180, y + 45), outline=30, width=2)
    for x, t in zip(cols, heads):
        d.text((x + 6, y), t, font=bold, fill=20)
    for fam, units, claims, cause in WARRANTY:
        y += 70
        d.line((100, y - 15, 1180, y - 15), fill=90, width=1)
        for x, t in zip(cols, [fam, f"{units:,}", str(claims), cause]):
            d.text((x + 6, y), t, font=reg, fill=25)
    y += 90
    total_units, total_claims = sum(u for _, u, _, _ in WARRANTY), sum(c for _, _, c, _ in WARRANTY)
    d.text((110, y), f"Total: {total_units:,} units shipped, {total_claims} warranty claims.", font=reg, fill=25)
    d.text((110, y + 50), "Claim rate is claims divided by units shipped in the same year.", font=reg, fill=60)
    for _ in range(9000):  # speckle noise
        d.point((random.randint(0, w - 1), random.randint(0, h - 1)), fill=random.randint(120, 200))
    return img.rotate(0.9, fillcolor=245, resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(0.6))


def build():
    s = []
    # 1 cover
    s += [Spacer(1, 70 * mm), Paragraph(CO, ParagraphStyle("c", parent=H1, fontSize=30, leading=36)),
          Paragraph("Annual Report 2025", ParagraphStyle("c2", parent=H2, fontSize=18)), Spacer(1, 10 * mm),
          Paragraph("FICTIONAL DOCUMENT - created to demonstrate the Veritas document question-answering system. "
                    "The company, people and figures are invented.", SMALL), PageBreak()]
    # 2 contents
    sections = ["Letter from the CEO", "2025 highlights", "Business segments", "Quarterly results", "Product catalogue",
                "Research and development", "Customer case studies", "Revenue by region", "People and policies", "Sustainability",
                "Risk factors", "Customer support commitments", "Pricing and volume discounts",
                "Appendix C - Warranty claims (scanned)", "Balance sheet summary", "Glossary", "Governance and contacts"]
    s += [Paragraph("Contents", H1)] + [Paragraph(f"{i}. {t}", P) for i, t in enumerate(sections, 1)] + [PageBreak()]
    # 3 CEO letter
    s += [Paragraph("Letter from the CEO", H1),
          Paragraph(f"2025 was a record year for {CO}. Revenue grew {GROWTH}% to {money(REVENUE)}, up from "
                    f"{money(PREV_REVENUE)} in 2024, and net income reached {money(NET_INCOME)}. We ended the year "
                    f"with {HEADCOUNT:,} employees across four continents.", P),
          Paragraph("The launch of the Atlas X2 autonomous mobile robot in March was the defining moment of the year. "
                    "It is now our best-selling warehouse product and carried us into three new grocery chains.", P),
          Paragraph("In September we opened our new engineering headquarters in Pune, India, which will house up to "
                    "400 engineers by 2027. Our Medical Robotics segment received regulatory clearance for the "
                    "Heron Pro in the European Union.", P),
          Paragraph("- Mira Okafor, Chief Executive Officer", P)] + lorem_ops("leadership", 3) + [PageBreak()]
    # 4 highlights
    s += [Paragraph("2025 highlights", H1),
          table([["Indicator", "2025", "2024"], ["Revenue", money(REVENUE), money(PREV_REVENUE)],
                 ["Net income", money(NET_INCOME), money(29.4)], ["Gross margin", "42.6%", "40.9%"],
                 ["R&D spend", money(RD_SPEND), money(52.3)], ["Patents filed", str(PATENTS), "44"],
                 ["Employees (year end)", f"{HEADCOUNT:,}", "1,065"], ["Robots in the field", "21,480", "16,920"],
                 ["Customer satisfaction (CSAT)", "91%", "88%"]], [80 * mm, 40 * mm, 40 * mm])]
    s += [Spacer(1, 6 * mm)] + lorem_ops("finance", 3) + [PageBreak()]
    # 5-6 segments
    s += [Paragraph("Business segments", H1),
          table([["Segment", "Revenue 2025", "Growth vs 2024"]] + [[n, money(r), f"{g}%"] for n, r, g in SEGMENTS]
                + [["Total", money(REVENUE), f"{GROWTH}%"]], [80 * mm, 40 * mm, 40 * mm])]
    for n, r, g in SEGMENTS:
        s += [Paragraph(n, H2), Paragraph(f"{n} revenue was {money(r)} in 2025, growth of {g}% on the prior year.", P)]
        s += lorem_ops(n.lower(), 2)
    s += [PageBreak()]
    # 7 quarterly
    s += [Paragraph("Quarterly results", H1), Paragraph("Unaudited quarterly figures, US$ millions.", SMALL),
          table([["Quarter", "Revenue", "Gross margin", "Operating income"]]
                + [[q, money(r), f"{gm}%", money(oi)] for q, r, gm, oi in QUARTERS]
                + [["Full year", money(REVENUE), "42.6%", money(round(sum(q[3] for q in QUARTERS), 1))]],
                [40 * mm, 40 * mm, 40 * mm, 45 * mm])] + [Spacer(1, 5 * mm)] + lorem_ops("finance", 4) + [PageBreak()]
    # 8-9 products (multi-page table)
    s += [Paragraph("Product catalogue", H1),
          Paragraph("All products available in 2025. Payload in kilograms, battery runtime in hours, list price in "
                    "US dollars per unit before volume discounts.", SMALL),
          table([["Model", "Segment", "Type", "Payload (kg)", "Runtime (h)", "List price", "Launched"]]
                + [[m, sg, k, str(pl), str(rt), f"${pr:,.0f}", str(y)] for m, sg, k, pl, rt, pr, y in PRODUCTS],
                [26 * mm, 38 * mm, 26 * mm, 20 * mm, 19 * mm, 22 * mm, 17 * mm]), PageBreak()]
    # 10 R&D
    s += [Paragraph("Research and development", H1),
          Paragraph(f"R&amp;D spending was {money(RD_SPEND)} in 2025, equal to {round(RD_SPEND / REVENUE * 100, 1)}% of "
                    f"revenue. We filed {PATENTS} patents, most of them in grasp planning and fleet coordination. "
                    "Our three research labs are in Boston, Munich and Pune.", P)] + lorem_ops("research", 5) + [PageBreak()]
    # case studies
    s += [Paragraph("Customer case studies", H1)]
    for name, text in CASES:
        s += [Paragraph(name, H2), Paragraph(text, P)]
    s += [PageBreak()]
    # 11 regions
    s += [Paragraph("Revenue by region", H1),
          table([["Region", "Revenue 2025", "Share"]]
                + [[n, money(r), f"{round(r / REVENUE * 100, 1)}%"] for n, r in REGIONS], [70 * mm, 45 * mm, 35 * mm])]
    s += [Spacer(1, 5 * mm)] + lorem_ops("regional sales", 4) + [PageBreak()]
    # 12 people
    s += [Paragraph("People and policies", H1),
          table([["Function", "Employees"]] + [[f, str(n)] for f, n in FUNCTIONS] + [["Total", f"{HEADCOUNT:,}"]],
                [80 * mm, 40 * mm]),
          Paragraph("Policies", H2),
          Paragraph("Every employee receives 24 days of paid annual leave plus public holidays. Parental leave is 26 "
                    "weeks at full pay for the primary caregiver and 8 weeks for the secondary caregiver. Office-based "
                    "staff may work remotely up to 2 days per week. The annual learning budget is $2,000 per person.", P),
          PageBreak()]
    # 13 sustainability
    s += [Paragraph("Sustainability", H1),
          table([["Emissions (tCO2e)", "2025", "2024"]] + [[n, f"{a:,}", f"{b:,}"] for n, a, b in EMISSIONS],
                [70 * mm, 40 * mm, 40 * mm]),
          Paragraph(f"Renewable electricity covered {RENEWABLE}% of our consumption in 2025. Our target is 100% "
                    "renewable electricity by 2028 and a 50% cut in Scope 1 and 2 emissions by 2030 against 2022.", P)]
    s += lorem_ops("sustainability", 3) + [PageBreak()]
    # 14 risks
    s += [Paragraph("Risk factors", H1)]
    for title, text in [("Component supply", "Shortages of lidar units or battery cells could delay deliveries."),
                        ("Regulation", "Medical products depend on clearances that can take longer than planned."),
                        ("Customer concentration", "Our five largest customers made up 31% of 2025 revenue."),
                        ("Currency", "About 45% of revenue is earned outside the United States."),
                        ("Cybersecurity", "Fleet software is a target; a breach could stop customer operations.")]:
        s += [Paragraph(title, H2), Paragraph(text, P)]
    s += [PageBreak()]
    # 15 SLA
    s += [Paragraph("Customer support commitments", H1),
          Paragraph("Service level agreement for customers on the Premier support plan.", SMALL),
          table([["Priority", "First response", "Target resolution", "Coverage"]] + [list(r) for r in SLA],
                [55 * mm, 35 * mm, 40 * mm, 35 * mm]),
          Paragraph("Customers on the Standard plan receive the same first-response targets during business hours "
                    "only. Spare parts for P1 incidents ship within 24 hours from regional depots.", P), PageBreak()]
    # 16 discounts (range lookup)
    s += [Paragraph("Pricing and volume discounts", H1),
          Paragraph("Volume discounts apply to the list price of robots ordered in a single purchase order.", P),
          table([["Units in one order - at least", "But not more than", "Discount"]]
                + [[str(a), str(b) if b else "and above", d] for a, b, d in DISCOUNTS], [60 * mm, 50 * mm, 35 * mm]),
          Paragraph("Software subscriptions are billed per robot per month at $149 for Fleet Basic and $299 for "
                    "Fleet Pro, and are not discounted.", P), PageBreak()]
    # 17 scanned appendix (image only)
    buf = BytesIO()
    scanned_page_image().save(buf, format="PNG")
    buf.seek(0)
    s += [RLImage(buf, width=180 * mm, height=254 * mm), PageBreak()]
    # 18 balance
    s += [Paragraph("Balance sheet summary", H1), Paragraph("As at 31 December, US$ millions.", SMALL),
          table([["Item", "2025", "2024"]] + [[n, f"{a:.1f}", f"{b:.1f}"] for n, a, b in BALANCE],
                [80 * mm, 35 * mm, 35 * mm])] + lorem_ops("treasury", 3) + [PageBreak()]
    # 19 glossary
    s += [Paragraph("Glossary", H1)] + [Paragraph(f"<b>{t}</b> - {d}", P) for t, d in GLOSSARY] + [PageBreak()]
    # 20 governance
    s += [Paragraph("Governance and contacts", H1),
          table([["Name", "Role"], ["Mira Okafor", "Chief Executive Officer"], ["Daniel Reyes", "Chief Financial Officer"],
                 ["Anika Shah", "Chief Technology Officer"], ["Lars Vinter", "Chair of the Board"]],
                [60 * mm, 80 * mm]),
          Paragraph("The Board met 9 times in 2025. Investor questions: investors@kestrel.example (fictional).", P)]
    s += lorem_ops("governance", 2)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm,
                            bottomMargin=18 * mm, title=f"{CO} Annual Report 2025 (fictional)")
    doc.build(s)  # no page footer: the scanned page must have no text layer at all


def write_questions():
    total_units, total_claims = sum(u for _, u, _, _ in WARRANTY), sum(c for _, _, c, _ in WARRANTY)
    q4 = QUARTERS[3]
    qs = [
        ("Direct", "What was Kestrel's total revenue in 2025 and how much did it grow?",
         f"{money(REVENUE)}, up {GROWTH}% from {money(PREV_REVENUE)}."),
        ("Direct", "How many patents were filed in 2025?", f"{PATENTS}."),
        ("Direct", "Where is the new engineering headquarters?", "Pune, India (opened in September; up to 400 engineers by 2027)."),
        ("Table", "What was Q4 revenue and operating income?", f"{money(q4[1])} revenue, {money(q4[3])} operating income."),
        ("Table", "How many people work in Customer Support?", f"{dict(FUNCTIONS)['Customer Support']}."),
        ("Table", "What is the list price and payload of the Atlas X2?",
         f"${ATLAS_X2[5]:,.0f}, payload {ATLAS_X2[3]} kg."),
        ("Table", "What were Scope 2 emissions in 2025 compared with 2024?",
         f"{EMISSIONS[1][1]:,} tCO2e in 2025 vs {EMISSIONS[1][2]:,} in 2024."),
        ("Range lookup", "What discount applies to an order of 60 robots?", "10% (50-99 units)."),
        ("Range lookup", "What discount do I get for 250 robots in one order?", "15% (250 and above)."),
        ("Range lookup", "For a P2 incident, how fast is the first response and target resolution?",
         "4 hours first response, 2 business days resolution, 24/7."),
        ("Scanned page", "How many warranty claims did the Sparrow family have and what was the main cause?",
         f"{WARRANTY[2][2]} claims; wheel hub seal."),
        ("Scanned page", "Which product family had the most warranty claims?", f"Atlas ({WARRANTY[0][2]} claims)."),
        ("Scanned page", "How many units were shipped in total according to the warranty appendix?",
         f"{total_units:,} units, {total_claims} claims."),
        ("Multi-part", "What share of revenue came from Europe, and what is the currency risk mentioned?",
         f"{round(REGIONS[1][1] / REVENUE * 100, 1)}% ({money(REGIONS[1][1])}); about 45% of revenue is earned outside the US."),
        ("Multi-part", "What is the parental leave policy and how many remote days are allowed?",
         "26 weeks full pay (primary), 8 weeks (secondary); up to 2 remote days per week."),
        ("Vague", "how green is the company's electricity?", f"{RENEWABLE}% renewable in 2025; target 100% by 2028."),
        ("Vague", "who runs the money side of the business?", "Daniel Reyes, Chief Financial Officer."),
        ("Vague", "is the company getting more profitable?",
         f"Yes: net income {money(NET_INCOME)} vs $29.4M; gross margin 42.6% vs 40.9%."),
        ("Direct", "How much did FreshMart cut picking time and with how many robots?",
         "35% within six months, with 120 Atlas X2 robots."),
        ("Unanswerable (should say not found)", "What is Kestrel's share price?", "Not in the document."),
        ("Unanswerable (should say not found)", "How many robots did Kestrel sell in 2019?", "Not in the document."),
    ]
    lines = ["# Demo questions for `data/kestrel_annual_report_2025.pdf`", "",
             "Fictional 20-page annual report built by `scripts/make_demo_pdf.py`. Upload it in the web app, then ask:",
             "", "| # | Kind | Question | Expected answer |", "|---|---|---|---|"]
    lines += [f"| {i} | {k} | {q} | {a} |" for i, (k, q, a) in enumerate(qs, 1)]
    lines += ["", "Page 17 (warranty appendix) is an image-only scan: it is read by the vision model at upload "
              "(~$0.005). Asking the same question twice shows the cache (instant, $0)."]
    QUESTIONS.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
    write_questions()
    import pypdfium2
    pdf = pypdfium2.PdfDocument(OUT.read_bytes())
    scans = [i + 1 for i in range(len(pdf)) if not pdf[i].get_textpage().get_text_range().strip()]
    print(f"{OUT}: {len(pdf)} pages, image-only pages {scans}; {QUESTIONS} written")
