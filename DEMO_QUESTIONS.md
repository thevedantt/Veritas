# Demo questions for `data/kestrel_annual_report_2025.pdf`

Fictional 20-page annual report built by `scripts/make_demo_pdf.py`. Upload it in the web app, then ask:

| # | Kind | Question | Expected answer |
|---|---|---|---|
| 1 | Direct | What was Kestrel's total revenue in 2025 and how much did it grow? | $412.6M, up 18.0% from $349.7M. |
| 2 | Direct | How many patents were filed in 2025? | 57. |
| 3 | Direct | Where is the new engineering headquarters? | Pune, India (opened in September; up to 400 engineers by 2027). |
| 4 | Table | What was Q4 revenue and operating income? | $121.7M revenue, $16.9M operating income. |
| 5 | Table | How many people work in Customer Support? | 121. |
| 6 | Table | What is the list price and payload of the Atlas X2? | $37,400, payload 90 kg. |
| 7 | Table | What were Scope 2 emissions in 2025 compared with 2024? | 4,870 tCO2e in 2025 vs 6,210 in 2024. |
| 8 | Range lookup | What discount applies to an order of 60 robots? | 10% (50-99 units). |
| 9 | Range lookup | What discount do I get for 250 robots in one order? | 15% (250 and above). |
| 10 | Range lookup | For a P2 incident, how fast is the first response and target resolution? | 4 hours first response, 2 business days resolution, 24/7. |
| 11 | Scanned page | How many warranty claims did the Sparrow family have and what was the main cause? | 58 claims; wheel hub seal. |
| 12 | Scanned page | Which product family had the most warranty claims? | Atlas (97 claims). |
| 13 | Scanned page | How many units were shipped in total according to the warranty appendix? | 8,247 units, 205 claims. |
| 14 | Multi-part | What share of revenue came from Europe, and what is the currency risk mentioned? | 28.7% ($118.3M); about 45% of revenue is earned outside the US. |
| 15 | Multi-part | What is the parental leave policy and how many remote days are allowed? | 26 weeks full pay (primary), 8 weeks (secondary); up to 2 remote days per week. |
| 16 | Vague | how green is the company's electricity? | 64% renewable in 2025; target 100% by 2028. |
| 17 | Vague | who runs the money side of the business? | Daniel Reyes, Chief Financial Officer. |
| 18 | Vague | is the company getting more profitable? | Yes: net income $38.9M vs $29.4M; gross margin 42.6% vs 40.9%. |
| 19 | Direct | How much did FreshMart cut picking time and with how many robots? | 35% within six months, with 120 Atlas X2 robots. |
| 20 | Unanswerable (should say not found) | What is Kestrel's share price? | Not in the document. |
| 21 | Unanswerable (should say not found) | How many robots did Kestrel sell in 2019? | Not in the document. |

Page 17 (warranty appendix) is an image-only scan: it is read by the vision model at upload (~$0.005). Asking the same question twice shows the cache (instant, $0).
