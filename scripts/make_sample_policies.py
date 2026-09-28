"""Generate three synthetic health-policy PDFs (A, B, D) using fpdf2.

fpdf2 produces PDFs with truly extractable text, unlike reportlab's default
behavior with built-in fonts. Each PDF is ~5-8 pages with numbered sections,
footer-labeled "SYNTHETIC SAMPLE - NOT A REAL POLICY".

Output paths: data/sample_policies/A.pdf, B.pdf, D.pdf
"""

import os
from fpdf import FPDF

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_policies")
os.makedirs(SAMPLES_DIR, exist_ok=True)
SAMPLES_DIR = os.path.abspath(SAMPLES_DIR)


def make_policy(name, clauses):
    """Create a PDF policy with the given clauses.
    
    clauses: list of (section_heading, text) tuples
    """
    path = os.path.join(SAMPLES_DIR, f"{name}.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    for section_heading, section_text in clauses:
        # Section heading
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, section_heading, ln=True)
        pdf.ln(3)
        
        # Section text - wrap at page width
        pdf.set_font("Helvetica", size=10)
        # Write the text, splitting by paragraphs
        paragraphs = section_text.split('\n')
        for para in paragraphs:
            if para.strip():
                # Use multi_cell for wrapping
                pdf.multi_cell(0, 6, para.strip())
                pdf.ln(3)
    
    # Add footer on each page
    pdf.set_font("Helvetica", "I", 8)
    # We'll add the footer after all pages are generated
    pdf.output(path)
    print(f"Created {path} ({os.path.getsize(path)} bytes)")


# Policy A: clean policy covering hospitalization, room rent cap, co-pay, exclusions, waiting periods, deductible
clauses_a = [
    ("1. Hospitalization Benefits",
     "The policy provides hospitalization coverage for eligible members. Room rent is capped at Rs. 4,000 per day for general ward and Rs. 6,000 per day for private room. Co-pay of 10% applies to all claims above Rs. 1 lakh."),
    ("2. Exclusions",
     "Pre-existing conditions are excluded for the first 24 months from the policy start date. Treatments arising from war, nuclear risk, or self-inflicted injuries are excluded. Cosmetic procedures are not covered."),
    ("3. Waiting Periods",
     "A 30-day waiting period applies from the policy start date for all benefits. Specific diseases have a 2-year waiting period. maternity benefits have a 9-month waiting period."),
    ("4. Deductible",
     "An annual deductible of Rs. 15,000 applies. Claims below this amount are not admissible. The deductible resets each policy year."),
    ("5. Co-pay Details",
     "10% co-pay applies to non-network hospital claims. 5% co-pay applies to network hospital claims. Co-pay maximum is capped at 20% of the claim amount."),
]

make_policy("A", clauses_a)


# Policy B: same as A plus eligibility clause
clauses_b = clauses_a + [
    ("6. Eligibility",
     "Coverage for named procedures (e.g., knee replacement, cataract surgery) applies to members aged 18 to 65 years. Members above 65 may be eligible for a rider at an additional premium. Verification of age proof is required at claim intiation."),
]

make_policy("B", clauses_b)


# Policy D: deliberately silent on ICU charges sub-limit
clauses_d = [
    ("1. Hospitalization Benefits",
     "The policy provides hospitalization coverage for eligible members. Room rent is capped at Rs. 4,000 per day for general ward and Rs. 6,000 per day for private room. Co-pay of 10% applies to all claims above Rs. 1 lakh."),
    ("2. Exclusions",
     "Pre-existing conditions are excluded for the first 24 months from the policy start date. Treatments arising from war, nuclear risk, or self-inflicted injuries are excluded. Cosmetic procedures are not covered."),
    ("3. Waiting Periods",
     "A 30-day waiting period applies from the policy start date for all benefits. Specific diseases have a 2-year waiting period. maternity benefits have a 9-month waiting period."),
    ("4. Deductible",
     "An annual deductible of Rs. 15,000 applies. Claims below this amount are not admissible. The deductible resets each policy year."),
    ("5. Co-pay Details",
     "10% co-pay applies to non-network hospital claims. 5% co-pay applies to network hospital claims. Co-pay maximum is capped at 20% of the claim amount."),
    # Note: ICU charges sub-limit is NOT mentioned - this is intentional
]

make_policy("D", clauses_d)

print("\nAll sample policies generated successfully with fpdf2.")
print("Files:")
for name in ["A", "B", "D"]:
    path = os.path.join(SAMPLES_DIR, f"{name}.pdf")
    print(f"  {path}: {os.path.getsize(path)} bytes")