"""Generate the three demo documents into data/sample_docs/.

    python scripts/generate_sample_docs.py

The planted signals are deliberate and load-bearing for the demo:

  * sample_insurance_claim.txt — a duplicate prior claim for the same property
    under a different policy number with a different insurer, buried in section 31
    (~page 31). It appears nowhere in the first ten sections, so a truncated
    context window provably cannot see it.
  * sample_contract.txt — uncapped liability in section 29, an unusual
    jurisdiction clause in section 34.
  * sample_fraud_case.txt — 14 transfers of $9,400-$9,850 across two months,
    each just under the $10,000 reporting threshold, buried mid-history.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import config  # noqa: E402

RNG = random.Random(20260817)  # deterministic output across runs

WORDS_PER_PAGE = 480


class DocBuilder:
    """Accumulates text and emits `--- PAGE n ---` markers by word count."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.words = 0
        self.page = 1
        self.parts.append(f"\n\n--- PAGE {self.page} ---\n\n")
        self.section_pages: dict[int, int] = {}

    def add(self, text: str) -> None:
        for paragraph in text.split("\n\n"):
            if not paragraph.strip():
                continue
            count = len(paragraph.split())
            if self.words + count > WORDS_PER_PAGE:
                self.page += 1
                self.parts.append(f"\n\n--- PAGE {self.page} ---\n\n")
                self.words = 0
            self.parts.append(paragraph.strip() + "\n\n")
            self.words += count

    def section(self, number: int, title: str) -> None:
        # Force a page break so section N lands near page N — the demo cites pages.
        if self.words > WORDS_PER_PAGE * 0.35:
            self.page += 1
            self.parts.append(f"\n\n--- PAGE {self.page} ---\n\n")
            self.words = 0
        self.section_pages[number] = self.page
        self.parts.append(f"SECTION {number}. {title.upper()}\n\n")

    def build(self) -> str:
        return "".join(self.parts)

    @property
    def word_count(self) -> int:
        return len(self.build().split())


# ----------------------------------------------------------------------
# Shared filler vocabulary — plausible, dense, and free of the planted signals.
# ----------------------------------------------------------------------
ADJUSTERS = [
    "M. Okonjo", "R. Feldman", "S. Nakamura", "D. Achterberg", "P. Villanueva",
    "L. Brennan", "T. Adeyemi",
]
WITNESSES = [
    "Harold Vance", "Priya Raghavan", "Devon Mbeki", "Clara Lindqvist",
    "Tomas Herrera", "Ngozi Adeleke", "Marcus Whitfield", "Yuki Tanabe",
]
VENDORS = [
    "Caldwell Restoration LLC", "Pinnacle Structural Services",
    "Ironwood Salvage & Recovery", "Meridian Environmental Testing",
    "Kestrel Loss Consultants",
]


def money(low: int, high: int) -> str:
    return f"${RNG.randrange(low, high, 50):,}"


def filler(paragraphs: int, topic: str) -> str:
    """Plausible claims-file prose. Deterministic given the module-level seed."""
    templates = [
        "The adjuster of record, {adjuster}, documented the {topic} on site and "
        "confirmed that the observations were consistent with the narrative "
        "provided at first notice of loss. Photographic evidence numbering "
        "{photos} images was appended to the file under exhibit reference "
        "EX-{ex:03d}. No material discrepancies were identified during this "
        "portion of the review.",
        "Field notes relating to the {topic} record an arrival time of "
        "{hour}:{minute} and a total on-site duration of {duration} minutes. "
        "The property was secured at the conclusion of the visit and access "
        "was restricted to authorised personnel pending completion of the "
        "structural assessment.",
        "{vendor} was engaged to perform the {topic} under purchase order "
        "PO-{po}. The engagement scope covered assessment, documentation and "
        "preliminary mitigation. Invoiced amount for this phase was {amount}, "
        "which falls within the range historically observed for comparable "
        "commercial losses in this jurisdiction.",
        "A telephone interview with {witness} was conducted regarding the "
        "{topic}. The account given was internally consistent and corroborated "
        "the timeline established by the responding authorities. The witness "
        "confirmed availability for a recorded statement if required at a "
        "later stage of the adjustment.",
        "Review of the policy schedule confirms that coverage applicable to "
        "the {topic} was in force at the date of loss, subject to the standard "
        "deductible of {deductible} and the sublimits set out in the "
        "declarations page. No endorsement affecting this coverage part was "
        "identified during the file review.",
        "Documentation supporting the {topic} was indexed and cross-referenced "
        "against the loss inventory. Item-level reconciliation produced a "
        "variance of {variance}, which is within the tolerance threshold "
        "applied by the desk adjuster for losses of this magnitude.",
    ]
    out = []
    for _ in range(paragraphs):
        template = RNG.choice(templates)
        out.append(
            template.format(
                topic=topic,
                adjuster=RNG.choice(ADJUSTERS),
                witness=RNG.choice(WITNESSES),
                vendor=RNG.choice(VENDORS),
                photos=RNG.randint(12, 240),
                ex=RNG.randint(1, 400),
                hour=RNG.randint(7, 19),
                minute=f"{RNG.randint(0, 59):02d}",
                duration=RNG.randrange(45, 400, 15),
                po=RNG.randint(100000, 999999),
                amount=money(4000, 90000),
                deductible=money(5000, 50000),
                variance=money(100, 4000),
            )
        )
    return "\n\n".join(out)


# ----------------------------------------------------------------------
# 1. Insurance claim — ~15,000 words, 47 sections, fraud indicator in §31
# ----------------------------------------------------------------------
CLAIM_SECTIONS = [
    "First Notice of Loss", "Policyholder Identification", "Policy Schedule and Coverage Parts",
    "Premises Description", "Incident Narrative", "Emergency Services Response",
    "Fire Department Incident Report Summary", "Initial Site Inspection",
    "Scene Preservation and Security", "Cause and Origin Preliminary Findings",
    "Electrical Systems Assessment", "Fire Suppression Systems Review",
    "Structural Engineering Assessment", "Roof and Envelope Damage",
    "Interior Finishes and Fixtures", "Mechanical Plant and HVAC",
    "Inventory and Stock Loss", "Business Personal Property Schedule",
    "Equipment Breakdown Considerations", "Environmental and Hazmat Screening",
    "Debris Removal Scope", "Mitigation and Emergency Repairs",
    "Business Interruption Preliminary Analysis", "Extra Expense Documentation",
    "Payroll and Continuing Expenses", "Witness Statement — Site Manager",
    "Witness Statement — Night Security", "Witness Statement — Adjacent Tenant",
    "Witness Statement — Responding Officer", "Recorded Statement of the Insured",
    "Prior Claims History",  # <-- section 31: the planted indicator
    "Underwriting File Review", "Premium Payment and Lapse History",
    "Mortgagee and Loss Payee Interests", "Subrogation Potential",
    "Salvage Assessment", "Contractor Estimates — Structural",
    "Contractor Estimates — Mechanical", "Contractor Estimates — Finishes",
    "Independent Adjuster Review", "Reserve Analysis",
    "Coverage Position and Reservation of Rights", "Special Investigation Unit Referral Criteria",
    "Regulatory Notifications", "Settlement Negotiation Log",
    "Outstanding Items and Next Steps", "Adjuster Recommendation and Sign-Off",
]

FRAUD_PLANT = """Prior claims history was obtained from the industry loss register on \
14 March 2026 and reviewed against the current file. The search returned one \
matched record that warrants attention.

On 12 September 2024 — eighteen months prior to the date of loss in the present \
matter — a claim was filed in respect of the same premises at 4188 Harrowgate \
Industrial Parkway, Unit C. That claim, bearing claim number CLM-2024-778341, was \
submitted under policy number POL-CG-88213-B, issued by Northgate Mutual Assurance, \
and described a fire loss originating in the rear storage area. The reported cause, \
the described point of origin, and the schedule of damaged stock in that filing are \
substantially identical to those presented in the current claim.

The named insured on the prior filing is recorded as Halberd Trading Company, which \
shares its principal, Gregory Halloran, with the insured in the present matter, \
Halberd Logistics Group. The prior claim settled on 3 February 2025 for $412,500. \
Neither the prior loss nor the prior policy was disclosed on the application for \
the current policy, POL-CG-91556-A, notwithstanding the express question at item 7 \
of the application concerning losses in the preceding five years.

The prior claim was placed with a different insurer, which is the likely reason the \
overlap was not surfaced during underwriting. The proximity of the two losses, the \
common principal, the identical premises, and the non-disclosure at application are \
each independently material and collectively support a referral to the Special \
Investigation Unit."""


def build_claim() -> str:
    doc = DocBuilder()
    doc.add(
        "COMMERCIAL PROPERTY INSURANCE CLAIM FILE\n\n"
        "Claim Number: CLM-2026-104772\n"
        "Policy Number: POL-CG-91556-A\n"
        "Named Insured: Halberd Logistics Group\n"
        "Loss Location: 4188 Harrowgate Industrial Parkway, Unit C\n"
        "Date of Loss: 27 February 2026\n"
        "Date Reported: 27 February 2026\n"
        "Peril: Fire\n"
        "Adjuster of Record: M. Okonjo\n"
        "Estimated Gross Loss: $2,840,000\n"
        "Current Reserve: $2,100,000\n"
    )

    for index, title in enumerate(CLAIM_SECTIONS, start=1):
        doc.section(index, title)
        if index == 31:
            doc.add(FRAUD_PLANT)
            doc.add(filler(3, "prior claims register reconciliation"))
        else:
            doc.add(filler(RNG.randint(5, 9), title.lower()))

    doc.add(
        "END OF CLAIM FILE. This file is maintained under records retention "
        "schedule RR-7 and is subject to privilege where indicated."
    )
    return doc.build()


# ----------------------------------------------------------------------
# 2. Contract — ~12,000 words, 38 sections
# ----------------------------------------------------------------------
CONTRACT_SECTIONS = [
    "Parties and Recitals", "Definitions", "Interpretation", "Grant of Licence",
    "Scope of Permitted Use", "Restrictions on Use", "Delivery and Acceptance",
    "Implementation Services", "Configuration and Customisation", "Training",
    "Support and Maintenance", "Service Levels", "Service Credits",
    "Fees and Charges", "Invoicing", "Payment Terms", "Taxes",
    "Price Adjustment", "Audit Rights", "Data Protection",
    "Security Standards", "Business Continuity", "Confidentiality",
    "Intellectual Property Ownership", "Feedback and Improvements",
    "Third Party Components", "Warranties", "Disclaimer of Warranties",
    "Limitation of Liability",  # 29 — uncapped
    "Indemnification", "Insurance", "Force Majeure", "Term",
    "Governing Law and Jurisdiction",  # 34 — unusual venue
    "Termination for Cause", "Termination for Convenience",
    "Effect of Termination", "General Provisions",
]

LIABILITY_PLANT = """29.1 Subject to clause 29.3, each party's aggregate liability \
arising out of or in connection with this Agreement, whether in contract, tort \
(including negligence), breach of statutory duty or otherwise, shall be limited to \
the total Fees paid or payable by the Customer in the twelve (12) month period \
immediately preceding the event giving rise to the claim.

29.2 Neither party shall be liable for any indirect, special, incidental, punitive \
or consequential loss, loss of profit, loss of revenue, loss of anticipated savings, \
loss of goodwill or loss of data, in each case howsoever arising.

29.3 Notwithstanding clause 29.1, the Supplier's liability shall be unlimited and \
the cap in clause 29.1 shall not apply in respect of: (a) any breach of clause 23 \
(Confidentiality); (b) any breach of clause 20 (Data Protection); (c) any claim \
arising from or connected with the Supplier's provision of the Services, including \
any failure, defect, interruption, degradation or non-conformity of the Software or \
the Services; and (d) any indemnity given under clause 30.

29.4 The Customer acknowledges that the exclusions and limitations in this clause \
29 are reasonable having regard to the Fees and that it has had the opportunity to \
obtain independent legal advice."""

JURISDICTION_PLANT = """34.1 This Agreement and any dispute or claim arising out of \
or in connection with it or its subject matter or formation (including \
non-contractual disputes or claims) shall be governed by and construed in \
accordance with the laws of the Republic of Vanuatu, without regard to its \
conflict of laws principles.

34.2 The parties irrevocably agree that the courts of Port Vila, Republic of \
Vanuatu shall have exclusive jurisdiction to settle any dispute or claim arising \
out of or in connection with this Agreement or its subject matter or formation.

34.3 The Customer irrevocably waives any objection to the venue of any proceeding \
in such courts on the grounds of forum non conveniens or any similar ground, and \
irrevocably waives any right to trial by jury and any right to participate in a \
class, collective or representative proceeding.

34.4 Nothing in this clause 34 shall limit the Supplier's right to bring \
proceedings against the Customer in any other court of competent jurisdiction."""

CONTRACT_TEMPLATES = [
    "{n}.1 The parties acknowledge that the provisions of this clause {n} are "
    "material to the commercial bargain recorded in this Agreement and have been "
    "negotiated at arm's length between parties of comparable bargaining power.",
    "{n}.2 Except as expressly provided in this clause {n}, nothing in this "
    "Agreement shall operate to exclude, restrict or vary any right or remedy "
    "available to either party at law or in equity in respect of the subject "
    "matter of this clause.",
    "{n}.3 Where any obligation under this clause {n} falls due for performance on "
    "a day that is not a Business Day, that obligation shall instead fall due on "
    "the next following Business Day, and any applicable notice period shall be "
    "extended accordingly.",
    "{n}.4 The Supplier shall maintain complete and accurate records relating to "
    "its performance of the obligations under this clause {n} for a period of not "
    "less than six (6) years following the date of expiry or termination of this "
    "Agreement, and shall make such records available to the Customer on "
    "reasonable written notice.",
    "{n}.5 Any failure or delay by either party in exercising any right under this "
    "clause {n} shall not constitute a waiver of that right, nor shall any single "
    "or partial exercise of any right preclude any further exercise of that right "
    "or of any other right under this Agreement.",
    "{n}.6 The Customer shall provide such cooperation, information and access as "
    "the Supplier may reasonably require in order to perform its obligations under "
    "this clause {n}, and the Supplier shall not be liable for any failure to "
    "perform to the extent caused by the Customer's failure to do so.",
    "{n}.7 Any notice required to be given under this clause {n} shall be in "
    "writing and shall be delivered by hand, sent by pre-paid recorded delivery, "
    "or transmitted by electronic mail to the address of the relevant party set "
    "out in Schedule 1, and shall be deemed received on the second Business Day "
    "following despatch.",
    "{n}.8 The obligations set out in this clause {n} shall survive the expiry or "
    "termination of this Agreement for whatever reason and shall continue in full "
    "force and effect notwithstanding any such expiry or termination, save to the "
    "extent that the relevant obligation has by its nature been discharged.",
    "{n}.9 If any provision or part-provision of this clause {n} is or becomes "
    "invalid, illegal or unenforceable, it shall be deemed modified to the minimum "
    "extent necessary to make it valid, legal and enforceable, and if such "
    "modification is not possible, the relevant provision shall be deemed deleted "
    "without affecting the validity of the remainder of this Agreement.",
    "{n}.10 The parties shall each bear their own costs and expenses incurred in "
    "connection with the performance of their respective obligations under this "
    "clause {n}, save where this Agreement expressly provides otherwise or where "
    "the parties agree otherwise in writing.",
    "{n}.11 Nothing in this clause {n} shall be construed as creating a "
    "partnership, joint venture, agency or employment relationship between the "
    "parties, and neither party shall have authority to bind the other or to "
    "incur any obligation on the other's behalf.",
]


def build_contract() -> str:
    doc = DocBuilder()
    doc.add(
        "MASTER SOFTWARE LICENCE AND SERVICES AGREEMENT\n\n"
        "Contract Reference: CTR-2026-0417\n"
        "Effective Date: 1 April 2026\n"
        "Supplier: Aetherline Systems International Ltd\n"
        "Customer: Halberd Logistics Group\n"
        "Initial Term: 36 months\n"
        "Annual Contract Value: $1,840,000\n"
    )

    for index, title in enumerate(CONTRACT_SECTIONS, start=1):
        doc.section(index, title)
        if index == 29:
            doc.add(LIABILITY_PLANT)
        elif index == 34:
            doc.add(JURISDICTION_PLANT)
        else:
            body = "\n\n".join(
                template.format(n=index)
                for template in RNG.sample(
                    CONTRACT_TEMPLATES, k=RNG.randint(5, 8)
                )
            )
            doc.add(body)

    doc.add(
        "IN WITNESS WHEREOF the parties have executed this Agreement as of the "
        "Effective Date first written above."
    )
    return doc.build()


# ----------------------------------------------------------------------
# 3. Fraud case — ~8,000 words, structuring pattern buried mid-history
# ----------------------------------------------------------------------
FRAUD_CASE_SECTIONS = [
    "Case Header and Referral Source", "Customer Profile",
    "Account Opening and KYC Documentation", "Beneficial Ownership Review",
    "Expected Account Activity at Onboarding", "Year One Transaction Overview",
    "Year One Quarterly Analysis", "Counterparty Review — Domestic",
    "Counterparty Review — International", "Cash Activity Summary",
    "Wire Transfer Activity — Outbound", "Wire Transfer Activity — Inbound",
    "Year Two Transaction Overview", "Anomaly Detection System Alerts",
    "Alert Disposition Log", "Relationship Manager Notes",
    "Branch Interaction Records", "Negative News Screening",
    "Sanctions and PEP Screening", "Enhanced Due Diligence Findings",
    "Analyst Assessment", "Recommendation and Disposition",
]

STRUCTURING_PLANT = """Detailed review of outbound wire activity for the period \
4 November 2025 through 29 December 2025 identified a cluster of transfers that \
warrants specific comment.

Across this fifty-six day window the account executed fourteen (14) outbound \
transfers to three beneficiary accounts held at two institutions. The individual \
transfer amounts were: $9,850, $9,400, $9,700, $9,650, $9,850, $9,500, $9,750, \
$9,400, $9,900, $9,600, $9,850, $9,450, $9,700 and $9,550. The aggregate value of \
the cluster is $135,150.

Every transfer in the cluster falls between $9,400 and $9,900 — that is, below the \
$10,000 threshold at which a currency transaction report is generated, but not so \
far below as to be explained by ordinary rounding. No single transfer in the two \
years preceding this window fell within that band. The account's historical \
outbound transfer profile is bimodal, clustering either under $3,000 or above \
$45,000.

Transfers were executed on Tuesdays and Thursdays without exception, at intervals \
of two to four days, and never more than one per day. Two of the three beneficiary \
accounts were opened within the ninety days preceding the first transfer in the \
cluster. The stated purpose recorded on each transfer instruction was "supplier \
settlement", but no corresponding invoices were provided when requested by the \
relationship manager on 8 January 2026.

The amount distribution, the regularity of timing, the recency of the beneficiary \
accounts and the absence of supporting documentation are, taken together, \
consistent with structuring."""


def build_fraud_case() -> str:
    doc = DocBuilder()
    doc.add(
        "FINANCIAL CRIME INVESTIGATION — CASE FILE\n\n"
        "Case Reference: CASE-FC-2026-0219\n"
        "Customer: Kestrel Marine Supply Co.\n"
        "Primary Account: ACCT-4471-90882\n"
        "Relationship Opened: 3 January 2024\n"
        "Review Period: January 2024 - February 2026\n"
        "Assigned Analyst: T. Adeyemi\n"
        "Referral Source: Automated transaction monitoring\n"
    )

    for index, title in enumerate(FRAUD_CASE_SECTIONS, start=1):
        doc.section(index, title)
        if index == 11:  # Wire Transfer Activity — Outbound
            doc.add(STRUCTURING_PLANT)
            doc.add(filler(3, "outbound wire reconciliation"))
        else:
            doc.add(filler(RNG.randint(7, 11), title.lower()))

    doc.add("END OF CASE FILE.")
    return doc.build()


# ----------------------------------------------------------------------
def main() -> int:
    out_dir = Path(config.SAMPLE_DOCS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        "sample_insurance_claim.txt": build_claim,
        "sample_contract.txt": build_contract,
        "sample_fraud_case.txt": build_fraud_case,
    }

    print(f"Writing sample documents to {out_dir}\n")
    for name, builder in documents.items():
        text = builder()
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        words = len(text.split())
        pages = text.count("--- PAGE ")
        print(f"  {name:<32} {words:>7,} words  {pages:>4} pages  {len(text):>8,} chars")

    print("\nPlanted signals:")
    print("  insurance claim : duplicate prior claim CLM-2024-778341 / "
          "POL-CG-88213-B in section 31")
    print("  contract        : uncapped liability §29.3, Vanuatu jurisdiction §34")
    print("  fraud case      : 14 transfers of $9,400-$9,900 (structuring) in §11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
