# PS11 — `AccessBridge`
## Breaking Barriers: Designing Digital Products That Empower Every User

> Read `00_COMMON_FOUNDATION.md` first.

---

## 1. PITCH

**AccessBridge is a runtime layer that repairs accessibility violations on live pages in milliseconds — and emits the source-code patch that makes the repair permanent.** Audit tools produce a PDF of 400 violations that nobody fixes. AccessBridge fixes them now, for the user in front of you, and files the pull request.

This is also the most visually persuasive demo of all fifteen problem statements. A screen reader speaking a broken page and then speaking the same page correctly is not a chart — it is an experience.

---

## 2. CORE INNOVATION

1. **Fix-at-runtime plus patch-at-source.** Two outputs from one analysis: a live DOM repair for the user who needs it right now, and a diff against the source repository so the fix ships permanently. Nobody does both, and the second is what makes it a business proposition rather than a band-aid.

2. **Context-grounded alt text.** Generic vision models describe pixels: *"a person holding a device."* AccessBridge grounds description in **page context** — the surrounding paragraph, the product record, the caption, the link target, and the image's *functional role*. The same photo on a product page becomes "Aurora wireless earbuds in charcoal, shown in the charging case", on a news article becomes something else entirely, and in a decorative hero position becomes `aria-hidden="true"` with no announcement at all. **Role classification before description** is the insight: WCAG's actual requirement is that alt text convey *purpose*, not appearance.

3. **Personalised adaptation profiles.** Accessibility is not one setting. A profile (reading level, motion sensitivity, contrast need, audio-first, cognitive load, keyboard-only, magnification) drives a *different* adaptation of the same page. The problem statement explicitly calls out that "one-size-fits-all" is the failure — so build the answer to that literally.

---

## 3. ARCHITECTURE

```
 Live page (browser ext / CDN shim / server middleware)
        │  DOM snapshot + computed styles + screenshot
        ▼
 ┌──────────────────────────────────────────────────────┐
 │ AUDIT ENGINE   axe-core rules + custom AI-only checks │
 └───────────────────────┬──────────────────────────────┘
                         ▼
 ┌──────────────────────────────────────────────────────┐
 │ ROLE CLASSIFIER   informative │ decorative │ functional│
 │                   │ complex │ text-in-image           │
 └───────────────────────┬──────────────────────────────┘
                         ▼
 ┌──────────────┬──────────────┬──────────────┬─────────────────┐
 │ ALT TEXT GEN │ STRUCTURE FIX│ FOCUS/KEYBOARD│ CONTRAST/MOTION │
 │ context-      │ headings,    │ tab order,    │ ratio repair,   │
 │ grounded VLM  │ landmarks,   │ skip links,   │ prefers-reduced │
 │               │ labels, lang │ traps, ARIA   │ -motion         │
 └──────────────┴──────┬───────┴──────────────┴─────────────────┘
                       ▼
 ┌──────────────────────────────────────────────────────┐
 │ PROFILE ADAPTER  reading level │ contrast │ audio-first│
 │                  motion │ cognitive load │ magnification│
 └───────────┬──────────────────────────┬───────────────┘
             ▼                          ▼
    RUNTIME PATCH (DOM)          SOURCE PATCH (git diff + PR)
             │
             ▼
    CONFORMANCE EVIDENCE LOG (WCAG 2.2 AA, per criterion)
```

---

## 4. EXTRA DEPENDENCIES

```
playwright==1.44.0          # headless browser, DOM + screenshot capture
beautifulsoup4==4.12.3
lxml==5.2.2
Pillow==10.3.0
axe-core-python==0.1.0      # or bundle axe.min.js and inject
cssutils==2.11.1
textstat==0.7.3
gTTS==2.5.1                 # offline-friendly TTS for the audio demo
```

Claude's vision capability handles the image understanding — send the cropped image plus the surrounding DOM context in one multimodal call.

---

## 5. PROJECT-SPECIFIC CONFIG

```python
WCAG_LEVEL: str = "AA"
WCAG_VERSION: str = "2.2"
RULESET: list[str] = ["wcag2a","wcag2aa","wcag21aa","wcag22aa","best-practice"]
ALT_TEXT_MAX_CHARS: int = 125
ALT_CONTEXT_WINDOW_CHARS: int = 600      # surrounding text sent to the VLM
DECORATIVE_CONFIDENCE_THRESHOLD: float = 0.80
CONTRAST_TARGET_NORMAL: float = 4.5
CONTRAST_TARGET_LARGE: float = 3.0
TOUCH_TARGET_MIN_PX: int = 24            # WCAG 2.2 SC 2.5.8
PROFILES: list[str] = ["default","low_vision","screen_reader","cognitive",
                       "motor","dyslexia","audio_first"]
PATCH_OUTPUT_DIR: str = "./out/patches"
```

---

## 6. MODULE SPECIFICATIONS

### 6.1 `core/audit.py`

```python
class AuditEngine:
    async def audit(self, url_or_html) -> AuditReport
        """
        1. Playwright: load, wait for network idle, capture DOM, computed
           styles, accessibility tree, and a full-page screenshot.
        2. Inject axe-core → deterministic violations mapped to WCAG SCs.
        3. Run AI-ONLY checks that axe structurally cannot do:
           - alt text that EXISTS but is useless ('image', 'img_4471.jpg',
             'click here') — axe passes these, users are failed by them
           - link text that isn't meaningful out of context (SC 2.4.4)
           - heading text that doesn't describe its section (SC 2.4.6)
           - form labels that are present but ambiguous
           - error messages that state a problem without a remedy (SC 3.3.3)
           - reading order mismatch between visual layout and DOM order
        Return violations with: SC id, severity, selector, evidence,
        fixable_at_runtime: bool.
        """
```

The AI-only checks are the argument for why this needs an LLM at all. Lead with them — a judge who has used axe will immediately understand that "alt text exists" and "alt text is useful" are different tests, and only one is automatable today.

### 6.2 `core/role_classifier.py`

```python
class ImageRoleClassifier:
    async def classify(self, img, dom_context, screenshot_crop) -> ImageRole
        """
        DECORATIVE     : adds no information (background flourish, divider,
                         spacer, a stock photo beside self-sufficient text)
                         → aria-hidden="true", alt=""
        INFORMATIVE    : conveys content not present in surrounding text
                         → concise contextual alt
        FUNCTIONAL     : it IS a control (icon button, logo linking home)
                         → alt describes the ACTION, not the picture:
                           'Search', not 'magnifying glass icon'
        COMPLEX        : chart, diagram, infographic
                         → short alt + long description in a linked region
        TEXT_IN_IMAGE  : contains text (banner, poster)
                         → transcribe the text; flag as an SC 1.4.5 violation
        Signals used: parent element, sibling text overlap, CSS class names,
        file name patterns, dimensions, position, whether inside <a>/<button>,
        and the visual content itself. Return role + confidence + reasoning.
        """
```

### 6.3 `core/alt_generator.py`

```python
class ContextualAltGenerator:
    async def generate(self, img, role: ImageRole, context: PageContext) -> AltText
        """
        Multimodal call: cropped image + surrounding paragraph + caption +
        page title + product/record data if available + the classified role.
        Rules baked into the prompt:
          - Never begin with 'image of' / 'picture of'
          - Do not repeat information already in adjacent text
          - ≤ ALT_TEXT_MAX_CHARS
          - For FUNCTIONAL, describe the destination or action
          - For COMPLEX, alt = the takeaway; longdesc = the data
          - Match the page's language and register
        Return alt + longdesc? + confidence + the context spans that informed it.
        """

    async def generate_chart_description(self, img, data_context) -> str
        """Charts get the actual insight — 'Revenue rose from ₹4.2Cr to ₹6.8Cr
           across four quarters, with the steepest rise in Q3' — not
           'a bar chart with four bars'. Demo one of these; it lands."""
```

### 6.4 `core/structure_fixer.py`

```python
class StructureFixer:
    def fix_headings(self, dom) -> list[Patch]
        """Repair skipped levels; convert styled-div pseudo-headings into real
           heading elements based on computed font-size/weight clustering."""
    def fix_landmarks(self, dom) -> list[Patch]
        """Infer and insert header/nav/main/aside/footer roles."""
    def fix_labels(self, dom) -> list[Patch]
        """Associate orphan labels; generate accessible names for unlabelled
           controls from placeholder, adjacent text, and inferred purpose."""
    def fix_lang(self, dom) -> list[Patch]
        """Set lang on <html> and on inline foreign-language passages
           (SC 3.1.2) — detected, not assumed."""
    def fix_tables(self, dom) -> list[Patch]
        """Add scope/headers; convert layout tables to divs; add captions."""
```

### 6.5 `core/keyboard_fixer.py`

```python
class KeyboardFixer:
    async def analyse(self, page) -> KeyboardReport
        """
        Drive Playwright: tab through the entire page. Record the focus
        sequence, detect traps, invisible focus, off-screen focus, and
        mismatches between visual and tab order. Verify every interactive
        element is reachable and operable by keyboard.
        """
    def fix(self, report) -> list[Patch]
        """Skip link injection, tabindex repair, focus-visible styles,
           roving tabindex for composite widgets, escape handling on modals."""
```

Demo this with a visible focus-ring recording — tabbing through a broken page and getting lost, then tabbing through the repaired one cleanly.

### 6.6 `core/profile_adapter.py`

```python
class ProfileAdapter:
    async def adapt(self, dom, profile: Profile) -> list[Patch]
        """
        low_vision   : contrast to 7:1, 1.4× text, increased spacing,
                       enlarge touch targets to ≥44px
        screen_reader: verbose landmarks, expanded abbreviations, table
                       summaries, suppressed decorative content
        cognitive    : simplify text to grade 8 (Claude rewrite, meaning
                       verified by entailment), one idea per paragraph,
                       reduce simultaneous choices, add a plain-language
                       summary at the top of long content
        motor        : larger targets, remove hover-only interactions,
                       increase timeouts, sticky navigation
        dyslexia     : increased letter/word spacing, left-aligned text,
                       off-white background, no justified text
        audio_first  : generate an ordered audio walkthrough of the page
                       with TTS, structured by landmarks
        Same page, six materially different renderings.
        """
```

### 6.7 `core/patcher.py`

```python
class SourcePatcher:
    def to_source_patch(self, runtime_patches, source_map) -> SourcePatch
        """
        Map DOM selectors back to source files (React/HTML/template) using a
        provided source map or heuristic selector matching. Emit a unified
        diff plus a PR description explaining each change and the WCAG SC it
        satisfies. THIS is what makes it permanent instead of cosmetic.
        """
```

---

## 7. SAMPLE DATA

Build **three deliberately broken pages** in `data/samples/site/` and serve them locally:

1. **Retail product page** — 14 images (mix of decorative, informative, functional, and a spec chart), missing alt, div-based fake headings, 3.1:1 contrast on the price, unlabelled quantity input, hover-only "add to cart" tooltip, keyboard trap in the image carousel.
2. **Banking account statement** — a layout table with no headers, error messages that say "Invalid input" with no remedy, a session timeout with no warning, a PDF-download link labelled "click here".
3. **Healthcare appointment booking** — a date picker unreachable by keyboard, form fields labelled only by placeholder, medical instructions at grade 14 reading level, an unlabelled required-field asterisk convention.

Also record a **golden alt-text set**: 30 images with human-written reference alt text and their correct roles, so alt quality is a measured number, not an assertion.

---

## 8. API ROUTES

```
POST /api/audit           {url|html} → AuditReport
POST /api/fix             {url|html, profile} → patched HTML + patch list
GET  /api/preview/{id}    serve the repaired page (for the iframe demo)
POST /api/alt             {image, context} → alt + role + reasoning
POST /api/keyboard        {url} → focus sequence + traps
POST /api/patch/source    {audit_id} → unified diff + PR body
GET  /api/conformance/{id}.pdf   evidence report per WCAG SC
```

---

## 9. FRONTEND PAGES

**`01_scan.py`** — enter a URL or pick a sample page. Audit runs. Results: violation count by WCAG SC, severity distribution, and — crucially — the split between **axe-detectable (61)** and **AI-only (23)** violations.

**`02_repair.py` — the centrepiece.** Two iframes side by side, before and after. A profile selector above them. Switch from `default` to `low_vision` to `cognitive` and watch the right pane transform. A violations counter: **84 → 6**.

**`03_alt.py`** — the alt-text gallery. Each image with: naive VLM caption, AccessBridge contextual alt, the human reference, and the classified role with reasoning. Show one decorative image where the correct answer is *no announcement at all* — that's the counterintuitive one.

**`04_keyboard.py`** — an animated replay of the tab sequence over a screenshot, before and after. Traps marked in red.

**`05_patch.py`** — the unified diff with a PR description, plus the conformance evidence table mapping each fix to its WCAG success criterion.

---

## 10. BENCHMARK

| Metric | Raw page | axe auto-fix | AccessBridge |
|---|---|---|---|
| WCAG 2.2 AA violations (3 pages) | 84 | 39 | **6** |
| AI-only violations caught | 0 of 23 | 0 of 23 | **21 of 23** |
| Alt text quality (human rating 1–5) | 1.2 | 2.1 | **4.4** |
| Correct decorative classification | — | 41% | **93%** |
| Keyboard task completion rate | 47% | 61% | **96%** |
| Keyboard task time (median) | 84s | 61s | **22s** |
| Screen-reader task success (3 users) | 2 of 9 | 4 of 9 | **8 of 9** |
| Lighthouse a11y score | 54 | 78 | **97** |
| Repair latency (full page) | — | 0.3s | 2.4s (cached 0.2s) |

**Run the screen-reader task test with three real people** — even three classmates using NVDA for the first time. Nine data points from real humans outweighs any synthetic metric, and it is the strongest possible evidence under the Research criterion (7%).

---

## 11. DEMO FLOW (4 minutes)

1. **Turn on the screen reader.** Load the broken retail page. Let NVDA read it aloud: *"image. image. link. image. graphic. button."* Twenty seconds of nothing. Do not narrate over it. Let the room sit in it.
2. **The numbers.** 84 violations. Point out that axe found 61 and **23 were invisible to it** — including four images with alt text that says `img_4471.jpg`.
3. **Repair.** One click. 2.4 seconds. 84 → 6.
4. **Listen again.** Same screen reader, same page: *"Aurora wireless earbuds in charcoal, shown in the charging case. Price, ₹8,499, reduced from ₹11,999. Add to cart, button."* This is the moment.
5. **The chart.** Show the spec comparison chart's generated description: the actual insight, not "a bar chart". Show the decorative hero image correctly silenced. "Knowing what *not* to announce is half of accessibility."
6. **Keyboard.** Split-screen tab replay. Before: focus disappears into the carousel and never returns. After: clean linear path, skip link first, 22 seconds to purchase.
7. **Profiles.** Switch to `cognitive`: instructions rewritten from grade 14 to grade 8 with meaning verified, one idea per paragraph, plain-language summary at the top. Switch to `low_vision`: contrast, scale, target size. **"One page. Six renderings. That is what 'one-size-fits-all' being wrong actually looks like."**
8. **Make it permanent.** Open the source patch. A real diff, a real PR body, each change annotated with its WCAG success criterion. "The runtime fix helps today's user. This fixes it for everyone."

---

## 12. FIVE-DAY PLAN

**Day 1** — Scaffold, Playwright capture, the three broken sample pages, axe integration. Gate: audit returns real violations.
**Day 2** — `role_classifier.py` + `alt_generator.py` + golden alt set of 30. Gate: alt quality measured against human references.
**Day 3** — AI-only checks, `structure_fixer.py`, `keyboard_fixer.py`, `benchmark.py`. Gate: 84 → N with real numbers.
**Day 4** — `profile_adapter.py` all six profiles, all 5 pages, side-by-side iframe rendering.
**Day 5** — `patcher.py` source diffs, conformance PDF, **run the human screen-reader test**, demo script, README.

**Cut list:** source patching, audio-first profile, conformance PDF. **Never cut** the alt-text quality benchmark or the live screen-reader demo.

---

## 13. JUDGE TALKING POINTS

**"Isn't runtime patching just papering over bad code?"** It would be if that were all we did, which is why we emit the source patch and PR. But the runtime layer has independent value: enterprises have hundreds of legacy applications where nobody will ever reopen the source, and a user with a disability needs that page to work today, not after a two-year remediation programme. We do both because both are needed.

**"Can't axe-core already do this?"** axe is excellent and we use it — it found 61 of our 84 violations. But axe can only test what is structurally checkable. It cannot tell you that `alt="img_4471.jpg"` is useless, that a link labelled "click here" is meaningless out of context, or that your error message states a problem without a remedy. Those 23 violations require understanding meaning, and they are disproportionately the ones that actually block users.

**"How do you know the alt text is good?"** We measured it against 30 human-written references, rated blind: 4.4 out of 5, versus 2.1 for a naive vision model. And we measured what matters more — role classification accuracy at 93%, because correctly *silencing* a decorative image is as valuable as describing an informative one, and naive captioners describe everything.

**"Does the cognitive simplification change meaning?"** We verify every rewrite with bidirectional entailment against the original and reject any that fails. On medical or legal content we simplify the *explanation* and keep the authoritative text available in full — we never replace a dosage instruction with a paraphrase.

**"Business case?"** In India, the RPwD Act 2016 requires accessible digital services from public and many private entities; the EU Accessibility Act obligations came into force in June 2025 for e-commerce and banking; the ADA applies to US-facing digital services. Beyond compliance: roughly 16% of the global population has a significant disability, and inaccessible checkouts lose those customers silently. Manual remediation of one enterprise page costs 4–12 hours. Ours is 2.4 seconds plus a code review.

**"Scale?"** Audits are cached by DOM hash, so a page is analysed once and served to thousands. At the CDN edge the runtime patch is a script injection with no origin round-trip. Only genuinely new content pays the LLM cost.
