# Phase 1 — human submit checklist

Local artifacts are ready. **You** must post / email (no AIcrowd API from this agent).

## Deadlines (do not conflate)

| What | Due (UTC) | Notes |
|------|-----------|-------|
| Phase 1 **code** submissions | **10 Aug 2026 23:59** | `estimator.py` already graded; no new Phase-1 slots needed |
| Algorithmic Contribution **write-up** | **17 Aug 2026 23:59** | Forum post or email — **not** required today |
| Private re-eval nominations | when organizers email | Keep #325459 + #325112 noted |

Write-up can wait until 8/17. Posting earlier is optional (community visibility), not a Phase-1 code deadline.

## A. Algorithmic Contribution write-up (due **17 Aug 2026 23:59 UTC**)

Per forum [#18041](https://discourse.aicrowd.com/t/18041) / [#18125](https://discourse.aicrowd.com/t/18125):

| Field | Value |
|-------|-------|
| Submission ID (exactly one) | **#325459** |
| PDF | `writeups/dist/phase1_algo_contribution.pdf` |
| Channel (either) | Forum **or** email `arc-whestbench@aicrowd.com` |
| Forum also counts for | Community Contribution prizes |

**Forum:** copy title + body from `POSTING_DRAFTS.md` § “Forum — ready to paste”. Attach the PDF.  
**Email:** copy from `POSTING_DRAFTS.md` § “Email — ready to paste”. Attach the PDF.

## B. Private re-eval nominations (up to 2)

Organizers said they will **email** nomination instructions ([#18125](https://discourse.aicrowd.com/t/18125)). Until then, keep:

1. **#325459** — `pure-0.2.1s` (current evaluator)  
2. **#325112** — same estimator (confirmation)

If you nominate none, they take your top-2 public scores (also fine for us — same method).

## C. Do not

- New util / gain / residual submissions  
- Change `estimator.py` off `pure-0.2.1s`  
- Nominate residual-pack IDs  

## D. Regen PDF after edits

```bash
.venv/bin/python writeups/regen_pdf.py
```
