# Forum post draft (copy into Challenge Discussion)

**Title:** Phase 1 write-up: Orthogonality, antipodes, and a depth-32 falsification map (submission #325459)

**Body:**

---

Technical write-up for the Algorithmic Contribution Prize, tied to graded submission **#325459** (`pure-0.2.1s`, adj **3.202e-07**).

**Method.** Sylvester–Hadamard ONB frames + random row perm / column signs + antipodes + χ-mean radius, util ≈0.40 float32. Structural claim: ONB+antipodes fix the first two angular moments of the direction measure, so remaining error is nearly pure sampling variance (bias–variance Mini-20 check consistent with ≈0 bias).

**Contribution (mostly negative results).** Falsification map under seed protocol 3.0: sign redesigns, partial ONBs, robust aggregators, JS/gain, last-layer Gaussian closures, residual MLPs / L30 recenter, Mini-tuned util/gain (Mini↑/online↓), and post–v0.10 residual/OpenBLAS packs (stable ~3.47e-07; one false win failed reconfirm). None beats locked Hadamard on official adj. Port of public moment chain #314695 improves kprop but stays several× worse than Hadamard on Mini raw MSE → Phase-2 axis.

**Attachment:** `phase1_algo_contribution.pdf`  
**LLM disclosure:** research + drafting AI-assisted; numbers tied to probe artifacts / graded subs.

---

# Email draft (private channel)

**To:** arc-whestbench@aicrowd.com  
**Subject:** Phase 1 Algorithmic Contribution write-up — submission #325459

Hello,

Attached is the Phase 1 Algorithmic Contribution write-up for graded submission **#325459** (`pure-0.2.1s`, adjusted 3.202e-07).

- PDF: `phase1_algo_contribution.pdf` (also in zip `phase1_algo_contribution_325459.zip`)
- Private re-eval nominations (when your nomination email arrives): **#325459** and **#325112** (same estimator)

LLM assistance is disclosed in the PDF.

Thank you,  
[Your name / team]
