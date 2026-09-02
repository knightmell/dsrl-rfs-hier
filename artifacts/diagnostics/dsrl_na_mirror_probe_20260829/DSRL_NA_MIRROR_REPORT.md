# DSRL-NA Mirror Probe Report

Run: `2026-08-29T12:18:56.565961+00:00` to `2026-08-29T12:19:21.556968+00:00`

## Scope and comparability

This is a strictly read-only geometry probe, not a best-of-K execution test. Both matched source variants had a real 50k checkpoint. Requested 100k–250k points therefore map to actual 50k with explicit paths and hashes in the JSON.

VS-Hier comparison is **unavailable**: the existing fixed-bank WBSD sweep is at actual 100k/300k/500k, not actual 50k. No cross-step superiority claim is made.

## Shared diagnostic contract

- 512 fixed WBSD states; state array SHA-256 `011b842acc0553417fa7cb4ac4ca7f8965f0d0e805a5371bfd684900d94315de`.
- Candidate sources: direct standard Gaussian `w ~ N(0,I)` and actor-reachable tanh-Gaussian; K=64.
- Proposal seeds / CRN: `[1101, 2202, 3303, 4404]`; candidate IDs are fixed prefixes, hash `9337f1bf62919f8e7241d0be6913450095668ca1eb90364745c1b647d2d4679c`.
- QW and QA use twin-min values. Direction uses only `torch.autograd.grad` with respect to input w; every parameter is frozen.

## Priority-ordered results


### current_actor / standard_gaussian / actual 50k

- Directional gain: eps 0.01 P(ΔQA>0)=0.7137, mean ΔQA=0.027876; eps 0.03 P=0.7145, mean=0.083538; eps 0.1 P=0.7149, mean=0.277800.
- Ranking: pairwise=0.5615; Spearman=0.1789.
- Selection: normalized top-1 regret=0.3515; top-1 agreement=0.0347.
- Twin disagreement: QW=27.353041; QA=10.276218. Exploitation gap=69.867018.

### current_actor / actor_reachable / actual 50k

- Directional gain: eps 0.01 P(ΔQA>0)=0.6866, mean ΔQA=0.023260; eps 0.03 P=0.6877, mean=0.069714; eps 0.1 P=0.6870, mean=0.231481.
- Ranking: pairwise=0.5278; Spearman=0.0801.
- Selection: normalized top-1 regret=0.4301; top-1 agreement=0.0151.
- Twin disagreement: QW=7.406082; QA=6.765836. Exploitation gap=5.605740.

### gaussian / standard_gaussian / actual 50k

- Directional gain: eps 0.01 P(ΔQA>0)=0.7503, mean ΔQA=0.053732; eps 0.03 P=0.7516, mean=0.161010; eps 0.1 P=0.7516, mean=0.534219.
- Ranking: pairwise=0.5797; Spearman=0.2292.
- Selection: normalized top-1 regret=0.3375; top-1 agreement=0.0581.
- Twin disagreement: QW=10.168274; QA=10.303771. Exploitation gap=9.004440.

### gaussian / actor_reachable / actual 50k

- Directional gain: eps 0.01 P(ΔQA>0)=0.6851, mean ΔQA=0.039288; eps 0.03 P=0.6870, mean=0.117698; eps 0.1 P=0.6870, mean=0.390863.
- Ranking: pairwise=0.5339; Spearman=0.0975.
- Selection: normalized top-1 regret=0.4209; top-1 agreement=0.0293.
- Twin disagreement: QW=8.625304; QA=8.131776. Exploitation gap=3.073231.

## Read-only audit

Audit passed: **True**. Environment interactions=0; optimizer constructions/steps=0; backward calls=0; global RNG restored=True; all module hashes unchanged and parameter grads remained absent=True.

## Interpretation

The DSRL-NA variants can be characterized internally at 50k, but a directional/ranking advantage over VS-Hier cannot be established without a VS-Hier mirror at the same actual step and identical candidates. The required four-way label is therefore assigned conservatively and marked inconclusive.

**M3 DSRL ranking+gradient 均类似 — inconclusive.**
