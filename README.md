# Cayley Agent Repository — Lean certificates for the Octonion Program

`lean/` is a Lean 4 (v4.34.0) + Mathlib project, compiled by GitHub Actions on every push (see the **Actions** tab for the log).

| file | content | status |
|---|---|---|
| `Eisenstein.lean` | elementary core of the existence theorem for Eisenstein congruences on a finite class set (`mass_criterion`, `cuspidal_stable`, `eisenstein_component_to_mass`) | certified, no `sorry` |
| `LogMarkov.lean` | Lemma B of the Galois-orbit note (log-Markov averaging inequality) | certified, no `sorry`, axioms propext/Classical.choice/Quot.sound |
| `EisensteinComponent.lean` | statements of the two remaining steps (Eisenstein idempotent, Deligne–Serre) | statements typecheck; proofs are `sorry` (in progress) |

Paper: C. Younge, *Class sets and Eisenstein congruences on compact exceptional groups* (Parts I and II), working drafts.
