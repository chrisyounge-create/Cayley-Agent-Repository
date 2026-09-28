# Cayley Agent Repository — Lean certificates for the Octonion Program

`lean/` is a Lean 4 (v4.34.0) + Mathlib project, compiled by GitHub Actions on every push (see the **Actions** tab for the log).

| file | content | status |
|---|---|---|
| `Eisenstein.lean` | elementary core of the existence theorem for Eisenstein congruences on a finite class set (`mass_criterion`, `cuspidal_stable`, `eisenstein_component_to_mass`) | certified, no `sorry` |
| `LogMarkov.lean` | Lemma B of the Galois-orbit note (log-Markov averaging inequality) | certified, no `sorry`, axioms propext/Classical.choice/Quot.sound |
| `EisensteinComponent.lean` | the two remaining steps of the existence theorem: `eisenstein_idempotent_step` (given a complete orthogonal family of idempotents of the Hecke algebra, the Eisenstein idempotent fixes the constant vector and carries a cuspidal `f ≡ 1` to a cuspidal element of the Eisenstein component that is still `≡ 1`) and `deligne_serre_prime` / `deligne_serre_lift` (algebraic core of the Deligne–Serre lemma: a torsion-free algebra over a domain has, below any prime `𝔫`, a prime `P` with `P ∩ O = 0`, giving a lifted eigen-system `A → A ⧸ P` into a domain, injective on `O`) | certified, no `sorry` |

Paper: C. Younge, *Class sets and Eisenstein congruences on compact exceptional groups* (Parts I and II), working drafts.
