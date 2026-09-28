/-
  The existence theorem for Eisenstein congruences on a finite class set — the elementary core,
  formalised.  (Octonion Program, 23 September 2026.)
  Lean 4 v4.34.0, Mathlib v4.34.0.  Builds with `lake build`; no `sorry`;
  #print axioms: [propext, Classical.choice, Quot.sound] for all three theorems.

  Setting.  X is a finite set (the class set of a compact group at some level), e : X → ℤ the
  scaled mass weights e_i = L/|Γ_i| (so ∑ e_i = L·mass), assumed coprime via a Bezout witness c.
  The cuspidal lattice is C = { f : X → ℤ | ∑ e_i f_i = 0 }.  A Hecke operator is an integer
  matrix T with constant row sums N (the constant function is an eigenvector) and the
  mass-weighted symmetry e_i T_ij = e_j T_ji.

  Theorem A (mass_criterion):  ℓ ∣ E  ↔  ∃ f ∈ C with f ≡ 1 (mod ℓ) coordinatewise.
  Theorem B (cuspidal_stable):  C is stable under every Hecke operator.
  Theorem C (eisenstein_component_to_mass):  if f ∈ C is congruent mod a prime ℓ to a nonzero
     multiple of the constant function, then ℓ ∣ E.
  (The remaining step of the paper theorem — the Eisenstein idempotent and Deligne–Serre — is
  not formalised here.)
-/
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Data.Matrix.Basic
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Ring
import Mathlib.Algebra.Prime.Defs

open Finset

namespace Eisenstein

variable {X : Type*} [Fintype X]

/-- **Theorem A (the mass criterion).**  With `E = ∑ i, e i` and a Bezout witness `c`
(`∑ e i * c i = 1`, i.e. the `e i` are coprime), an integer `ℓ` divides `E` iff there is a
cuspidal integer function (`∑ e i * f i = 0`) congruent to `1` modulo `ℓ` at every class. -/
theorem mass_criterion (e c : X → ℤ) (hc : ∑ i, e i * c i = 1) (ℓ : ℤ) :
    (∃ f : X → ℤ, ∑ i, e i * f i = 0 ∧ ∀ i, ℓ ∣ f i - 1) ↔ ℓ ∣ ∑ i, e i := by
  constructor
  · rintro ⟨f, hf, hℓ⟩
    choose u hu using hℓ
    have key : ∑ i, e i * f i = ∑ i, e i + ℓ * ∑ i, e i * u i := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      refine Finset.sum_congr rfl ?_
      intro i _
      linear_combination e i * hu i
    rw [hf] at key
    exact ⟨-(∑ i, e i * u i), by linear_combination (-1 : ℤ) * key⟩
  · rintro ⟨t, ht⟩
    refine ⟨fun i => 1 - ℓ * (t * c i), ?_, ?_⟩
    · have : ∑ i, e i * (1 - ℓ * (t * c i)) = ∑ i, e i - ℓ * t * ∑ i, e i * c i := by
        rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl ?_
        intro i _
        ring
      rw [this, hc, ht]
      ring
    · intro i
      exact ⟨-(t * c i), by ring⟩

/-- **Theorem B.**  A Hecke operator (constant row sums `N`, mass-weighted symmetry) preserves
the cuspidal lattice. -/
theorem cuspidal_stable (e : X → ℤ) (T : Matrix X X ℤ) (N : ℤ)
    (h1 : ∀ i, ∑ j, T i j = N) (h2 : ∀ i j, e i * T i j = e j * T j i)
    (f : X → ℤ) (hf : ∑ i, e i * f i = 0) : ∑ i, e i * (T.mulVec f) i = 0 := by
  have key : ∑ i, e i * (T.mulVec f) i = N * ∑ j, e j * f j := by
    simp only [Matrix.mulVec, dotProduct]
    calc ∑ i, e i * ∑ j, T i j * f j
        = ∑ i, ∑ j, e i * T i j * f j := by
          refine Finset.sum_congr rfl ?_
          intro i _
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro j _
          ring
      _ = ∑ j, ∑ i, e j * T j i * f j := by
          rw [Finset.sum_comm]
          refine Finset.sum_congr rfl ?_
          intro j _
          refine Finset.sum_congr rfl ?_
          intro i _
          rw [h2]
      _ = ∑ j, e j * f j * ∑ i, T j i := by
          refine Finset.sum_congr rfl ?_
          intro j _
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro i _
          ring
      _ = ∑ j, e j * f j * N := by
          refine Finset.sum_congr rfl ?_
          intro j _
          rw [h1]
      _ = N * ∑ j, e j * f j := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro j _
          ring
  rw [key, hf, mul_zero]

/-- **Theorem C.**  If a cuspidal integer function is congruent, modulo a prime `ℓ`, to `c` times
the constant function with `ℓ ∤ c`, then `ℓ ∣ E`.  (The direction (iii) ⇒ (i) of the paper
theorem, with the Eisenstein component replaced by any cuspidal function.) -/
theorem eisenstein_component_to_mass (e : X → ℤ) (ℓ : ℤ) (hℓ : Prime ℓ) (c : ℤ) (hc : ¬ ℓ ∣ c)
    (f : X → ℤ) (hf : ∑ i, e i * f i = 0) (hcong : ∀ i, ℓ ∣ f i - c) : ℓ ∣ ∑ i, e i := by
  choose u hu using hcong
  have key : ∑ i, e i * f i = c * ∑ i, e i + ℓ * ∑ i, e i * u i := by
    rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl ?_
    intro i _
    linear_combination e i * hu i
  rw [hf] at key
  have h : ℓ ∣ c * ∑ i, e i := ⟨-(∑ i, e i * u i), by linear_combination (-1 : ℤ) * key⟩
  exact (hℓ.dvd_or_dvd h).resolve_left hc

end Eisenstein
