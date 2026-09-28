/-
  Octonion Program — the two remaining steps of the existence theorem (Paper I, Theorem 2.1):
  the Eisenstein-idempotent step and the algebraic core of the Deligne–Serre lemma.
  28 September 2026.  Compiled by GitHub Actions against Mathlib.

  Conventions.  `A` is the (commutative) Hecke algebra acting on the module `M` (the O-valued functions
  on the class set); `C ≤ M` is the cuspidal submodule, assumed `A`-stable; `one : M` is the constant
  vector, on which `A` acts through the Eisenstein character `χ : A →+* O` (`a • one = χ a • one`).
  The decomposition of `A` into local pieces enters as a complete family of orthogonal idempotents
  `e : ι → A` — its existence for a finite algebra over a complete DVR is the standard structure
  theorem and is taken as a hypothesis here.
-/
import Mathlib

namespace Eisenstein

section IdempotentStep

variable {O : Type*} [CommRing O] [IsDomain O]
variable {A : Type*} [CommRing A] [Algebra O A]
variable {M : Type*} [AddCommGroup M] [Module O M] [Module A M] [IsScalarTower O A M]

/-- An idempotent of a commutative ring maps to `0` or `1` under a ring homomorphism to a domain. -/
lemma idempotent_map_domain (χ : A →+* O) {e : A} (he : IsIdempotentElem e) :
    χ e = 0 ∨ χ e = 1 := by
  have h : χ e * χ e = χ e := by rw [← map_mul, he.eq]
  have : χ e * (χ e - 1) = 0 := by ring_nf; linear_combination h
  rcases mul_eq_zero.mp this with h0 | h1
  · exact Or.inl h0
  · exact Or.inr (sub_eq_zero.mp h1)

/-- A complete orthogonal family of idempotents has exactly one member mapping to `1` under a ring
homomorphism to a domain. -/
lemma exists_unique_idempotent_map_one {ι : Type*} [Fintype ι] [DecidableEq ι]
    (χ : A →+* O) (e : ι → A) (hid : ∀ i, IsIdempotentElem (e i))
    (horth : ∀ i j, i ≠ j → e i * e j = 0) (hsum : ∑ i, e i = 1) :
    ∃ j, χ (e j) = 1 ∧ ∀ i, i ≠ j → χ (e i) = 0 := by
  have h1 : ∑ i, χ (e i) = 1 := by rw [← map_sum, hsum, map_one]
  have hex : ∃ j, χ (e j) ≠ 0 := by
    by_contra hcon
    push Not at hcon
    have : ∑ i, χ (e i) = 0 := Finset.sum_eq_zero (fun i _ => hcon i)
    rw [h1] at this; exact one_ne_zero this
  obtain ⟨j, hj⟩ := hex
  refine ⟨j, ?_, ?_⟩
  · rcases idempotent_map_domain χ (hid j) with h0 | h1'
    · exact absurd h0 hj
    · exact h1'
  · intro i hij
    have : χ (e i) * χ (e j) = 0 := by rw [← map_mul, horth i j hij, map_zero]
    rcases mul_eq_zero.mp this with h | h
    · exact h
    · exact absurd h hj

/-- **The Eisenstein-idempotent step.**  Let `e` be a complete orthogonal family of idempotents of
the Hecke algebra `A` acting on `M`, `C` an `A`-stable submodule, `one ∈ M` an `A`-eigenvector with
character `χ : A →+* O`, and `f₀ ∈ C` with `f₀ ≡ one` modulo `ℓ • M` for a scalar `ℓ : O`.  Then some
`e j` fixes `one`, and `e j • f₀` lies in `C`, in the component `e j • M`, and is `≡ one` modulo `ℓ • M`. -/
theorem eisenstein_idempotent_step {ι : Type*} [Fintype ι] [DecidableEq ι]
    (χ : A →+* O) (e : ι → A) (hid : ∀ i, IsIdempotentElem (e i))
    (horth : ∀ i j, i ≠ j → e i * e j = 0) (hsum : ∑ i, e i = 1)
    (one : M) (hone : ∀ a : A, a • one = χ a • one)
    (C : Submodule O M) (hC : ∀ a : A, ∀ f ∈ C, a • f ∈ C)
    (ℓ : O) (f₀ : M) (hf₀ : f₀ ∈ C) (hcong : ∃ u : M, f₀ - one = ℓ • u) :
    ∃ j, e j • one = one ∧ e j • f₀ ∈ C ∧ (∃ m : M, e j • f₀ = e j • m) ∧
      (∃ u : M, e j • f₀ - one = ℓ • u) := by
  obtain ⟨j, hj1, -⟩ := exists_unique_idempotent_map_one χ e hid horth hsum
  have hfix : e j • one = one := by rw [hone, hj1, one_smul]
  refine ⟨j, hfix, hC _ _ hf₀, ⟨f₀, rfl⟩, ?_⟩
  obtain ⟨u, hu⟩ := hcong
  refine ⟨e j • u, ?_⟩
  calc e j • f₀ - one = e j • f₀ - e j • one := by rw [hfix]
    _ = e j • (f₀ - one) := by rw [smul_sub]
    _ = e j • (ℓ • u) := by rw [hu]
    _ = ℓ • (e j • u) := smul_comm _ _ _

end IdempotentStep

section DeligneSerre

variable {O : Type*} [CommRing O] [IsDomain O]
variable {A : Type*} [CommRing A] [Algebra O A]

/-- **Deligne–Serre, algebraic core.**  If `A` is a torsion-free `O`-algebra (`O` a domain), then
every prime ideal `𝔫` of `A` contains a prime `P` with `P ∩ O = 0`.  Consequently `A ⧸ P` is a
domain into which `O` embeds, and the quotient map `A → A ⧸ P` is a system of eigenvalues
(a ring homomorphism to a domain) whose reduction modulo `𝔫 ⧸ P` is the given one. -/
theorem deligne_serre_prime [NoZeroSMulDivisors O A]
    (𝔫 : Ideal A) [𝔫.IsPrime] :
    ∃ P : Ideal A, P.IsPrime ∧ P ≤ 𝔫 ∧ P.comap (algebraMap O A) = ⊥ := by
  -- a minimal prime `P` below `𝔫` consists of zero-divisors; torsion-freeness forces `P ∩ O = 0`
  obtain ⟨P, hPmin, hP𝔫⟩ := Ideal.exists_minimalPrimes_le (I := (⊥ : Ideal A)) (J := 𝔫) bot_le
  refine ⟨P, hPmin.1.1, hP𝔫, ?_⟩
  refine le_antisymm ?_ bot_le
  intro o ho
  rw [Ideal.mem_comap] at ho
  obtain ⟨y, hy, hxy⟩ := Ideal.exists_mul_mem_of_mem_minimalPrimes hPmin ho
  have hy0 : y ≠ 0 := fun h => hy (by rw [h]; exact Ideal.zero_mem _)
  have hoy : o • y = 0 := by rw [Algebra.smul_def]; exact Ideal.mem_bot.mp hxy
  rcases smul_eq_zero.mp hoy with h | h
  · exact Ideal.mem_bot.mpr h
  · exact absurd h hy0

/-- The lifted system of eigenvalues: `A → A ⧸ P` is a ring homomorphism to a domain, restricting
to an injection on `O` (so `Frac(A ⧸ P)` is a field extension of `Frac(O)`, finite when `A` is a
finite `O`-algebra), and `P ≤ 𝔫` says it reduces modulo `𝔫 ⧸ P` to the given system. -/
theorem deligne_serre_lift [NoZeroSMulDivisors O A]
    (𝔫 : Ideal A) [𝔫.IsPrime] :
    ∃ P : Ideal A, P.IsPrime ∧ P ≤ 𝔫 ∧
      Function.Injective ((Ideal.Quotient.mk P).comp (algebraMap O A)) := by
  obtain ⟨P, hP, hP𝔫, hcomap⟩ := deligne_serre_prime (O := O) 𝔫
  refine ⟨P, hP, hP𝔫, ?_⟩
  rw [RingHom.injective_iff_ker_eq_bot, ← RingHom.comap_ker, Ideal.mk_ker]
  exact hcomap

end DeligneSerre

end Eisenstein
