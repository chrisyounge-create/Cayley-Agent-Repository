import Mathlib

open Real Finset

/-- `log z ≤ z / e` for `z > 0`. -/
lemma log_le_div_e {z : ℝ} (hz : 0 < z) : Real.log z ≤ z / Real.exp 1 := by
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have h1 : Real.log (z / Real.exp 1) ≤ z / Real.exp 1 - 1 :=
    Real.log_le_sub_one_of_pos (div_pos hz he)
  rw [Real.log_div hz.ne' he.ne', Real.log_exp] at h1
  linarith

/-- Lemma B (log-Markov): if positive reals on `s` have arithmetic mean `1` and mean logarithm `≥ -β`,
then the number of them below `ε` times `log (1/ε)` is at most `(β + 1/e) * #s`. -/
theorem log_markov {ι : Type*} (s : Finset ι) (x : ι → ℝ) (hx : ∀ i ∈ s, 0 < x i)
    (hsum : ∑ i ∈ s, x i = (s.card : ℝ)) (β : ℝ)
    (hlog : -(β * s.card) ≤ ∑ i ∈ s, Real.log (x i))
    (ε : ℝ) (hε0 : 0 < ε) (hε1 : ε < 1) :
    ((s.filter (fun i => x i < ε)).card : ℝ) * Real.log (1 / ε)
      ≤ (β + 1 / Real.exp 1) * s.card := by
  classical
  set S := s.filter (fun i => x i < ε) with hS
  set T := s.filter (fun i => ¬ x i < ε) with hT
  set n : ℝ := (s.card : ℝ) with hn
  set t : ℝ := (T.card : ℝ) with ht
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  -- split the sum of logs
  have hsplit : ∑ i ∈ s, Real.log (x i) = ∑ i ∈ S, Real.log (x i) + ∑ i ∈ T, Real.log (x i) := by
    rw [hS, hT]; exact (Finset.sum_filter_add_sum_filter_not s (fun i => x i < ε) _).symm
  -- small values: log x_i ≤ log ε
  have hSle : ∑ i ∈ S, Real.log (x i) ≤ (S.card : ℝ) * Real.log ε := by
    have : ∀ i ∈ S, Real.log (x i) ≤ Real.log ε := by
      intro i hi
      rw [hS, Finset.mem_filter] at hi
      exact Real.log_le_log (hx i hi.1) (le_of_lt hi.2)
    calc ∑ i ∈ S, Real.log (x i) ≤ ∑ i ∈ S, Real.log ε := Finset.sum_le_sum this
      _ = (S.card : ℝ) * Real.log ε := by rw [Finset.sum_const, nsmul_eq_mul]
  -- the rest: sum of logs ≤ n / e
  have hTle : ∑ i ∈ T, Real.log (x i) ≤ n / Real.exp 1 := by
    by_cases hT0 : T.card = 0
    · rw [Finset.card_eq_zero.mp hT0, Finset.sum_empty]; positivity
    · have htpos : 0 < t := by rw [ht]; exact_mod_cast Nat.pos_of_ne_zero hT0
      have hTsub : T ⊆ s := Finset.filter_subset _ _
      have hnpos : 0 < n := by
        rw [hn]; exact_mod_cast (Finset.card_pos.mpr ⟨_, hTsub (Finset.card_pos.mp (Nat.pos_of_ne_zero hT0)).choose_spec⟩)
      -- log x ≤ x * (t/n) - 1 - log (t/n)
      have hpt : ∀ i ∈ T, Real.log (x i) ≤ x i * (t / n) - 1 - Real.log (t / n) := by
        intro i hi
        have hxi : 0 < x i := hx i (hTsub hi)
        have hq : 0 < t / n := div_pos htpos hnpos
        have := Real.log_le_sub_one_of_pos (mul_pos hxi hq)
        rw [Real.log_mul hxi.ne' hq.ne'] at this
        linarith
      have hsumT : ∑ i ∈ T, x i ≤ n := by
        rw [← hsum]
        exact Finset.sum_le_sum_of_subset_of_nonneg hTsub (fun i hi _ => le_of_lt (hx i hi))
      calc ∑ i ∈ T, Real.log (x i) ≤ ∑ i ∈ T, (x i * (t / n) - 1 - Real.log (t / n)) := Finset.sum_le_sum hpt
        _ = (∑ i ∈ T, x i) * (t / n) - t - t * Real.log (t / n) := by
            rw [Finset.sum_sub_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul]
            simp [Finset.sum_const, nsmul_eq_mul, ht]
        _ ≤ n * (t / n) - t - t * Real.log (t / n) := by
            have : 0 ≤ t / n := le_of_lt (div_pos htpos hnpos)
            nlinarith
        _ = t * Real.log (n / t) := by
            rw [Real.log_div hnpos.ne' htpos.ne', Real.log_div htpos.ne' hnpos.ne']
            field_simp; ring
        _ ≤ t * ((n / t) / Real.exp 1) := by
            exact mul_le_mul_of_nonneg_left (log_le_div_e (div_pos hnpos htpos)) (le_of_lt htpos)
        _ = n / Real.exp 1 := by field_simp
  have hlogε : Real.log (1 / ε) = - Real.log ε := by
    rw [one_div, Real.log_inv]
  rw [hlogε]
  rw [hsplit] at hlog
  nlinarith [hSle, hTle, hlog]
