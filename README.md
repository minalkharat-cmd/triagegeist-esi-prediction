# Triagegeist: ESI Triage Prediction with Explainability, Fairness & Uncertainty Quantification

Submission for the [Kaggle Triagegeist Hackathon](https://www.kaggle.com/competitions/triagegeist) — a clinical ML challenge on predicting Emergency Severity Index (ESI) levels from structured intake data and free-text chief complaints, with emphasis on clinical relevance, fairness, and deployability.

## TL;DR

A LightGBM + CatBoost ensemble (60/40) trained on 80,000 synthetic ED encounters achieves:

- **OOF Quadratic Weighted Kappa: 0.9991** (LGBM alone: 0.9990, CatBoost alone: 0.9991)
- - **OOF Accuracy: 0.9985**
  - - **OOF Macro F1: 0.9963**
    - - **Undertriage: 0.104%** (pred > y — model predicts higher ESI number = less urgent than actual; ESI-1 is most urgent/lowest number)
      - - **Overtriage: 0.049%** (pred < y — resource-waste direction)
        - - **Critical ESI-1 misses: 53 of 80,000**
         
          - Paired with SHAP per-class explanations, a demographic equity audit with actual numbers (see below), a calibrated uncertainty-flagging layer for human-in-the-loop review, ESI-1 calibration curve, per-fold QWK stability chart, clinically annotated confusion matrix, and LightGBM learning curve.
         
          - ## Why this matters clinically
         
          - Every minute in the ED counts. Triage nurses and physicians must rapidly assign each arriving patient an ESI level (1 = immediate, life-threatening to 5 = non-urgent) under extreme cognitive load, incomplete information, and chronic understaffing. The ESI system, while widely adopted, relies entirely on unaided human judgment. Inter-rater variability is well-documented in the literature, and systematic undertriage of elderly patients, non-English speakers, and the uninsured is an active patient-safety concern.
         
          - This tool is designed to run at the point of triage intake, before the physician sees the patient, using only data available in the first 90 seconds of contact: structured vitals, demographics, comorbidity flags, and the free-text chief complaint typed by the nurse.
         
          - This project addresses a focused, high-stakes question: can an ML system trained on structured intake data plus free-text chief complaints reliably predict ESI acuity, explain its reasoning clinically, flag cases of genuine uncertainty, and surface potential equity gaps?
         
          - ## Architecture
         
          - - **Data ingestion:** train + chief_complaints + patient_history, joined on patient_id
            - - **Leakage removal:** ed_los_hours, disposition, triage_nurse_id dropped before training (community discussion confirmed retaining these yields near-100% accuracy on a clinically useless model)
              - - **Feature engineering:** clinical composites (shock_index, MAP, NEWS2, BMI), informative-missingness indicators, vital-age interactions, TF-IDF + TruncatedSVD (40 components) + 16 keyword flags on chief complaints, operational categoricals
                - - **Ensemble:** LightGBM (60%) + CatBoost (40%), 5-fold stratified CV, OOF predictions
                  - - **Hyperparameter tuning:** manual grid search over num_leaves in {63,127,255}, learning_rate in {0.02,0.05,0.1}, feature_fraction in {0.7,0.8,0.9}; CatBoost depth=8, lr=0.05, 1000 iterations
                    - - **Explainability:** SHAP per-class, equity audit by age/sex/insurance/language, calibrated uncertainty flagging
                     
                      - ## Reported metrics (V13 — final)
                     
                      - | Metric | Value | Why it matters |
                      - |--------|-------|----------------|
                      - | QWK | 0.9991 | Ordinal agreement with ground truth |
                      - | Macro-F1 | 0.9963 | Balanced per-class performance under imbalance |
                      - | Accuracy | 0.9985 | Overall correctness |
                      - | Undertriage rate | 0.104% | pred > y (model predicts higher ESI = less urgent than actual) |
                      - | Overtriage rate | 0.049% | pred < y (model predicts lower ESI = more urgent than actual) |
                      - | ESI-1 misses | 53 / 80,000 | Absolute count of missed critical patients |
                     
                      - **V9 bug fix (critical):** V8 had undertriage/overtriage operators swapped. Corrected in V9: undertriage = (pred > y).mean(), overtriage = (pred < y).mean(). In ESI convention ESI-1 is most urgent (lowest number), so predicting a higher ESI number is undertriage (dangerous).
                     
                      - ## Technical Quality Additions (V13)
                     
                      - V13 adds four new analysis sections to fully address clinical and technical robustness:
                     
                      - - **Section 13 — Clinical Scope & Model Boundaries:** Explicit statement of what the model cannot do — psychiatric boarding, pediatric (<18), MTS-protocol EDs, high-missingness cohorts, re-triage after initial assessment. Prevents deployment in contexts where the model was not validated.
                        - - **Section 14 — Per-Fold QWK Stability + ESI-1 Calibration Curve:** Bar chart showing QWK variance across all 5 CV folds (demonstrates model stability, not single-fold luck). Reliability diagram comparing ESI-1 predicted probability vs actual fraction positive — confirms the model is not just accurate but well-calibrated for clinical decision support.
                          - - **Section 15 — Clinically Annotated Confusion Matrix:** Confusion matrix with red highlighting on the most dangerous misclassifications (ESI-1 predicted as ESI-3+) and orange on high-risk cells (ESI-1 → ESI-2). Includes ED-scale interpretation: "In a 50,000-patient ED, X patients per year would be undertriaged into a non-urgent queue."
                            - - **Section 16 — LightGBM Learning Curve:** Train vs. validation loss across boosting rounds with early stopping marker and overfitting region shading — demonstrates that the model generalises and did not overfit to training data.
                             
                              - ## Fairness Audit Results (V13)
                             
                              - All groups: QWK >= 0.999. Undertriage = pred > y; overtriage = pred < y.
                             
                              - **By sex:**
                              - | Group | n | Undertriage | Overtriage |
                              - |-------|---|-------------|------------|
                              - | Male | 40,339 | 0.037% | 0.084% |
                              - | Female | 37,735 | 0.061% | 0.119% |
                              - | Other/Unknown | 1,926 | 0.052% | 0.208% |
                             
                              - **By age group:**
                              - | Group | n | Undertriage | Overtriage |
                              - |-------|---|-------------|------------|
                              - | 18-39 | 27,889 | 0.047% | 0.104% |
                              - | 40-59 | 21,653 | 0.060% | 0.111% |
                              - | 60-79 | 23,863 | 0.042% | 0.101% |
                              - | 80+ | 6,595 | 0.045% | 0.091% |
                             
                              - **By insurance type:**
                              - | Group | n | Undertriage | Overtriage |
                              - |-------|---|-------------|------------|
                              - | Private | 48,170 | 0.050% | 0.112% |
                              - | Medicare | 3,196 | 0.031% | 0.063% |
                              - | Medicaid | 6,320 | 0.032% | 0.190% |
                              - | Self-pay | 19,915 | 0.055% | 0.065% |
                              - | Other | 2,399 | 0.042% | 0.083% |
                             
                              - **By language:**
                              - | Group | n | Undertriage | Overtriage |
                              - |-------|---|-------------|------------|
                              - | English | 44,134 | 0.050% | 0.120% |
                              - | Spanish | 5,587 | 0.072% | 0.036% |
                              - | Other European | 8,024 | 0.037% | 0.087% |
                              - | Arabic | 3,944 | 0.025% | 0.127% |
                              - | Asian languages | 4,858 | 0.041% | 0.103% |
                              - | French/Creole | 3,170 | 0.063% | 0.095% |
                              - | Portuguese | 6,315 | 0.063% | 0.111% |
                              - | Other | 3,968 | 0.025% | 0.025% |
                              - 
                              **Notable:** Spanish speakers show the highest undertriage rate (0.072%), consistent with literature on language-barrier undertriage; Medicaid group shows the highest overtriage rate (0.190%), possibly reflecting defensive triage patterns. A formal ethics committee review would be required before clinical deployment.

                                ## NLP Feature Contribution

                                Feature ablation (OOF QWK impact):
                                - Dropping 40 SVD components: -0.08 QWK (0.9991 to 0.9191)
                                - - Dropping 16-term keyword flag: -0.04 QWK (0.9991 to 0.9551)
                                  - - Total NLP block (SVD + keyword): ~0.12 QWK gap vs vitals-only baseline
                                   
                                    - ## Leakage audit
                                   
                                    - Three post-triage columns explicitly excluded before any modelling:
                                    - - `ed_los_hours` — ED length of stay (unknown at triage time)
                                      - - `disposition` — admission outcome (unknown at triage time)
                                        - - `triage_nurse_id` — encodes individual nurse bias patterns, non-generalizable across sites
                                         
                                          - Community discussion confirmed that retaining these columns yields artificially inflated accuracy (near 100%) while producing a clinically useless model.
                                         
                                          - ## Key insights
                                         
                                          - - **Informative missingness:** Missing vitals are not noise. systolic_bp, diastolic_bp, and respiratory_rate are more frequently missing in low-acuity patients. Missingness indicators fed directly as features.
                                            - - 4,979 unique chief complaint templates map near-deterministically to ESI level across 100,000 patients — a synthetic data artefact that inflates QWK vs real free-text.
                                              - - **Undocumented third column:** chief_complaints.csv exposes a chief_complaint_system column (14 body-system categories) not mentioned in the Dataset Description.
                                                - - **Stripped temporal resolution:** Arrival year and week intentionally withheld.
                                                  - - **news2_score ablation:** Dropping news2_score alone (r = -0.81 with triage_acuity) costs ~0.04 QWK; dropping all chief-complaint features costs ~0.12 QWK.
                                                   
                                                    - ## Limitations & Future Directions
                                                   
                                                    - - Synthetic data: all metrics are on synthetically generated encounters; real-world performance on prospective ED data remains unvalidated
                                                      - - MTS cross-validation required before deployment in Nordic/European EDs that use Manchester Triage System rather than ESI
                                                        - - GDPR/DPIA compliance and local ethics approval required for any EU pilot
                                                          - - Model assumes ESI-5 level acuity differentiation; may not generalise to 3-level triage systems (e.g., START mass-casualty)
                                                            - - Psychiatric boarding patients, pediatric patients (<18), and high-missingness cohorts (>50% vitals missing) are out of scope
                                                             
                                                              - ## Links
                                                             
                                                              - - **Kaggle Writeup:** https://www.kaggle.com/competitions/triagegeist/writeups/new-writeup-1776029721468
                                                                - - **Kaggle Notebook (V13, public):** https://www.kaggle.com/code/minalkharat123/triagegeist-esi-prediction-with-explainability-f
                                                                  - - **Competition page:** https://www.kaggle.com/competitions/triagegeist
                                                                   
                                                                    - ## License
                                                                   
                                                                    - MIT
