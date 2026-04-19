# Triagegeist: ESI Triage Prediction with Explainability, Fairness & Uncertainty Quantification

Submission for the [Kaggle Triagegeist Hackathon](https://www.kaggle.com/competitions/triagegeist) — a clinical ML challenge on predicting Emergency Severity Index (ESI) levels from structured intake data and free-text chief complaints, with emphasis on clinical relevance, fairness, and deployability.

## TL;DR

A LightGBM + CatBoost ensemble (60/40) trained on 80,000 synthetic ED encounters achieves:

- **OOF Quadratic Weighted Kappa: 0.9991**
- - **OOF Accuracy: 0.9985**
  - - **OOF Macro F1: 0.9963**
    - - **Undertriage: 0.049%** (the clinically dangerous direction)
      - - **Overtriage: 0.097%**
        - - **Critical ESI-1 misses: 53 of 80,000**
         
          - Paired with SHAP per-class explanations, a demographic equity audit across age, sex, insurance type, and language, and a calibrated uncertainty-flagging layer that routes low-confidence predictions to a human-in-the-loop.
         
          - ## Why this matters clinically
         
          - Every minute in the ED counts. Triage nurses and physicians must rapidly assign each arriving patient an ESI level (1 = immediate, life-threatening to 5 = non-urgent) under extreme cognitive load, incomplete information, and chronic understaffing. The ESI system, while widely adopted, relies entirely on unaided human judgment. Inter-rater variability is well-documented in the literature, and systematic undertriage of elderly patients, non-English speakers, and the uninsured is an active patient-safety concern.
         
          - This project addresses a focused, high-stakes question: can an ML system trained on structured intake data plus free-text chief complaints reliably predict ESI acuity, explain its reasoning clinically, flag cases of genuine uncertainty, and surface potential equity gaps?
         
          - ## Architecture
         
          - - Data ingestion: train + chief_complaints + patient_history, joined on patient_id
            - - Leakage removal: ed_los_hours, disposition, triage_nurse_id dropped before training
              - - Feature engineering: clinical composites (shock_index, MAP, NEWS2, BMI), informative-missingness indicators, vital-age interactions, TF-IDF + TruncatedSVD on chief complaints, operational categoricals
                - - Ensemble: LightGBM (60%) + CatBoost (40%), 5-fold stratified CV, OOF predictions
                  - - Explainability: SHAP per-class, equity audit by age/sex/insurance/language, calibrated uncertainty flagging for human-in-the-loop
                   
                    - ## Reported metrics
                   
                    - | Metric | Value | Why it matters |
                    - |---|---|---|
                    - | QWK | 0.9991 | Ordinal agreement with ground truth |
                    - | Macro-F1 | 0.9963 | Balanced per-class performance under imbalance |
                    - | Accuracy | 0.9985 | Overall correctness |
                    - | Undertriage rate | 0.049% | Clinically most dangerous error mode |
                    - | Overtriage rate | 0.097% | Resource-waste error mode |
                    - | ESI-1 misses | 53 / 80,000 | Absolute count of missed critical patients |
                   
                    - ## Leakage audit
                   
                    - Three post-triage columns explicitly excluded before any modelling:
                    - - `ed_los_hours` - ED length of stay (unknown at triage time)
                      - - `disposition` - admission outcome (unknown at triage time)
                        - - `triage_nurse_id` - encodes individual nurse bias patterns, non-generalizable across sites
                         
                          - ## Key insights
                         
                          - - **Informative missingness:** Missing vitals are not noise. systolic_bp, diastolic_bp, and respiratory_rate are more frequently missing in low-acuity patients. Missingness indicators fed directly as features.
                            - - **4,979 unique chief complaint templates** map near-deterministically to ESI level across 100,000 patients - a synthetic data artefact that inflates QWK vs real free-text.
                              - - **Undocumented third column:** chief_complaints.csv exposes a chief_complaint_system column (14 body-system categories) not mentioned in the Dataset Description.
                                - - **Stripped temporal resolution:** Arrival year and week intentionally withheld.
                                  - - **news2_score ablation:** Dropping news2_score alone (r = -0.81 with triage_acuity) costs ~0.04 QWK; dropping all chief-complaint features costs ~0.12 QWK.
                                   
                                    - ## Links
                                   
                                    - - **Kaggle Writeup:** https://www.kaggle.com/competitions/triagegeist/writeups/new-writeup-1776029721468
                                      - - **Kaggle Notebook (V8, public):** https://www.kaggle.com/code/minalkharat123/triagegeist-esi-prediction-with-explainability-f
                                        - - **Competition page:** https://www.kaggle.com/competitions/triagegeist
                                         
                                          - ## License
                                         
                                          - MIT
