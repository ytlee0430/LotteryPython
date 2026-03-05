# Story-10: Personalized Ensemble Weights

## Status
review

## Story
**As a** regular LotteryPython user,
**I want** to adjust which prediction algorithms I trust more,
**So that** my personalized recommendations reflect my own strategy.

## Acceptance Criteria
- [x] AC1: User personal weights stored in SQLite `users.preferences` (JSON column)
- [x] AC2: Settings page with per-algorithm weight sliders (0 = off, 1 = default, 2 = double)
- [x] AC3: Prediction uses personal weights when set; falls back to global config otherwise
- [x] AC4: "重設為預設" (Reset to default) button available
- [x] AC5: Settings persist across logins (stored in DB, not localStorage)
- [x] AC6: Weight change takes effect immediately on next prediction (no refresh required)

## Tasks / Subtasks
- [x] Task 1: DB migration — add `preferences TEXT` column to `users` table
  - [x] 1.1: Added migration logic to `UserManager._init_database()` using `ALTER TABLE` with try/except for idempotency
- [x] Task 2: Update `UserManager` with preferences methods
  - [x] 2.1: `get_preferences(user_id) -> dict` — returns personal weights or empty dict
  - [x] 2.2: `set_preferences(user_id, prefs: dict)` — saves JSON to preferences column
- [x] Task 3: Add API endpoints in `app.py`
  - [x] 3.1: `GET /api/preferences` — returns current user's preferences
  - [x] 3.2: `PUT /api/preferences` — updates current user's preferences
- [x] Task 4: Update ensemble to accept user weights override
  - [x] 4.1: `collect_predictions(df, index, user_weights=None)` — merges user_weights over global weights
  - [x] 4.2: Updated `logic.py` to load user preferences and pass to `predict_ensemble()`
- [x] Task 5: Create `templates/settings.html` settings page
  - [x] 5.1: Per-algorithm weight sliders (range 0–2, step 0.1)
  - [x] 5.2: Reset to defaults button
  - [x] 5.3: Save button with AJAX PUT to `/api/preferences`
- [x] Task 6: Add route `GET /settings` in `app.py`
- [x] Task 7: Write tests
  - [x] 7.1: Test personal weights override global config in ensemble
  - [x] 7.2: Test preferences persist and load correctly

## Dev Notes
- Migration pattern: `ALTER TABLE users ADD COLUMN preferences TEXT` — wrapped in try/except for idempotency
- Preferences JSON format: `{"Hot-50": 1.5, "LSTM": 0.0, ...}` — stores all slider values
- Merge strategy: `{**global_weights, **(user_weights or {})}` so missing keys fall back to global
- Algorithm names match exactly with keys in `DEFAULT_CONFIG["ensemble_weights"]`

## Dev Agent Record
### Implementation Plan
- `UserManager._init_database()`: idempotent `ALTER TABLE users ADD COLUMN preferences TEXT`
- `get_preferences/set_preferences` methods using JSON serialization
- `GET /api/preferences` / `PUT /api/preferences` in app.py (login_required)
- `collect_predictions(user_weights=None)` + `predict_ensemble(user_weights=None)` parameter chain
- `logic.py` loads user weights from `UserManager.get_preferences(user_id)` before calling ensemble
- `templates/settings.html`: sliders 0–2 step 0.1, default/custom tag labels, save/reset/back buttons
- `GET /settings` route passing `algo_names` from `DEFAULT_CONFIG`

### Debug Log
- No major issues during implementation

### Completion Notes
- Created `tests/test_personalized_weights.py` with 6 tests (all passing)
- UserManager get/set preferences round-trip verified
- Zero-weight algorithm still appears in predictions but has no voting impact
- All 35 non-slow tests pass including full suite regression

## File List
- predict/astrology/profiles.py (modified — preferences column migration, get/set_preferences)
- predict/lotto_predict_ensemble.py (modified — user_weights param in collect_predictions + predict_ensemble)
- lotterypython/logic.py (modified — user weights loading and ensemble passthrough)
- app.py (modified — /settings route, GET/PUT /api/preferences endpoints)
- templates/settings.html (new)
- tests/test_personalized_weights.py (new)

## Change Log
- 2026-03-05: Implemented all tasks. 6/6 personalized weight tests pass. Full 35-test suite green.
