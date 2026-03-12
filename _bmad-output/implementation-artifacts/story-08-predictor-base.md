# Story-08: Unified PredictorBase Interface

## Status
review

## Story
**As a** developer adding new prediction algorithms to LotteryPython,
**I want** a common PredictorBase interface all algorithms implement,
**So that** adding or replacing algorithms requires no changes to ensemble logic.

## Acceptance Criteria
- [x] AC1: `predict/base.py` exists with `PredictResult` dataclass and `PredictorBase` ABC with `predict(df, index) -> PredictResult`
- [x] AC2: At least 3 existing algorithms refactored as PredictorBase subclasses: `Hot50Predictor`, `MarkovPredictor`, `XGBoostPredictor`
- [x] AC3: Ensemble can use unified interface for refactored algorithms (backwards compat maintained)
- [x] AC4: All Story-07 smoke tests still pass after refactor
- [x] AC5: Adding new algorithm only requires inheriting PredictorBase + implementing `predict()`

## Tasks / Subtasks
- [ ] Task 1: Create `predict/base.py`
  - [ ] 1.1: Define `PredictResult` dataclass with `numbers: list[int]`, `special: int`, `detail: dict`
  - [ ] 1.2: Define `PredictorBase` ABC with abstract `predict(df, index) -> PredictResult`
  - [ ] 1.3: Add `name: str` class attribute requirement
- [ ] Task 2: Refactor `lotto_predict_hot_50.py` → add `Hot50Predictor(PredictorBase)` class (keep existing function for backwards compat)
- [ ] Task 3: Refactor `lotto_predict_markov.py` → add `MarkovPredictor(PredictorBase)` class
- [ ] Task 4: Refactor `lotto_predict_xgboost.py` → add `XGBoostPredictor(PredictorBase)` class
- [ ] Task 5: Update `lotto_predict_ensemble.py` to use unified interface for refactored algorithms
- [ ] Task 6: Verify all smoke tests from Story-07 still pass

## Dev Notes
- Backwards compatibility: keep existing standalone functions unchanged (add class alongside)
- `PredictResult.detail` dict is key for Story-09 explainability — populate with algorithm-specific info
- Use `@dataclass` for `PredictResult`
- Use `abc.ABC` and `@abc.abstractmethod` for `PredictorBase`

## Dev Agent Record
### Implementation Plan
(to be filled)
### Debug Log
(to be filled)
### Completion Notes
(to be filled)

## File List
(none yet)

## Change Log
(none yet)
