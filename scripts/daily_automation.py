#!/usr/bin/env python3
"""
Daily Automation Script for LotteryPython

Executes the complete daily workflow:
1. Update lottery data
2. Clear outdated cache
3. Run backtests (cached)
4. Run parameter optimization
5. Auto-tune ensemble weights
6. Run predictions
7. Save results

Usage:
    python daily_automation.py [options]

Options:
    --type TYPE         Lottery type: 'big', 'super', or 'auto' (default: auto)
    --force             Force run even if not a draw day
    --skip-update       Skip data update step
    --skip-backtest     Skip backtest step
    --skip-predict      Skip prediction step
    --skip-autotune     Skip auto-tune step
    --dry-run           Test mode, don't write results
    -v, --verbose       Verbose output
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present (fallback when not invoked via shell script)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        # dotenv not installed; parse manually for simple KEY=VALUE lines
        with open(_env_file) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())

# Import project modules
from lotterypython.update_data import main as update_lottery_data
from lotterypython.logic import run_predictions
from lotterypython.analysis_sheet import append_analysis_results
from predict.backtest import (
    run_full_backtest, rolling_backtest, optimize_window_size,
    clear_outdated_backtest_cache, get_backtest_cache_stats, run_autotune
)
from predict.config import get_config


# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_date = datetime.now().strftime("%Y%m%d")
    log_file = LOG_DIR / f"daily_{log_date}.log"
    error_file = LOG_DIR / f"errors_{log_date}.log"

    # Create logger
    logger = logging.getLogger("daily_automation")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler (all logs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)-5s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(file_handler)

    # Error file handler
    error_handler = logging.FileHandler(error_file, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)-5s %(message)s\n%(exc_info)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(error_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)-5s %(message)s',
        datefmt='%H:%M:%S'
    ))
    logger.addHandler(console_handler)

    return logger


def get_today_lottery_type() -> str:
    """
    Determine which lottery type should be processed today.

    On draw days, returns that day's lottery type.
    On non-draw days, returns the next upcoming draw's lottery type.

    Returns:
        'big' for Tuesday/Friday (大樂透)
        'super' for Monday/Thursday (威力彩)
        For non-draw days:
          Wednesday → 'super' (next draw: Thursday)
          Saturday  → 'super' (next draw: Monday)
          Sunday    → 'super' (next draw: Monday)
    """
    weekday = datetime.now().weekday()  # 0=Monday, 6=Sunday

    if weekday in [1, 4]:  # Tuesday, Friday
        return 'big'
    elif weekday in [0, 3]:  # Monday, Thursday
        return 'super'
    else:
        # Non-draw days: next draw is always 威力彩
        # Wed→Thu(super), Sat→Mon(super), Sun→Mon(super)
        return 'super'


def is_draw_day() -> bool:
    """Check if today is a lottery draw day (Mon/Tue/Thu/Fri)."""
    return datetime.now().weekday() in [0, 1, 3, 4]


def get_lottery_name(lottery_type: str) -> str:
    """Get Chinese name for lottery type."""
    return "大樂透" if lottery_type == 'big' else "威力彩"


class DailyAutomation:
    """Daily automation workflow executor."""

    def __init__(self, lottery_type: str, dry_run: bool = False, verbose: bool = False):
        self.lottery_type = lottery_type
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = setup_logging(verbose)
        self.start_time = datetime.now()
        self.results = {
            "status": "pending",
            "lottery_type": lottery_type,
            "timestamp": self.start_time.isoformat(),
            "steps": {}
        }

        # Configuration
        self.backtest_periods = int(os.environ.get('BACKTEST_PERIODS', 50))
        self.rolling_window = int(os.environ.get('ROLLING_WINDOW', 20))
        self.rolling_total = int(os.environ.get('ROLLING_TOTAL', 100))

    def log_step(self, step: int, total: int, message: str):
        """Log a step progress message."""
        self.logger.info(f"Step {step}/{total}: {message}")

    def run_step(self, step_name: str, func, *args, **kwargs) -> Tuple[bool, any]:
        """Run a step with error handling."""
        try:
            result = func(*args, **kwargs)
            self.results["steps"][step_name] = {"status": "success", "result": result}
            return True, result
        except Exception as e:
            self.logger.error(f"Step '{step_name}' failed: {e}", exc_info=True)
            self.results["steps"][step_name] = {"status": "error", "error": str(e)}
            return False, None

    def step_update_data(self) -> bool:
        """Step 1: Update lottery data."""
        self.log_step(1, 8, "Updating lottery data...")

        if self.dry_run:
            self.logger.info("  [DRY RUN] Would update lottery data")
            self.results["steps"]["update"] = {"status": "skipped", "reason": "dry_run"}
            return True

        # Always update BOTH lottery types so neither falls behind
        all_types = ['big', 'super']
        total_new = 0
        update_summary = {}

        for ltype in all_types:
            try:
                result = update_lottery_data(ltype)
                new_records = result.get('new_records', 0) if isinstance(result, dict) else 0
                new_draws = result.get('draws', []) if isinstance(result, dict) else []
                update_summary[ltype] = {"status": "success", "new_records": new_records}
                total_new += new_records

                if new_records > 0:
                    self.logger.info(f"  {get_lottery_name(ltype)}: {new_records} new record(s)")
                    try:
                        from notify.line_notifier import notify_data_update
                        notify_data_update(
                            lottery_type=ltype,
                            lottery_name=get_lottery_name(ltype),
                            draws=new_draws,
                            dry_run=self.dry_run,
                        )
                        self.logger.info(f"  LINE update notification sent for {get_lottery_name(ltype)}")
                    except Exception as ne:
                        self.logger.warning(f"  LINE update notify failed for {ltype}: {ne}")
                else:
                    self.logger.info(f"  {get_lottery_name(ltype)}: no new records")
            except Exception as e:
                self.logger.warning(f"  {get_lottery_name(ltype)} update failed: {e}")
                update_summary[ltype] = {"status": "error", "error": str(e)}

        self.log_step(1, 8, f"Data update done ({total_new} total new records)")
        self.results["steps"]["update"] = {"status": "success", "total_new": total_new, **update_summary}
        return True

    def step_clear_cache(self) -> bool:
        """Step 2: Clear outdated cache."""
        self.log_step(2, 8, "Clearing outdated cache...")

        try:
            cleared = clear_outdated_backtest_cache(self.lottery_type)
            total_cleared = cleared.get('total', 0)

            self.log_step(2, 8, f"Cleared {total_cleared} outdated entries")
            self.results["steps"]["clear_cache"] = {"status": "success", "cleared": total_cleared}
            return True
        except Exception as e:
            self.logger.warning(f"Step 2/8: Cache clear failed: {e}")
            self.results["steps"]["clear_cache"] = {"status": "error", "error": str(e)}
            return True

    def step_full_backtest(self) -> bool:
        """Step 3: Run full backtest."""
        self.log_step(3, 8, "Running full backtest...")

        import time
        start = time.time()

        try:
            result = run_full_backtest(self.lottery_type, self.backtest_periods, use_cache=True)
            elapsed = time.time() - start
            from_cache = result.get('from_cache', False)

            self.log_step(3, 8, f"Backtest completed ({elapsed:.1f}s, from_cache={from_cache})")
            self.results["steps"]["backtest"] = {
                "status": "success",
                "cached": from_cache,
                "duration_seconds": round(elapsed, 1)
            }
            return True
        except Exception as e:
            self.logger.error(f"Step 3/8: Full backtest failed: {e}", exc_info=True)
            self.results["steps"]["backtest"] = {"status": "error", "error": str(e)}
            return False

    def step_rolling_backtest(self) -> bool:
        """Step 4: Run rolling backtest."""
        self.log_step(4, 8, "Running rolling backtest...")

        import time
        start = time.time()

        try:
            result = rolling_backtest(
                self.lottery_type,
                self.rolling_window,
                self.rolling_total,
                use_cache=True
            )
            elapsed = time.time() - start
            from_cache = result.get('from_cache', False)

            self.log_step(4, 8, f"Rolling backtest completed ({elapsed:.1f}s, from_cache={from_cache})")
            self.results["steps"]["rolling"] = {
                "status": "success",
                "cached": from_cache,
                "duration_seconds": round(elapsed, 1)
            }
            return True
        except Exception as e:
            self.logger.error(f"Step 4/8: Rolling backtest failed: {e}", exc_info=True)
            self.results["steps"]["rolling"] = {"status": "error", "error": str(e)}
            return False

    def step_optimize(self) -> bool:
        """Step 5: Run parameter optimization."""
        self.log_step(5, 8, "Running parameter optimization...")

        import time
        start = time.time()

        try:
            result = optimize_window_size(
                self.lottery_type,
                min_window=20,
                max_window=100,
                step=10,
                use_cache=True
            )
            elapsed = time.time() - start
            from_cache = result.get('from_cache', False)
            optimal = result.get('optimal', {})

            self.log_step(5, 8, f"Optimization completed ({elapsed:.1f}s)")
            self.logger.info(f"  Optimal Hot window: {optimal.get('hot_window', 'N/A')}")
            self.logger.info(f"  Optimal Cold window: {optimal.get('cold_window', 'N/A')}")

            self.results["steps"]["optimize"] = {
                "status": "success",
                "cached": from_cache,
                "optimal": optimal,
                "duration_seconds": round(elapsed, 1)
            }
            return True
        except Exception as e:
            self.logger.error(f"Step 5/8: Optimization failed: {e}", exc_info=True)
            self.results["steps"]["optimize"] = {"status": "error", "error": str(e)}
            return False

    def step_autotune(self) -> bool:
        """Step 6: Auto-tune ensemble weights using softmax scoring."""
        self.log_step(6, 8, "Auto-tuning ensemble weights...")

        if self.dry_run:
            self.logger.info("  [DRY RUN] Would update ensemble weights")
            self.results["steps"]["autotune"] = {"status": "skipped", "reason": "dry_run"}
            return True

        config = get_config()
        if not config.get("auto_tune_enabled", False):
            self.logger.info("  Auto-tune disabled (auto_tune_enabled=false)")
            self.results["steps"]["autotune"] = {"status": "skipped", "reason": "disabled"}
            return True

        try:
            result = run_autotune(self.lottery_type, self.backtest_periods)

            if result.get("skipped"):
                self.logger.info(f"  Auto-tune skipped: {result.get('reason')}")
                self.results["steps"]["autotune"] = {
                    "status": "skipped",
                    "reason": result.get("reason")
                }
                return True

            new_weights = result.get("updated_weights", {})
            self.log_step(6, 8, "Weights updated")
            if self.verbose:
                for algo, weight in new_weights.items():
                    self.logger.debug(f"  {algo}: {weight}")

            self.results["steps"]["autotune"] = {
                "status": "success",
                "weights_updated": True,
                "new_weights": new_weights
            }
            return True
        except Exception as e:
            self.logger.warning(f"Step 6/8: Auto-tune failed: {e}")
            self.results["steps"]["autotune"] = {"status": "error", "error": str(e)}
            return True

    def step_predict(self) -> tuple:
        """Step 7: Run predictions for both lottery types and save results.

        Returns:
            Tuple[bool, dict, dict]: (success, predictions_big, predictions_super)
        """
        self.log_step(7, 8, "Running predictions (大樂透 + 威力彩)...")

        if self.dry_run:
            self.logger.info("  [DRY RUN] Would run predictions and save to Google Sheets")
            self.results["steps"]["predict"] = {"status": "skipped", "reason": "dry_run"}
            return True, {}, {}

        from predict.backtest import load_historical_data

        predictions_big = {}
        predictions_super = {}
        saved_to_sheets = False

        for ltype in ['big', 'super']:
            lname = get_lottery_name(ltype)
            try:
                df = load_historical_data(ltype)
                if df.empty:
                    self.logger.warning(f"  {lname}: no historical data, skipping")
                    continue

                preds = run_predictions(df)
                self.logger.info(f"  {lname}: {len(preds)} algorithm(s)")

                if ltype == 'big':
                    predictions_big = preds
                    self._last_df_big = df
                else:
                    predictions_super = preds
                    self._last_df_super = df

                # Save primary lottery type to Google Sheets
                if ltype == self.lottery_type:
                    try:
                        prediction_tuples = [
                            (algo, r.get('next_period', ''), r.get('numbers', []), r.get('special', 0))
                            for algo, r in preds.items()
                            if isinstance(r, dict) and 'error' not in r
                        ]
                        if prediction_tuples:
                            append_analysis_results(prediction_tuples, ltype)
                            self.logger.info(f"  {lname}: results saved to Google Sheets")
                            saved_to_sheets = True
                    except Exception as e:
                        self.logger.warning(f"  {lname}: failed to save to Google Sheets: {e}")

            except Exception as e:
                self.logger.error(f"  {lname}: prediction failed: {e}", exc_info=True)

        success = bool(predictions_big or predictions_super)
        total_algos = len(predictions_big) + len(predictions_super)
        self.log_step(7, 8, f"Predictions completed ({total_algos} total across both types)")
        self.results["steps"]["predict"] = {
            "status": "success" if success else "error",
            "algorithms_big": len(predictions_big),
            "algorithms_super": len(predictions_super),
            "saved_to_sheets": saved_to_sheets,
        }
        return success, predictions_big, predictions_super

    def step_line_notify(self, predictions_big: dict, predictions_super: dict, skip: bool = False) -> bool:
        """Step 8: Push combined LINE notifications (both lottery types) to enabled users."""
        self.log_step(8, 8, "Sending LINE notifications (大樂透 + 威力彩)...")

        if skip:
            self.logger.info("  Step 8/8: Skipped (--skip-notify)")
            self.results["steps"]["line_notify"] = {"status": "skipped", "reason": "flag"}
            return True

        if self.dry_run:
            self.logger.info("  [DRY RUN] Would send LINE notifications")
            self.results["steps"]["line_notify"] = {"status": "skipped", "reason": "dry_run"}
            return True

        if not os.environ.get("LINE_CHANNEL_ACCESS_TOKEN"):
            self.logger.warning("  LINE_CHANNEL_ACCESS_TOKEN not set — skipping notifications")
            self.results["steps"]["line_notify"] = {"status": "skipped", "reason": "no_token"}
            return True

        try:
            from notify.line_notifier import notify_all_users_both

            summary = notify_all_users_both(
                predictions_big=predictions_big,
                predictions_super=predictions_super,
                df_big=getattr(self, '_last_df_big', None),
                df_super=getattr(self, '_last_df_super', None),
                dry_run=self.dry_run,
            )
            self.log_step(8, 8, f"Notifications: {summary['notified']} sent, {summary['failed']} failed, {summary['skipped']} skipped")
            self.results["steps"]["line_notify"] = {"status": "success", **summary}
            return True
        except Exception as e:
            self.logger.error(f"Step 8/8: LINE notify failed: {e}", exc_info=True)
            self.results["steps"]["line_notify"] = {"status": "error", "error": str(e)}
            return True  # Notification failure does not fail automation

    def run(self, skip_update=False, skip_backtest=False, skip_predict=False, skip_autotune=False, skip_notify=False) -> Dict:
        """Execute the complete daily automation workflow."""
        lottery_name = get_lottery_name(self.lottery_type)

        self.logger.info("=" * 50)
        self.logger.info("=== Daily Automation Started ===")
        self.logger.info(f"Lottery type: {lottery_name} ({self.lottery_type})")
        self.logger.info(f"Dry run: {self.dry_run}")
        self.logger.info("=" * 50)

        success = True

        # Step 1: Update data
        if not skip_update:
            self.step_update_data()
        else:
            self.logger.info("Step 1/8: Skipped (--skip-update)")
            self.results["steps"]["update"] = {"status": "skipped", "reason": "flag"}

        # Step 2: Clear outdated cache
        self.step_clear_cache()

        # Steps 3-5: Backtests
        if not skip_backtest:
            if not self.step_full_backtest():
                success = False
            if not self.step_rolling_backtest():
                success = False
            if not self.step_optimize():
                success = False
        else:
            self.logger.info("Steps 3-5/8: Skipped (--skip-backtest)")
            for step in ["backtest", "rolling", "optimize"]:
                self.results["steps"][step] = {"status": "skipped", "reason": "flag"}

        # Step 6: Auto-tune
        if not skip_autotune:
            self.step_autotune()
        else:
            self.logger.info("Step 6/8: Skipped (--skip-autotune)")
            self.results["steps"]["autotune"] = {"status": "skipped", "reason": "flag"}

        # Step 7: Predict
        predictions_big = {}
        predictions_super = {}
        if not skip_predict:
            predict_ok, predictions_big, predictions_super = self.step_predict()
            if not predict_ok:
                success = False
        else:
            self.logger.info("Step 7/8: Skipped (--skip-predict)")
            self.results["steps"]["predict"] = {"status": "skipped", "reason": "flag"}

        # Step 8: LINE Notifications
        self.step_line_notify(predictions_big, predictions_super, skip=skip_notify)

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # Get cache stats
        cache_stats = get_backtest_cache_stats()

        # Finalize results
        self.results["status"] = "success" if success else "partial"
        self.results["duration_seconds"] = round(duration, 1)
        self.results["cache_stats"] = {
            "total_entries": cache_stats.get("total_entries", 0),
            "total_size_kb": cache_stats.get("total_size_kb", 0)
        }

        # Log completion
        self.logger.info("=" * 50)
        self.logger.info("=== Daily Automation Completed ===")
        self.logger.info(f"Status: {self.results['status']}")
        self.logger.info(f"Total time: {int(duration // 60)}m {int(duration % 60)}s")
        self.logger.info(f"Cache entries: {cache_stats.get('total_entries', 0)}")
        self.logger.info("=" * 50)

        # Save results to JSON
        results_file = LOG_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Results saved to: {results_file}")

        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Daily automation script for LotteryPython",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--type', '-t',
        choices=['big', 'super', 'auto'],
        default='auto',
        help="Lottery type (default: auto - based on today's draw schedule)"
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Force run even if not a draw day"
    )
    parser.add_argument(
        '--skip-update',
        action='store_true',
        help="Skip data update step"
    )
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help="Skip backtest steps"
    )
    parser.add_argument(
        '--skip-predict',
        action='store_true',
        help="Skip prediction step"
    )
    parser.add_argument(
        '--skip-autotune',
        action='store_true',
        help="Skip auto-tune step"
    )
    parser.add_argument(
        '--skip-notify',
        action='store_true',
        help="Skip LINE notification step"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Test mode, don't write results"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Verbose output"
    )

    args = parser.parse_args()

    # Determine lottery type
    if args.type == 'auto':
        lottery_type = get_today_lottery_type()
        draw_day = is_draw_day()
    else:
        lottery_type = args.type
        draw_day = True  # explicit type always runs full pipeline

    # On non-draw days: auto-skip heavy steps (backtest + autotune)
    skip_backtest = args.skip_backtest
    skip_autotune = args.skip_autotune
    if not draw_day and not args.force:
        skip_backtest = True
        skip_autotune = True
        day_name = datetime.now().strftime('%A')
        lottery_name = "大樂透" if lottery_type == "big" else "威力彩"
        print(f"Non-draw day ({day_name}): lightweight run for next draw ({lottery_name})")

    # Run automation
    automation = DailyAutomation(
        lottery_type=lottery_type,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    results = automation.run(
        skip_update=args.skip_update,
        skip_backtest=skip_backtest,
        skip_predict=args.skip_predict,
        skip_autotune=skip_autotune,
        skip_notify=args.skip_notify,
    )

    # Exit with appropriate code
    if results["status"] == "success":
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
