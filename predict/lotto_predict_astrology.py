"""
Astrology-based Lottery Prediction Module

Combines Chinese (紫微斗數) and Western (星座) astrology predictions
for lottery number recommendations using Gemini AI.
"""

from collections import Counter
from typing import Optional, Tuple, List, Dict
import re

from predict.astrology.profiles import BirthProfileManager, PredictionCacheManager
from predict.astrology.gemini_client import GeminiAstrologyClient


# Singleton instances
_profile_manager: Optional[BirthProfileManager] = None
_gemini_client: Optional[GeminiAstrologyClient] = None
_cache_manager: Optional[PredictionCacheManager] = None


def get_profile_manager() -> BirthProfileManager:
    """Get or create the profile manager singleton."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = BirthProfileManager()
    return _profile_manager


def get_gemini_client() -> GeminiAstrologyClient:
    """Get or create the Gemini client singleton."""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiAstrologyClient()
    return _gemini_client


def get_cache_manager() -> PredictionCacheManager:
    """Get or create the cache manager singleton."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = PredictionCacheManager()
    return _cache_manager


def get_next_period(lottery_type: str = 'big') -> int:
    """
    Get the next lottery period number from historical data.

    Args:
        lottery_type: 'big' for 大樂透 or 'super' for 威力彩

    Returns:
        int: Next period number
    """
    import pandas as pd
    from pathlib import Path

    csv_file = Path(__file__).parent.parent / "lotterypython" / f"{lottery_type}_sequence.csv"
    if not csv_file.exists():
        # Try alternate path
        csv_file = Path(__file__).parent.parent.parent / "lotterypython" / f"{lottery_type}_sequence.csv"

    if not csv_file.exists():
        return 1  # Default if file not found

    try:
        df = pd.read_csv(csv_file)
        if 'Period' not in df.columns or df.empty:
            return 1

        periods = df['Period'].astype(str).tolist()
        numeric_periods = []
        for p in periods:
            digits_only = re.sub(r'\D', '', p)
            if digits_only:
                numeric_periods.append(int(digits_only))

        if numeric_periods:
            return max(numeric_periods) + 1
        return 1
    except Exception:
        return 1


def predict_ziwei(lottery_type: str = 'big', profile_name: Optional[str] = None,
                  period: Optional[int] = None, user_id: int = None) -> Tuple[List[int], int, Dict]:
    """
    Predict lottery numbers using 紫微斗數 (Purple Star Astrology).

    Args:
        lottery_type: 'big' for 大樂透 or 'super' for 威力彩
        profile_name: Specific profile to use (uses all if None)
        period: Lottery period number (auto-detected if None)
        user_id: Owner user's ID for filtering profiles

    Returns:
        tuple: (numbers, special, details)
    """
    manager = get_profile_manager()
    client = get_gemini_client()
    cache = get_cache_manager()

    # Get profiles
    if profile_name:
        profile = manager.get_profile(profile_name, user_id)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        profiles = [profile]
    else:
        profiles = manager.get_all_profiles(user_id)

    if not profiles:
        raise ValueError("No birth profiles found. Please add at least one profile.")

    # Get period (auto-detect if not provided)
    if period is None:
        period = get_next_period(lottery_type)

    # Get profile IDs for cache key
    profile_ids = [p['id'] for p in profiles]

    # Check cache first
    cached = cache.get_cached_prediction(lottery_type, period, 'ziwei', profile_ids, user_id)
    if cached:
        cached['details']['from_cache'] = True
        cached['details']['period'] = period
        return cached['numbers'], cached['special'], cached['details']

    # Cache miss - calculate prediction
    all_numbers = []
    all_specials = []
    details = {"predictions": [], "method": "紫微斗數", "period": period, "from_cache": False}

    for profile in profiles:
        try:
            result = client.predict_ziwei(profile, lottery_type)
            all_numbers.extend(result['numbers'])
            all_specials.append(result['special'])
            details["predictions"].append({
                "name": profile['name'],
                "numbers": result['numbers'],
                "special": result['special'],
                "analysis": result.get('analysis', '')
            })
        except Exception as e:
            details["predictions"].append({
                "name": profile['name'],
                "error": str(e)
            })

    if not all_numbers:
        raise ValueError("All Ziwei predictions failed")

    # Aggregate by frequency voting
    number_counts = Counter(all_numbers)
    top_6 = [n for n, _ in number_counts.most_common(6)]

    special_counts = Counter(all_specials)
    special = special_counts.most_common(1)[0][0]

    # Ensure we have 6 numbers
    max_num = 49 if lottery_type == 'big' else 38
    while len(top_6) < 6:
        for n in range(1, max_num + 1):
            if n not in top_6:
                top_6.append(n)
                break

    final_numbers = sorted(top_6[:6])
    details["final_numbers"] = final_numbers
    details["final_special"] = special

    # Save to cache
    cache.save_prediction(lottery_type, period, 'ziwei', profile_ids,
                         final_numbers, special, details, user_id)

    return final_numbers, special, details


def predict_zodiac(lottery_type: str = 'big', profile_name: Optional[str] = None,
                   period: Optional[int] = None, user_id: int = None) -> Tuple[List[int], int, Dict]:
    """
    Predict lottery numbers using Western Zodiac astrology.

    Args:
        lottery_type: 'big' for 大樂透 or 'super' for 威力彩
        profile_name: Specific profile to use (uses all if None)
        period: Lottery period number (auto-detected if None)
        user_id: Owner user's ID for filtering profiles

    Returns:
        tuple: (numbers, special, details)
    """
    manager = get_profile_manager()
    client = get_gemini_client()
    cache = get_cache_manager()

    # Get profiles
    if profile_name:
        profile = manager.get_profile(profile_name, user_id)
        if not profile:
            raise ValueError(f"Profile '{profile_name}' not found")
        profiles = [profile]
    else:
        profiles = manager.get_all_profiles(user_id)

    if not profiles:
        raise ValueError("No birth profiles found. Please add at least one profile.")

    # Get period (auto-detect if not provided)
    if period is None:
        period = get_next_period(lottery_type)

    # Get profile IDs for cache key
    profile_ids = [p['id'] for p in profiles]

    # Check cache first
    cached = cache.get_cached_prediction(lottery_type, period, 'zodiac', profile_ids, user_id)
    if cached:
        cached['details']['from_cache'] = True
        cached['details']['period'] = period
        return cached['numbers'], cached['special'], cached['details']

    # Cache miss - calculate prediction
    all_numbers = []
    all_specials = []
    details = {"predictions": [], "method": "西洋星座", "period": period, "from_cache": False}

    for profile in profiles:
        try:
            result = client.predict_zodiac(profile, lottery_type)
            all_numbers.extend(result['numbers'])
            all_specials.append(result['special'])
            details["predictions"].append({
                "name": profile['name'],
                "zodiac": result['zodiac'],
                "numbers": result['numbers'],
                "special": result['special'],
                "lucky_element": result.get('lucky_element', '')
            })
        except Exception as e:
            details["predictions"].append({
                "name": profile['name'],
                "error": str(e)
            })

    if not all_numbers:
        raise ValueError("All Zodiac predictions failed")

    # Aggregate by frequency voting
    number_counts = Counter(all_numbers)
    top_6 = [n for n, _ in number_counts.most_common(6)]

    special_counts = Counter(all_specials)
    special = special_counts.most_common(1)[0][0]

    # Ensure we have 6 numbers
    max_num = 49 if lottery_type == 'big' else 38
    while len(top_6) < 6:
        for n in range(1, max_num + 1):
            if n not in top_6:
                top_6.append(n)
                break

    final_numbers = sorted(top_6[:6])
    details["final_numbers"] = final_numbers
    details["final_special"] = special

    # Save to cache
    cache.save_prediction(lottery_type, period, 'zodiac', profile_ids,
                         final_numbers, special, details, user_id)

    return final_numbers, special, details


def predict_astrology_combined(lottery_type: str = 'big') -> Tuple[List[int], int, Dict]:
    """
    Combine both Ziwei and Zodiac predictions.

    Args:
        lottery_type: 'big' for 大樂透 or 'super' for 威力彩

    Returns:
        tuple: (numbers, special, details)
    """
    details = {"ziwei": None, "zodiac": None, "method": "綜合命理"}
    all_numbers = []
    all_specials = []

    # Get Ziwei predictions
    try:
        ziwei_nums, ziwei_sp, ziwei_details = predict_ziwei(lottery_type)
        all_numbers.extend(ziwei_nums)
        all_specials.append(ziwei_sp)
        details["ziwei"] = ziwei_details
    except Exception as e:
        details["ziwei"] = {"error": str(e)}

    # Get Zodiac predictions
    try:
        zodiac_nums, zodiac_sp, zodiac_details = predict_zodiac(lottery_type)
        all_numbers.extend(zodiac_nums)
        all_specials.append(zodiac_sp)
        details["zodiac"] = zodiac_details
    except Exception as e:
        details["zodiac"] = {"error": str(e)}

    if not all_numbers:
        raise ValueError("Both Ziwei and Zodiac predictions failed")

    # Combine by frequency voting
    number_counts = Counter(all_numbers)
    top_6 = [n for n, _ in number_counts.most_common(6)]

    special_counts = Counter(all_specials)
    special = special_counts.most_common(1)[0][0] if all_specials else 1

    details["final_numbers"] = sorted(top_6[:6])
    details["final_special"] = special

    return sorted(top_6[:6]), special, details


# Profile management functions (exposed for API)
def add_profile(name: str, birth_year: int, birth_month: int,
                birth_day: int, birth_hour: int,
                family_group: str = 'default', relationship: str = '',
                user_id: int = None) -> dict:
    """Add a new birth profile."""
    return get_profile_manager().add_profile(
        name, birth_year, birth_month, birth_day, birth_hour,
        family_group, relationship, user_id
    )


def get_profile(name: str, user_id: int = None) -> Optional[dict]:
    """Get a profile by name."""
    return get_profile_manager().get_profile(name, user_id)


def get_all_profiles(user_id: int = None) -> list:
    """Get all profiles."""
    return get_profile_manager().get_all_profiles(user_id)


def get_profiles_by_family(family_group: str, user_id: int = None) -> list:
    """Get all profiles in a specific family group."""
    return get_profile_manager().get_profiles_by_family(family_group, user_id)


def get_all_family_groups(user_id: int = None) -> list:
    """Get list of all family groups."""
    return get_profile_manager().get_all_family_groups(user_id)


def delete_profile(name: str, user_id: int = None) -> bool:
    """Delete a profile."""
    return get_profile_manager().delete_profile(name, user_id)


def has_profiles(user_id: int = None) -> bool:
    """Check if any profiles exist."""
    return len(get_profile_manager().get_all_profiles(user_id)) > 0


# Cache management functions (exposed for API)
def get_cache_stats(user_id: int = None) -> dict:
    """Get prediction cache statistics."""
    return get_cache_manager().get_cache_stats(user_id)


def clear_all_prediction_cache(user_id: int = None) -> int:
    """Clear all prediction cache. Returns number of entries deleted."""
    return get_cache_manager().clear_all_cache(user_id)


if __name__ == "__main__":
    print("=== Astrology Lottery Prediction Test ===\n")

    manager = get_profile_manager()

    # Check if we have profiles
    profiles = manager.get_all_profiles()
    if not profiles:
        print("No profiles found. Adding test profile...")
        try:
            manager.add_profile("測試用戶", 1990, 5, 15, 14)
            profiles = manager.get_all_profiles()
        except Exception as e:
            print(f"Error adding profile: {e}")

    print(f"Profiles: {len(profiles)}")
    for p in profiles:
        print(f"  - {p['name']}: {p['birth_year']}/{p['birth_month']}/{p['birth_day']}")

    print("\n--- Testing predictions ---")

    try:
        print("\n[紫微斗數]")
        nums, sp, details = predict_ziwei('big')
        print(f"Numbers: {nums} + Special: {sp}")
    except Exception as e:
        print(f"Ziwei error: {e}")

    try:
        print("\n[西洋星座]")
        nums, sp, details = predict_zodiac('big')
        print(f"Numbers: {nums} + Special: {sp}")
    except Exception as e:
        print(f"Zodiac error: {e}")
