"""
Birth Profile Manager

Manages birth date/time profiles for multiple users using SQLite database.
Supports CRUD operations for astrology-based lottery predictions.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List


# Database path
DB_PATH = Path(__file__).parent / "birth_data.db"


class BirthProfileManager:
    """Manages birth profiles stored in SQLite database."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the profile manager.

        Args:
            db_path: Optional custom database path
        """
        self.db_path = db_path or DB_PATH
        self._init_database()

    def _init_database(self):
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            # Create birth_profiles table with family_group support
            conn.execute('''
                CREATE TABLE IF NOT EXISTS birth_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    birth_year INTEGER NOT NULL,
                    birth_month INTEGER NOT NULL,
                    birth_day INTEGER NOT NULL,
                    birth_hour INTEGER NOT NULL,
                    family_group TEXT DEFAULT 'default',
                    relationship TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create prediction_cache table for caching astrology predictions
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lottery_type TEXT NOT NULL,
                    period INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    profile_ids TEXT NOT NULL,
                    numbers TEXT NOT NULL,
                    special INTEGER NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(lottery_type, period, method, profile_ids)
                )
            ''')
            # Add family_group column if not exists (for existing databases)
            try:
                conn.execute('ALTER TABLE birth_profiles ADD COLUMN family_group TEXT DEFAULT "default"')
            except sqlite3.OperationalError:
                pass  # Column already exists
            try:
                conn.execute('ALTER TABLE birth_profiles ADD COLUMN relationship TEXT DEFAULT ""')
            except sqlite3.OperationalError:
                pass  # Column already exists
            conn.commit()

    def add_profile(self, name: str, birth_year: int, birth_month: int,
                    birth_day: int, birth_hour: int,
                    family_group: str = 'default', relationship: str = '') -> dict:
        """
        Add a new birth profile.

        Args:
            name: Person's name (unique identifier)
            birth_year: Birth year (e.g., 1990)
            birth_month: Birth month (1-12)
            birth_day: Birth day (1-31)
            birth_hour: Birth hour (0-23)
            family_group: Family group name (e.g., "王家", "李家")
            relationship: Relationship in family (e.g., "父親", "母親", "長子")

        Returns:
            dict: Created profile data

        Raises:
            ValueError: If name already exists or invalid data
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValueError("Name cannot be empty")
        if not (1900 <= birth_year <= datetime.now().year):
            raise ValueError(f"Invalid birth year: {birth_year}")
        if not (1 <= birth_month <= 12):
            raise ValueError(f"Invalid birth month: {birth_month}")
        if not (1 <= birth_day <= 31):
            raise ValueError(f"Invalid birth day: {birth_day}")
        if not (0 <= birth_hour <= 23):
            raise ValueError(f"Invalid birth hour: {birth_hour}")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO birth_profiles
                    (name, birth_year, birth_month, birth_day, birth_hour, family_group, relationship)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (name.strip(), birth_year, birth_month, birth_day, birth_hour,
                      family_group.strip() or 'default', relationship.strip()))
                conn.commit()

                return self.get_profile(name)

        except sqlite3.IntegrityError:
            raise ValueError(f"Profile with name '{name}' already exists")

    def get_profile(self, name: str) -> Optional[dict]:
        """
        Get a profile by name.

        Args:
            name: Person's name

        Returns:
            dict: Profile data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM birth_profiles WHERE name = ?',
                (name.strip(),)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_profiles(self) -> list:
        """
        Get all profiles.

        Returns:
            list: List of all profile dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM birth_profiles ORDER BY family_group, name')
            return [dict(row) for row in cursor.fetchall()]

    def get_profiles_by_family(self, family_group: str) -> list:
        """
        Get all profiles in a specific family group.

        Args:
            family_group: Family group name

        Returns:
            list: List of profile dicts in the family
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM birth_profiles WHERE family_group = ? ORDER BY name',
                (family_group.strip(),)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_family_groups(self) -> list:
        """
        Get list of all unique family groups.

        Returns:
            list: List of family group names with member counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT family_group, COUNT(*) as member_count
                FROM birth_profiles
                GROUP BY family_group
                ORDER BY family_group
            ''')
            return [{"name": row[0], "member_count": row[1]} for row in cursor.fetchall()]

    def update_profile(self, name: str, **kwargs) -> Optional[dict]:
        """
        Update an existing profile.

        Args:
            name: Person's name
            **kwargs: Fields to update (birth_year, birth_month, birth_day, birth_hour)

        Returns:
            dict: Updated profile data or None if not found
        """
        profile = self.get_profile(name)
        if not profile:
            return None

        # Build update query
        valid_fields = ['birth_year', 'birth_month', 'birth_day', 'birth_hour', 'family_group', 'relationship']
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}

        if not updates:
            return profile

        updates['updated_at'] = datetime.now().isoformat()
        set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
        values = list(updates.values()) + [name]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f'UPDATE birth_profiles SET {set_clause} WHERE name = ?',
                values
            )
            conn.commit()

        return self.get_profile(name)

    def delete_profile(self, name: str) -> bool:
        """
        Delete a profile by name and clear related cache.

        Args:
            name: Person's name

        Returns:
            bool: True if deleted, False if not found
        """
        profile = self.get_profile(name)
        if not profile:
            return False

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'DELETE FROM birth_profiles WHERE name = ?',
                (name.strip(),)
            )
            # Clear all cache entries that include this profile
            conn.execute('DELETE FROM prediction_cache WHERE profile_ids LIKE ?',
                        (f'%{profile["id"]}%',))
            conn.commit()
            return cursor.rowcount > 0

    def get_zodiac_sign(self, month: int, day: int) -> str:
        """
        Get Western zodiac sign from birth date.

        Args:
            month: Birth month (1-12)
            day: Birth day (1-31)

        Returns:
            str: Zodiac sign in Chinese
        """
        zodiac_dates = [
            (1, 20, "摩羯座"), (2, 19, "水瓶座"), (3, 21, "雙魚座"),
            (4, 20, "牡羊座"), (5, 21, "金牛座"), (6, 21, "雙子座"),
            (7, 23, "巨蟹座"), (8, 23, "獅子座"), (9, 23, "處女座"),
            (10, 23, "天秤座"), (11, 22, "天蠍座"), (12, 22, "射手座"),
            (12, 31, "摩羯座")
        ]
        for end_month, end_day, sign in zodiac_dates:
            if month < end_month or (month == end_month and day <= end_day):
                return sign
        return "摩羯座"

    def get_chinese_hour(self, hour: int) -> str:
        """
        Convert hour to Chinese time period (時辰).

        Args:
            hour: Hour (0-23)

        Returns:
            str: Chinese time period name
        """
        periods = [
            "子時", "丑時", "寅時", "卯時", "辰時", "巳時",
            "午時", "未時", "申時", "酉時", "戌時", "亥時"
        ]
        # Each period covers 2 hours, starting from 23:00 (子時)
        index = ((hour + 1) % 24) // 2
        return periods[index]


class PredictionCacheManager:
    """Manages prediction cache stored in SQLite database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        # Ensure tables exist
        BirthProfileManager(self.db_path)

    def get_cached_prediction(self, lottery_type: str, period: int,
                              method: str, profile_ids: List[int]) -> Optional[dict]:
        """
        Get cached prediction if exists.

        Args:
            lottery_type: 'big' or 'super'
            period: Lottery period number
            method: 'ziwei' or 'zodiac'
            profile_ids: List of profile IDs used

        Returns:
            dict with numbers, special, details or None if not cached
        """
        # Sort profile_ids to ensure consistent cache key
        sorted_ids = json.dumps(sorted(profile_ids))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT numbers, special, details FROM prediction_cache
                WHERE lottery_type = ? AND period = ? AND method = ? AND profile_ids = ?
            ''', (lottery_type, period, method, sorted_ids))
            row = cursor.fetchone()

            if row:
                return {
                    'numbers': json.loads(row['numbers']),
                    'special': row['special'],
                    'details': json.loads(row['details']) if row['details'] else {}
                }
            return None

    def save_prediction(self, lottery_type: str, period: int, method: str,
                       profile_ids: List[int], numbers: List[int],
                       special: int, details: dict) -> None:
        """
        Save prediction to cache.

        Args:
            lottery_type: 'big' or 'super'
            period: Lottery period number
            method: 'ziwei' or 'zodiac'
            profile_ids: List of profile IDs used
            numbers: Predicted numbers
            special: Special number
            details: Full prediction details
        """
        sorted_ids = json.dumps(sorted(profile_ids))
        numbers_json = json.dumps(numbers)
        details_json = json.dumps(details, ensure_ascii=False)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO prediction_cache
                (lottery_type, period, method, profile_ids, numbers, special, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (lottery_type, period, method, sorted_ids, numbers_json, special, details_json))
            conn.commit()

    def clear_cache_for_profile(self, profile_id: int) -> int:
        """
        Clear all cache entries containing a specific profile.

        Args:
            profile_id: The profile ID to clear cache for

        Returns:
            int: Number of cache entries deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'DELETE FROM prediction_cache WHERE profile_ids LIKE ?',
                (f'%{profile_id}%',)
            )
            conn.commit()
            return cursor.rowcount

    def clear_all_cache(self) -> int:
        """Clear all cached predictions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM prediction_cache')
            conn.commit()
            return cursor.rowcount

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT COUNT(*) FROM prediction_cache')
            total = cursor.fetchone()[0]

            cursor = conn.execute('''
                SELECT lottery_type, method, COUNT(*) as count
                FROM prediction_cache
                GROUP BY lottery_type, method
            ''')
            breakdown = [{"lottery_type": r[0], "method": r[1], "count": r[2]}
                        for r in cursor.fetchall()]

            return {"total_cached": total, "breakdown": breakdown}


if __name__ == "__main__":
    # Test the profile manager
    manager = BirthProfileManager()

    # Add test profile
    try:
        profile = manager.add_profile(
            name="測試用戶",
            birth_year=1990,
            birth_month=5,
            birth_day=15,
            birth_hour=14
        )
        print(f"Added profile: {profile}")
    except ValueError as e:
        print(f"Profile exists: {e}")

    # List all profiles
    profiles = manager.get_all_profiles()
    print(f"\nAll profiles ({len(profiles)}):")
    for p in profiles:
        zodiac = manager.get_zodiac_sign(p['birth_month'], p['birth_day'])
        hour_name = manager.get_chinese_hour(p['birth_hour'])
        print(f"  - {p['name']}: {p['birth_year']}/{p['birth_month']}/{p['birth_day']} {hour_name} ({zodiac})")
