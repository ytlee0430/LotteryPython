"""
Birth Profile Manager

Manages birth date/time profiles for multiple users using SQLite database.
Supports CRUD operations for astrology-based lottery predictions.
Includes user authentication and per-user profile isolation.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from werkzeug.security import generate_password_hash, check_password_hash


# Database path
DB_PATH = Path(__file__).parent / "birth_data.db"


class User:
    """User model for Flask-Login compatibility."""

    def __init__(self, id: int, username: str, email: str = None, created_at: str = None):
        self.id = id
        self.username = username
        self.email = email
        self.created_at = created_at
        self.is_authenticated = True
        self.is_active = True
        self.is_anonymous = False

    def get_id(self):
        return str(self.id)

    @staticmethod
    def from_dict(data: dict) -> 'User':
        return User(
            id=data['id'],
            username=data['username'],
            email=data.get('email'),
            created_at=data.get('created_at')
        )


class UserManager:
    """Manages user accounts stored in SQLite database."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._init_database()

    def _init_database(self):
        """Create users table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def create_user(self, username: str, password: str, email: str = None) -> dict:
        """
        Create a new user account.

        Args:
            username: Unique username
            password: Plain text password (will be hashed)
            email: Optional email address

        Returns:
            dict: Created user data (without password)

        Raises:
            ValueError: If username already exists or invalid data
        """
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
        if not password or len(password) < 4:
            raise ValueError("Password must be at least 4 characters")

        password_hash = generate_password_hash(password)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    INSERT INTO users (username, password_hash, email)
                    VALUES (?, ?, ?)
                ''', (username.strip(), password_hash, email.strip() if email else None))
                conn.commit()

                return self.get_user_by_id(cursor.lastrowid)
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")

    def authenticate(self, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user with username and password.

        Args:
            username: Username
            password: Plain text password

        Returns:
            User object if authentication successful, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT * FROM users WHERE username = ?',
                (username.strip(),)
            )
            row = cursor.fetchone()

            if row and check_password_hash(row['password_hash'], password):
                return User.from_dict(dict(row))
            return None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Get user by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT id, username, email, created_at FROM users WHERE id = ?',
                (user_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT id, username, email, created_at FROM users WHERE username = ?',
                (username.strip(),)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def load_user(self, user_id: int) -> Optional[User]:
        """Load User object by ID (for Flask-Login)."""
        data = self.get_user_by_id(user_id)
        return User.from_dict(data) if data else None

    def get_all_users(self) -> list:
        """Get all users (without passwords)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT id, username, email, created_at FROM users ORDER BY username'
            )
            return [dict(row) for row in cursor.fetchall()]


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
            # Ensure users table exists first
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create birth_profiles table with family_group and user_id support
            conn.execute('''
                CREATE TABLE IF NOT EXISTS birth_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    birth_year INTEGER NOT NULL,
                    birth_month INTEGER NOT NULL,
                    birth_day INTEGER NOT NULL,
                    birth_hour INTEGER NOT NULL,
                    family_group TEXT DEFAULT 'default',
                    relationship TEXT DEFAULT '',
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(name, user_id)
                )
            ''')
            # Create prediction_cache table with user_id support
            conn.execute('''
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    lottery_type TEXT NOT NULL,
                    period INTEGER NOT NULL,
                    method TEXT NOT NULL,
                    profile_ids TEXT NOT NULL,
                    numbers TEXT NOT NULL,
                    special INTEGER NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, lottery_type, period, method, profile_ids),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            # Create all_predictions_cache table with user_id support
            conn.execute('''
                CREATE TABLE IF NOT EXISTS all_predictions_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    lottery_type TEXT NOT NULL,
                    period TEXT NOT NULL,
                    results TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, lottery_type, period),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            # Migration: Add columns for existing databases
            migration_queries = [
                'ALTER TABLE birth_profiles ADD COLUMN family_group TEXT DEFAULT "default"',
                'ALTER TABLE birth_profiles ADD COLUMN relationship TEXT DEFAULT ""',
                'ALTER TABLE birth_profiles ADD COLUMN user_id INTEGER',
                'ALTER TABLE prediction_cache ADD COLUMN user_id INTEGER',
                'ALTER TABLE all_predictions_cache ADD COLUMN user_id INTEGER',
            ]
            for query in migration_queries:
                try:
                    conn.execute(query)
                except sqlite3.OperationalError:
                    pass  # Column already exists
            conn.commit()

    def add_profile(self, name: str, birth_year: int, birth_month: int,
                    birth_day: int, birth_hour: int,
                    family_group: str = 'default', relationship: str = '',
                    user_id: int = None) -> dict:
        """
        Add a new birth profile.

        Args:
            name: Person's name (unique per user)
            birth_year: Birth year (e.g., 1990)
            birth_month: Birth month (1-12)
            birth_day: Birth day (1-31)
            birth_hour: Birth hour (0-23)
            family_group: Family group name (e.g., "王家", "李家")
            relationship: Relationship in family (e.g., "父親", "母親", "長子")
            user_id: Owner user's ID

        Returns:
            dict: Created profile data

        Raises:
            ValueError: If name already exists for this user or invalid data
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
                    (name, birth_year, birth_month, birth_day, birth_hour, family_group, relationship, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (name.strip(), birth_year, birth_month, birth_day, birth_hour,
                      family_group.strip() or 'default', relationship.strip(), user_id))
                conn.commit()

                return self.get_profile(name, user_id)

        except sqlite3.IntegrityError:
            raise ValueError(f"Profile with name '{name}' already exists")

    def get_profile(self, name: str, user_id: int = None) -> Optional[dict]:
        """
        Get a profile by name.

        Args:
            name: Person's name
            user_id: Owner user's ID (if None, searches all profiles)

        Returns:
            dict: Profile data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if user_id is not None:
                cursor = conn.execute(
                    'SELECT * FROM birth_profiles WHERE name = ? AND user_id = ?',
                    (name.strip(), user_id)
                )
            else:
                cursor = conn.execute(
                    'SELECT * FROM birth_profiles WHERE name = ?',
                    (name.strip(),)
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_profiles(self, user_id: int = None) -> list:
        """
        Get all profiles for a user.

        Args:
            user_id: Owner user's ID (if None, returns all profiles)

        Returns:
            list: List of profile dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if user_id is not None:
                cursor = conn.execute(
                    'SELECT * FROM birth_profiles WHERE user_id = ? ORDER BY family_group, name',
                    (user_id,)
                )
            else:
                cursor = conn.execute('SELECT * FROM birth_profiles ORDER BY family_group, name')
            return [dict(row) for row in cursor.fetchall()]

    def get_profiles_by_family(self, family_group: str, user_id: int = None) -> list:
        """
        Get all profiles in a specific family group.

        Args:
            family_group: Family group name
            user_id: Owner user's ID

        Returns:
            list: List of profile dicts in the family
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if user_id is not None:
                cursor = conn.execute(
                    'SELECT * FROM birth_profiles WHERE family_group = ? AND user_id = ? ORDER BY name',
                    (family_group.strip(), user_id)
                )
            else:
                cursor = conn.execute(
                    'SELECT * FROM birth_profiles WHERE family_group = ? ORDER BY name',
                    (family_group.strip(),)
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_family_groups(self, user_id: int = None) -> list:
        """
        Get list of all unique family groups.

        Args:
            user_id: Owner user's ID

        Returns:
            list: List of family group names with member counts
        """
        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                cursor = conn.execute('''
                    SELECT family_group, COUNT(*) as member_count
                    FROM birth_profiles
                    WHERE user_id = ?
                    GROUP BY family_group
                    ORDER BY family_group
                ''', (user_id,))
            else:
                cursor = conn.execute('''
                    SELECT family_group, COUNT(*) as member_count
                    FROM birth_profiles
                    GROUP BY family_group
                    ORDER BY family_group
                ''')
            return [{"name": row[0], "member_count": row[1]} for row in cursor.fetchall()]

    def update_profile(self, name: str, user_id: int = None, **kwargs) -> Optional[dict]:
        """
        Update an existing profile.

        Args:
            name: Person's name
            user_id: Owner user's ID
            **kwargs: Fields to update (birth_year, birth_month, birth_day, birth_hour)

        Returns:
            dict: Updated profile data or None if not found
        """
        profile = self.get_profile(name, user_id)
        if not profile:
            return None

        # Build update query
        valid_fields = ['birth_year', 'birth_month', 'birth_day', 'birth_hour', 'family_group', 'relationship']
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}

        if not updates:
            return profile

        updates['updated_at'] = datetime.now().isoformat()
        set_clause = ', '.join(f'{k} = ?' for k in updates.keys())

        if user_id is not None:
            values = list(updates.values()) + [name, user_id]
            where_clause = 'WHERE name = ? AND user_id = ?'
        else:
            values = list(updates.values()) + [name]
            where_clause = 'WHERE name = ?'

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f'UPDATE birth_profiles SET {set_clause} {where_clause}',
                values
            )
            conn.commit()

        return self.get_profile(name, user_id)

    def delete_profile(self, name: str, user_id: int = None) -> bool:
        """
        Delete a profile by name and clear related cache.

        Args:
            name: Person's name
            user_id: Owner user's ID

        Returns:
            bool: True if deleted, False if not found
        """
        profile = self.get_profile(name, user_id)
        if not profile:
            return False

        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                cursor = conn.execute(
                    'DELETE FROM birth_profiles WHERE name = ? AND user_id = ?',
                    (name.strip(), user_id)
                )
            else:
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
                              method: str, profile_ids: List[int],
                              user_id: int = None) -> Optional[dict]:
        """
        Get cached prediction if exists.

        Args:
            lottery_type: 'big' or 'super'
            period: Lottery period number
            method: 'ziwei' or 'zodiac'
            profile_ids: List of profile IDs used
            user_id: Owner user's ID

        Returns:
            dict with numbers, special, details or None if not cached
        """
        # Sort profile_ids to ensure consistent cache key
        sorted_ids = json.dumps(sorted(profile_ids))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if user_id is not None:
                cursor = conn.execute('''
                    SELECT numbers, special, details FROM prediction_cache
                    WHERE user_id = ? AND lottery_type = ? AND period = ? AND method = ? AND profile_ids = ?
                ''', (user_id, lottery_type, period, method, sorted_ids))
            else:
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
                       special: int, details: dict, user_id: int = None) -> None:
        """
        Save prediction to cache. Only keeps latest period per user.

        Args:
            lottery_type: 'big' or 'super'
            period: Lottery period number
            method: 'ziwei' or 'zodiac'
            profile_ids: List of profile IDs used
            numbers: Predicted numbers
            special: Special number
            details: Full prediction details
            user_id: Owner user's ID
        """
        sorted_ids = json.dumps(sorted(profile_ids))
        numbers_json = json.dumps(numbers)
        details_json = json.dumps(details, ensure_ascii=False)

        with sqlite3.connect(self.db_path) as conn:
            # Clear old cache for this user/type/method (only keep latest period)
            if user_id is not None:
                conn.execute('''
                    DELETE FROM prediction_cache
                    WHERE user_id = ? AND lottery_type = ? AND method = ?
                ''', (user_id, lottery_type, method))
            else:
                conn.execute('''
                    DELETE FROM prediction_cache
                    WHERE user_id IS NULL AND lottery_type = ? AND method = ?
                ''', (lottery_type, method))

            conn.execute('''
                INSERT INTO prediction_cache
                (user_id, lottery_type, period, method, profile_ids, numbers, special, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, lottery_type, period, method, sorted_ids, numbers_json, special, details_json))
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

    def clear_all_cache(self, user_id: int = None) -> int:
        """Clear all cached predictions for a user."""
        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                cursor = conn.execute('DELETE FROM prediction_cache WHERE user_id = ?', (user_id,))
            else:
                cursor = conn.execute('DELETE FROM prediction_cache')
            conn.commit()
            return cursor.rowcount

    def get_cache_stats(self, user_id: int = None) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                cursor = conn.execute('SELECT COUNT(*) FROM prediction_cache WHERE user_id = ?', (user_id,))
            else:
                cursor = conn.execute('SELECT COUNT(*) FROM prediction_cache')
            total = cursor.fetchone()[0]

            if user_id is not None:
                cursor = conn.execute('''
                    SELECT lottery_type, method, COUNT(*) as count
                    FROM prediction_cache WHERE user_id = ?
                    GROUP BY lottery_type, method
                ''', (user_id,))
            else:
                cursor = conn.execute('''
                    SELECT lottery_type, method, COUNT(*) as count
                    FROM prediction_cache
                    GROUP BY lottery_type, method
                ''')
            breakdown = [{"lottery_type": r[0], "method": r[1], "count": r[2]}
                        for r in cursor.fetchall()]

            return {"total_cached": total, "breakdown": breakdown}


class AllPredictionsCacheManager:
    """Manages cache for all algorithm predictions."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        # Ensure tables exist
        BirthProfileManager(self.db_path)

    def get_cached_predictions(self, lottery_type: str, period: str,
                               user_id: int = None) -> Optional[dict]:
        """
        Get cached predictions for all algorithms.

        Args:
            lottery_type: 'big' or 'super'
            period: Lottery period number (as string)
            user_id: Owner user's ID

        Returns:
            dict with all algorithm results or None if not cached
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if user_id is not None:
                cursor = conn.execute('''
                    SELECT results FROM all_predictions_cache
                    WHERE user_id = ? AND lottery_type = ? AND period = ?
                ''', (user_id, lottery_type, period))
            else:
                cursor = conn.execute('''
                    SELECT results FROM all_predictions_cache
                    WHERE lottery_type = ? AND period = ?
                ''', (lottery_type, period))
            row = cursor.fetchone()

            if row:
                return json.loads(row['results'])
            return None

    def save_predictions(self, lottery_type: str, period: str, results: dict,
                        user_id: int = None) -> None:
        """
        Save all algorithm predictions to cache. Only keeps latest period per user.

        Args:
            lottery_type: 'big' or 'super'
            period: Lottery period number (as string)
            results: All algorithm results dict
            user_id: Owner user's ID
        """
        results_json = json.dumps(results, ensure_ascii=False)

        with sqlite3.connect(self.db_path) as conn:
            # Clear old cache for this user/type (only keep latest period)
            if user_id is not None:
                conn.execute('''
                    DELETE FROM all_predictions_cache
                    WHERE user_id = ? AND lottery_type = ?
                ''', (user_id, lottery_type))
            else:
                conn.execute('''
                    DELETE FROM all_predictions_cache
                    WHERE user_id IS NULL AND lottery_type = ?
                ''', (lottery_type,))

            conn.execute('''
                INSERT INTO all_predictions_cache
                (user_id, lottery_type, period, results)
                VALUES (?, ?, ?, ?)
            ''', (user_id, lottery_type, period, results_json))
            conn.commit()

    def clear_all_cache(self, user_id: int = None) -> int:
        """Clear all cached predictions for a user."""
        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                cursor = conn.execute('DELETE FROM all_predictions_cache WHERE user_id = ?', (user_id,))
            else:
                cursor = conn.execute('DELETE FROM all_predictions_cache')
            conn.commit()
            return cursor.rowcount

    def get_cache_stats(self, user_id: int = None) -> dict:
        """Get cache statistics."""
        with sqlite3.connect(self.db_path) as conn:
            if user_id is not None:
                cursor = conn.execute('SELECT COUNT(*) FROM all_predictions_cache WHERE user_id = ?', (user_id,))
            else:
                cursor = conn.execute('SELECT COUNT(*) FROM all_predictions_cache')
            total = cursor.fetchone()[0]

            if user_id is not None:
                cursor = conn.execute('''
                    SELECT lottery_type, period, created_at
                    FROM all_predictions_cache WHERE user_id = ?
                    ORDER BY created_at DESC
                ''', (user_id,))
            else:
                cursor = conn.execute('''
                    SELECT lottery_type, period, created_at
                    FROM all_predictions_cache
                    ORDER BY created_at DESC
                ''')
            entries = [{"lottery_type": r[0], "period": r[1], "created_at": r[2]}
                      for r in cursor.fetchall()]

            return {"total_cached": total, "entries": entries}


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
