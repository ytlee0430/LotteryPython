"""
Gemini CLI Client for Astrology Predictions

Wraps Gemini CLI calls for generating lottery number predictions
based on Chinese (紫微斗數) and Western (星座) astrology.
"""

import subprocess
import json
import re
from typing import Optional


class GeminiAstrologyClient:
    """Client for making astrology predictions via Gemini CLI."""

    def __init__(self, timeout: int = 120):
        """
        Initialize the Gemini client.

        Args:
            timeout: Maximum seconds to wait for Gemini response
        """
        self.timeout = timeout
        self.gemini_path = self._find_gemini()

    def _find_gemini(self) -> str:
        """Find the gemini CLI executable path."""
        import shutil
        path = shutil.which('gemini')
        if not path:
            raise RuntimeError("Gemini CLI not found. Please install it first.")
        return path

    def _call_gemini(self, prompt: str) -> str:
        """
        Call Gemini CLI with a prompt.

        Args:
            prompt: The prompt to send to Gemini

        Returns:
            str: Gemini's response text
        """
        try:
            result = subprocess.run(
                [self.gemini_path, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Gemini CLI timed out after {self.timeout} seconds")
        except Exception as e:
            raise RuntimeError(f"Gemini CLI error: {e}")

    def _extract_json(self, text: str) -> dict:
        """
        Extract JSON from Gemini response text.

        Args:
            text: Response text that may contain JSON

        Returns:
            dict: Parsed JSON data
        """
        # Try to find JSON in the response
        # Look for pattern like {"numbers": [...], "special": ...}
        json_pattern = r'\{[^{}]*"numbers"[^{}]*\}'
        match = re.search(json_pattern, text, re.DOTALL)

        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try to parse the entire response as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract numbers from text
        numbers = re.findall(r'\b([1-4]?[0-9])\b', text)
        numbers = [int(n) for n in numbers if 1 <= int(n) <= 49]

        if len(numbers) >= 7:
            return {
                "numbers": numbers[:6],
                "special": numbers[6] if len(numbers) > 6 else numbers[0],
                "raw_response": text[:200]
            }

        raise ValueError(f"Could not parse Gemini response: {text[:200]}")

    def predict_ziwei(self, profile: dict, lottery_type: str = 'big') -> dict:
        """
        Get lottery predictions using 紫微斗數 (Purple Star Astrology).

        Args:
            profile: Birth profile dict with name, birth_year, birth_month, birth_day, birth_hour
            lottery_type: 'big' for 大樂透 (1-49) or 'super' for 威力彩 (1-38)

        Returns:
            dict: Prediction with numbers, special, and analysis
        """
        max_num = 49 if lottery_type == 'big' else 38
        lottery_name = "大樂透" if lottery_type == 'big' else "威力彩"

        # Convert hour to Chinese time period
        hour_periods = [
            "子時(23-1點)", "丑時(1-3點)", "寅時(3-5點)", "卯時(5-7點)",
            "辰時(7-9點)", "巳時(9-11點)", "午時(11-13點)", "未時(13-15點)",
            "申時(15-17點)", "酉時(17-19點)", "戌時(19-21點)", "亥時(21-23點)"
        ]
        hour_index = ((profile['birth_hour'] + 1) % 24) // 2
        chinese_hour = hour_periods[hour_index]

        prompt = f'''你是一位精通紫微斗數的命理大師。請根據以下生辰資料進行命盤分析，並推薦最適合購買{lottery_name}的號碼。

生辰資料:
- 姓名: {profile['name']}
- 國曆出生年: {profile['birth_year']}年
- 國曆出生月: {profile['birth_month']}月
- 國曆出生日: {profile['birth_day']}日
- 出生時辰: {chinese_hour}

請根據此人的紫微命盤特質（如命宮、財帛宮、遷移宮等），分析其財運走勢，並推薦:
- 6 個主要號碼 (範圍 1-{max_num})
- 1 個特別號 (範圍 1-{max_num})

重要：請只回傳純 JSON 格式，不要有任何其他文字或解釋:
{{"numbers": [1, 2, 3, 4, 5, 6], "special": 7, "analysis": "簡短命理分析(30字內)"}}'''

        response = self._call_gemini(prompt)
        result = self._extract_json(response)

        # Validate and fix numbers
        result['numbers'] = self._validate_numbers(result.get('numbers', []), max_num, 6)
        result['special'] = self._validate_number(result.get('special', 1), max_num)
        result['method'] = '紫微斗數'
        result['profile_name'] = profile['name']

        return result

    def predict_zodiac(self, profile: dict, lottery_type: str = 'big') -> dict:
        """
        Get lottery predictions using Western Zodiac astrology.

        Args:
            profile: Birth profile dict with name, birth_year, birth_month, birth_day
            lottery_type: 'big' for 大樂透 (1-49) or 'super' for 威力彩 (1-38)

        Returns:
            dict: Prediction with numbers, special, zodiac sign, and lucky elements
        """
        max_num = 49 if lottery_type == 'big' else 38
        lottery_name = "大樂透" if lottery_type == 'big' else "威力彩"

        # Calculate zodiac sign
        zodiac = self._get_zodiac_sign(profile['birth_month'], profile['birth_day'])

        prompt = f'''你是一位專業的西洋占星術師。請根據以下星座資料分析運勢，並推薦最適合購買{lottery_name}的號碼。

個人資料:
- 姓名: {profile['name']}
- 星座: {zodiac}
- 出生日期: {profile['birth_year']}年{profile['birth_month']}月{profile['birth_day']}日

請根據此星座的特質、幸運數字、本週運勢等，推薦:
- 6 個主要號碼 (範圍 1-{max_num})
- 1 個特別號 (範圍 1-{max_num})

重要：請只回傳純 JSON 格式，不要有任何其他文字或解釋:
{{"numbers": [1, 2, 3, 4, 5, 6], "special": 7, "zodiac": "{zodiac}", "lucky_element": "幸運元素"}}'''

        response = self._call_gemini(prompt)
        result = self._extract_json(response)

        # Validate and fix numbers
        result['numbers'] = self._validate_numbers(result.get('numbers', []), max_num, 6)
        result['special'] = self._validate_number(result.get('special', 1), max_num)
        result['zodiac'] = zodiac
        result['method'] = '西洋星座'
        result['profile_name'] = profile['name']

        return result

    def _get_zodiac_sign(self, month: int, day: int) -> str:
        """Get Western zodiac sign from birth date."""
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

    def _validate_numbers(self, numbers: list, max_num: int, count: int) -> list:
        """Validate and ensure we have the right count of valid numbers."""
        import random

        valid = []
        for n in numbers:
            try:
                num = int(n)
                if 1 <= num <= max_num and num not in valid:
                    valid.append(num)
            except (ValueError, TypeError):
                continue

        # Fill with random numbers if not enough
        while len(valid) < count:
            num = random.randint(1, max_num)
            if num not in valid:
                valid.append(num)

        return sorted(valid[:count])

    def _validate_number(self, number, max_num: int) -> int:
        """Validate a single number."""
        import random
        try:
            num = int(number)
            if 1 <= num <= max_num:
                return num
        except (ValueError, TypeError):
            pass
        return random.randint(1, max_num)


if __name__ == "__main__":
    # Test the client
    client = GeminiAstrologyClient()

    test_profile = {
        'name': '測試用戶',
        'birth_year': 1990,
        'birth_month': 5,
        'birth_day': 15,
        'birth_hour': 14
    }

    print("Testing Gemini Astrology Client...")
    print(f"Profile: {test_profile}")

    try:
        print("\n=== 紫微斗數預測 ===")
        ziwei = client.predict_ziwei(test_profile, 'big')
        print(f"Numbers: {ziwei['numbers']}")
        print(f"Special: {ziwei['special']}")
        print(f"Analysis: {ziwei.get('analysis', 'N/A')}")
    except Exception as e:
        print(f"Ziwei error: {e}")

    try:
        print("\n=== 西洋星座預測 ===")
        zodiac = client.predict_zodiac(test_profile, 'big')
        print(f"Zodiac: {zodiac['zodiac']}")
        print(f"Numbers: {zodiac['numbers']}")
        print(f"Special: {zodiac['special']}")
        print(f"Lucky: {zodiac.get('lucky_element', 'N/A')}")
    except Exception as e:
        print(f"Zodiac error: {e}")
