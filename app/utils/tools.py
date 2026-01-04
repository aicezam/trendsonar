"""
本文件用于提供通用工具函数，当前主要用于解析自然语言中的时间范围提示。
主要函数:
- `parse_query_time_range`: 从问句解析 `(start, end)` 时间范围
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional


def clean_html_tags(text: str) -> str:
    """
    深度清洗文本：
    1. 去除所有 HTML 标签
    2. 去除 Markdown 图片/链接语法
    3. 去除 URL 链接
    4. 去除多余空白
    只保留纯文本内容。
    """
    if not text:
        return ""
    
    # 0. 预处理：移除脚本和样式内容 (防止残留 js 代码)
    clean = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 1. 移除 Markdown 图片 ![alt](url) 和 链接 [text](url)
    # 先去图片
    clean = re.sub(r'!\[.*?\]\(.*?\)', ' ', clean)
    # 再去链接（保留链接文本，去掉URL部分）-> [text](url) => text
    # 注意：这里我们选择去除链接语法保留文本，或者直接把链接部分去掉
    # 如果要保留文本：
    clean = re.sub(r'\[([^\]]+)\]\(.*?\)', r'\1', clean)
    
    # 2. 移除 HTML 标签
    clean = re.sub(r'<[^>]+>', ' ', clean)
    
    # 3. 移除末尾可能的残缺标签
    clean = re.sub(r'<[a-zA-Z/][^>]*$', '', clean)
    
    # 4. 移除裸露的 URL (http/https 开头)
    clean = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', clean)

    # 5. 解码 HTML 实体 (如 &nbsp; -> space, &lt; -> <)
    # 由于没引入 html 库，这里做简单的常见实体替换，或者引入 html 模块
    import html
    clean = html.unescape(clean)

    # 6. 移除多余空白
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    return clean



def parse_query_time_range(query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    输入:
    - `query`: 用户自然语言问题（可包含“今天/昨天/本周/本月/今年/最近”等时间提示）

    输出:
    - `(start, end)` 时间范围；为空表示不限制

    作用:
    - 从问句中解析时间范围，用于对新闻候选集进行时间过滤
    """

    now = datetime.now()
    today = datetime.combine(now.date(), time.min)

    if "所有时间" in query or "全部时间" in query:
        return None, None

    if "今年" in query:
        return today.replace(month=1, day=1), None

    if "前天" in query:
        return today - timedelta(days=2), today - timedelta(days=1)

    if "昨天" in query:
        return today - timedelta(days=1), today

    if "今天" in query or "今日" in query:
        return today, None

    if "本周" in query:
        return today - timedelta(days=today.weekday()), None

    if "本月" in query:
        return today.replace(day=1), None

    if "最近" in query or "近几天" in query:
        return today - timedelta(days=7), None

    return today - timedelta(days=30), None


_COUNTRY_ALIASES = {
    "中华人民共和国": "中国",
    "中国大陆": "中国",
    "中国内地": "中国",
    "china": "中国",
    "美国": "美国",
    "美利坚合众国": "美国",
    "usa": "美国",
    "u.s.a.": "美国",
    "united states": "美国",
    "united states of america": "美国",
    "日本": "日本",
    "japan": "日本",
    "韩国": "韩国",
    "南韩": "韩国",
    "korea": "韩国",
    "俄罗斯": "俄罗斯",
    "俄国": "俄罗斯",
    "russia": "俄罗斯",
    "英国": "英国",
    "uk": "英国",
    "united kingdom": "英国",
    "法国": "法国",
    "france": "法国",
    "德国": "德国",
    "germany": "德国",
    "印度": "印度",
    "india": "印度",
    "加拿大": "加拿大",
    "canada": "加拿大",
    "澳大利亚": "澳大利亚",
    "australia": "澳大利亚",
    "意大利": "意大利",
    "italy": "意大利",
    "西班牙": "西班牙",
    "spain": "西班牙",
    "乌克兰": "乌克兰",
    "ukraine": "乌克兰",
    "以色列": "以色列",
    "israel": "以色列",
    "伊朗": "伊朗",
    "iran": "伊朗",
    "土耳其": "土耳其",
    "turkey": "土耳其",
    "巴西": "巴西",
    "brazil": "巴西",
    "墨西哥": "墨西哥",
    "mexico": "墨西哥",
    "新加坡": "新加坡",
    "singapore": "新加坡",
    "马来西亚": "马来西亚",
    "malaysia": "马来西亚",
    "印度尼西亚": "印度尼西亚",
    "印尼": "印度尼西亚",
    "indonesia": "印度尼西亚",
    "泰国": "泰国",
    "thailand": "泰国",
    "越南": "越南",
    "vietnam": "越南",
    "菲律宾": "菲律宾",
    "philippines": "菲律宾",
    "阿联酋": "阿联酋",
    "阿拉伯联合酋长国": "阿联酋",
    "uae": "阿联酋",
    "沙特": "沙特",
    "沙特阿拉伯": "沙特",
    "saudi": "沙特",
    "南非": "南非",
    "south africa": "南非",
    "埃及": "埃及",
    "egypt": "埃及",
    "全球": "全球",
    "世界": "全球",
}

_CHINA_SUBREGION_KEYWORDS = (
    "省",
    "市",
    "区",
    "县",
    "州",
    "旗",
    "盟",
    "乡",
    "镇",
    "村",
    "自治区",
    "特别行政区",
)

_CHINA_SUBREGIONS_EXACT = {
    "北京",
    "上海",
    "天津",
    "重庆",
    "河北",
    "山西",
    "辽宁",
    "吉林",
    "黑龙江",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "海南",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "台湾",
    "内蒙古",
    "广西",
    "西藏",
    "宁夏",
    "新疆",
    "香港",
    "澳门",
}

_MACRO_REGION_KEYWORDS = (
    "东亚",
    "东南亚",
    "南亚",
    "西亚",
    "中亚",
    "北美",
    "南美",
    "欧洲",
    "非洲",
    "中东",
    "拉美",
    "亚太",
)


def normalize_regions_to_countries(region_text: Optional[str]) -> str:
    if not region_text:
        return "其他"

    tokens = [t.strip() for t in str(region_text).replace("，", ",").split(",") if t.strip()]
    if not tokens:
        return "其他"

    countries = []
    seen = set()

    for raw in tokens:
        key = raw.strip()
        low = key.lower()
        mapped = _COUNTRY_ALIASES.get(low) or _COUNTRY_ALIASES.get(key) or _COUNTRY_ALIASES.get(low.replace(".", ""))
        if mapped:
            if mapped not in seen:
                countries.append(mapped)
                seen.add(mapped)
            continue

        if any(k in key for k in _MACRO_REGION_KEYWORDS):
            continue

        if "中国" in key:
            if "中国" not in seen:
                countries.append("中国")
                seen.add("中国")
            continue

        if key in _CHINA_SUBREGIONS_EXACT:
            if "中国" not in seen:
                countries.append("中国")
                seen.add("中国")
            continue

        if any(k in key for k in _CHINA_SUBREGION_KEYWORDS):
            if "中国" not in seen:
                countries.append("中国")
                seen.add("中国")
            continue

    if not countries:
        return "其他"
    return ",".join(countries)
