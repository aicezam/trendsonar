"""
本文件用于编排抓取、聚类、摘要与报表生成等全流程任务，并提供定时调度入口。
主要函数:
- `scheduled_task`: 定时调度循环
- `run_manual`: 手动触发全流程
"""

import asyncio
import ctypes
import gc
import json
import os
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import delete, desc, or_, select
from sqlalchemy.orm import defer

from app.core.config import get_settings
from app.core.database import AsyncSessionLocal, check_db_connection
from app.core.logger import logger
from app.core.exceptions import AIConfigurationError
from app.models.news import News
from app.services.ai_service import ai_service
from app.services.admin_service import schedule_restart
from app.services.cluster_service import cluster_service
from app.services.crawler_service import crawler_service
from app.services.report_service import report_service
from app.services.task_manager import task_manager
from app.services.topic_service import topic_service
from app.utils.tools import normalize_regions_to_countries

settings = get_settings()

DATA_DIR = Path("data")
STATE_FILE = DATA_DIR / "scheduler_state.json"

def _read_rss_mb() -> float | None:
    if os.name != "posix":
        return None
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in status:
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    kb = int(parts[1])
                    return kb / 1024.0
    except Exception:
        return None
    return None

def _read_cgroup_memory_mb() -> float | None:
    if os.name != "posix":
        return None
    candidates = [
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ]
    for p in candidates:
        try:
            if p.exists():
                b = int(p.read_text(encoding="utf-8", errors="ignore").strip())
                return b / 1024.0 / 1024.0
        except Exception:
            continue
    return None

def _malloc_trim() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        trim = getattr(libc, "malloc_trim", None)
        if trim is None:
            return
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except Exception:
        return

def _load_scheduler_state() -> Dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        res = {}
        if "last_periodic_run" in data:
            res["last_periodic_run"] = datetime.fromisoformat(data["last_periodic_run"])
        if "last_topic_run" in data:
            res["last_topic_run"] = datetime.fromisoformat(data["last_topic_run"])
        if "last_daily_final" in data:
            res["last_daily_final"] = datetime.strptime(data["last_daily_final"], "%Y-%m-%d").date()
        if "last_weekly_final" in data:
            res["last_weekly_final"] = datetime.strptime(data["last_weekly_final"], "%Y-%m-%d").date()
        if "last_monthly_final" in data:
            res["last_monthly_final"] = datetime.strptime(data["last_monthly_final"], "%Y-%m-%d").date()
        return res
    except Exception as e:
        logger.warning(f"⚠️ 读取调度状态失败: {e}")
        return {}

def _save_scheduler_state(
    last_periodic_run: datetime,
    last_topic_run: datetime,
    last_daily_final=None,
    last_weekly_final=None,
    last_monthly_final=None
):
    try:
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            
        data = {
            "last_periodic_run": last_periodic_run.isoformat(),
            "last_topic_run": last_topic_run.isoformat(),
        }
        if last_daily_final:
            data["last_daily_final"] = last_daily_final.strftime("%Y-%m-%d")
        if last_weekly_final:
            data["last_weekly_final"] = last_weekly_final.strftime("%Y-%m-%d")
        if last_monthly_final:
            data["last_monthly_final"] = last_monthly_final.strftime("%Y-%m-%d")
            
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"❌ 保存调度状态失败: {e}")


async def auto_batch_analyze_new_news() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 对时间窗口内尚未分类的新闻进行批量情感与分类分析
    """

    logger.info("🤖 开始批量初步分析新新闻...")
    async with AsyncSessionLocal() as db:
        time_window = datetime.now() - timedelta(hours=settings.CLUSTERING_TIME_WINDOW_HOURS)
        stmt = (
            select(News)
            .options(defer(News.content), defer(News.embedding))
            .where(News.publish_date >= time_window, News.category == "其他")
            .order_by(News.id.desc())
        )

        result = await db.execute(stmt)
        news_list = result.scalars().all()

        if not news_list:
            logger.info("✅ 没有待处理的新闻")
            return

        total = len(news_list)
        logger.debug(f"   📊 待分析新闻数: {total}")

        batch_size = settings.ANALYSIS_BATCH_SIZE
        processed_count = 0

        for i in range(0, total, batch_size):
            batch = news_list[i : i + batch_size]
            batch_data = [{"id": n.id, "title": n.title} for n in batch]

            current_end = min(i + batch_size, total)
            logger.info(f"   🚀 正在分析: {i + 1}-{current_end}/{total} (本批: {len(batch)})...")
            results = await ai_service.batch_analyze_sentiment(batch_data)

            updates = 0
            for news in batch:
                if news.id in results:
                    res = results[news.id]
                    news.sentiment_label = res.get("label", "中立")
                    news.sentiment_score = res.get("score", 50)
                    news.category = res.get("category", "其他")
                    news.region = normalize_regions_to_countries(res.get("region", "其他"))
                    updates += 1

            await db.commit()
            processed_count += updates

        logger.info(f"✅ 批量分析完成，共更新 {processed_count} 条")


async def auto_generate_summaries_top_n() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 为当日热度 TopN 新闻生成摘要，并尽量补全向量与深度分析字段
    """

    top_n = settings.AUTO_SUMMARY_TOP_N
    logger.info(f"🤖 开始为今日热度Top{top_n}自动生成摘要...")
    async with AsyncSessionLocal() as db:
        today_start = datetime.combine(datetime.now().date(), time.min)

        stmt = (
            select(News)
            .options(defer(News.embedding))
            .where(News.publish_date >= today_start)
            .order_by(desc(News.heat_score))
            .limit(top_n)
        )
        result = await db.execute(stmt)
        top_news = result.scalars().all()

        # 筛选出尚未经过 AI 摘要生成的新闻 (is_ai_summary == False)
        # 注意：即便 news.summary 有值（RSS自带摘要），只要 is_ai_summary 为 False，也需要重新生成
        news_to_process = [n for n in top_news if not n.is_ai_summary]
        total_task = len(news_to_process)
        logger.debug(f"   📋 需生成摘要: {total_task} 条")

        count = 0
        for idx, news in enumerate(news_to_process, 1):
            progress_str = f"({idx}/{total_task})"
            try:
                content = news.content
                if not content or len(content) < 50:
                    logger.debug(f"   {progress_str} 🕷️ 补抓正文: {news.title}")
                    content = await crawler_service.crawl_content(news.url)
                    if content:
                        news.content = content
                        db.add(news)
                        await db.commit()
                    else:
                        logger.warning(f"   {progress_str} ❌ 无法获取正文，跳过: {news.title}")
                        continue

                if content:
                    logger.info(f"   {progress_str} 📝 生成摘要: {news.title}")
                    
                    # 组合输入：如果有原始摘要（RSS），则一起提供给 AI
                    input_content = content
                    if news.summary:
                        input_content = f"原始摘要：{news.summary}\n\n正文内容：{content}"

                    summary = await ai_service.generate_summary(news.title, input_content)
                    if summary:
                        news.summary = summary
                        news.is_ai_summary = True

                        try:
                            txt_to_embed = f"{news.title} {summary} {content[:1000]}"
                            embs = await ai_service.get_embeddings([txt_to_embed])
                            if embs and embs[0]:
                                news.embedding = embs[0]
                        except Exception as e:
                            logger.error(f"   {progress_str} ⚠️ 向量更新失败: {e}")

                        if not news.keywords:
                            try:
                                logger.debug(f"   {progress_str} 🧠 同步深度分析: {news.title}")
                                res = await ai_service.analyze_sentiment(news.title, summary)
                                if res:
                                    news.sentiment_score = res["score"]
                                    news.sentiment_label = res["label"]
                                    news.category = res.get("category", "其他")
                                    news.region = res.get("region", "其他")
                                    news.keywords = res.get("keywords", [])
                                    news.entities = res.get("entities", [])
                            except Exception as e:
                                logger.error(f"   {progress_str} ⚠️ 同步分析失败: {e}")

                        db.add(news)
                        await db.commit()
                        count += 1
            except Exception as e:
                logger.error(f"   {progress_str} ⚠️ 处理异常 ({news.title}): {e}")

        logger.info(f"✅ 自动摘要完成，共处理 {count} 条")


async def auto_generate_summaries_categories_top_n() -> None:
    """
    为每个领域的 Top 5 新闻生成摘要
    """
    logger.info("🤖 开始为各领域 Top 5 自动生成摘要...")
    categories = settings.NEWS_CATEGORIES
    top_n = 5
    
    async with AsyncSessionLocal() as db:
        today_start = datetime.combine(datetime.now().date(), time.min)
        
        for cat in categories:
            stmt = (
                select(News)
                .options(defer(News.embedding))
                .where(News.publish_date >= today_start)
                .where(News.category == cat)
                .order_by(desc(News.heat_score))
                .limit(top_n)
            )
            result = await db.execute(stmt)
            news_list = result.scalars().all()
            
            # 筛选出未生成 AI 摘要的
            news_to_process = [n for n in news_list if not n.is_ai_summary]
            if not news_to_process:
                continue
                
            logger.debug(f"   📋 [{cat}] 需生成摘要: {len(news_to_process)} 条")
            
            for i, news in enumerate(news_to_process, 1):
                try:
                    logger.info(f"   [{cat}] ({i}/{len(news_to_process)}) 📝 生成摘要: {news.title}")
                    content = news.content
                    if not content or len(content) < 50:
                        content = await crawler_service.crawl_content(news.url)
                        if content:
                            news.content = content
                            db.add(news) # 立即保存正文
                        else:
                            continue
                            
                    input_content = content
                    if news.summary:
                         input_content = f"原始摘要：{news.summary}\n\n正文内容：{content}"
                         
                    summary = await ai_service.generate_summary(news.title, input_content)
                    if summary:
                        news.summary = summary
                        news.is_ai_summary = True
                        db.add(news)
                except Exception as e:
                    logger.error(f"   ⚠️ 生成摘要失败 ({news.title}): {e}")
            
            await db.commit()
    logger.info("✅ 各领域 Top 5 摘要生成完成")


async def auto_analyze_sentiment_top_n() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 为当日热度 TopN 新闻进行深度分析（情感/关键词/实体/地区）
    """

    top_n = settings.AUTO_ANALYSIS_TOP_N
    logger.info(f"🧠 开始为今日热度Top{top_n}进行深度分析...")
    async with AsyncSessionLocal() as db:
        today_start = datetime.combine(datetime.now().date(), time.min)

        stmt = (
            select(News)
            .options(defer(News.embedding))
            .where(News.publish_date >= today_start)
            .order_by(desc(News.heat_score))
            .limit(top_n)
        )
        result = await db.execute(stmt)
        top_news = result.scalars().all()

        items_to_process = []
        for news in top_news:
            if not news.keywords or len(news.keywords) == 0:
                items_to_process.append(news)

        if not items_to_process:
            logger.info("✅ 所有Top新闻均已分析，无需处理")
            return

        logger.debug(f"   📊 待分析新闻数: {len(items_to_process)}")

        sem = asyncio.Semaphore(5)
        total_items = len(items_to_process)
        batch_size = 50

        async def analyze_task(news_item, index):
            async with sem:
                try:
                    if not news_item.content or len(news_item.content) < 50:
                        logger.debug(f"   ({index}/{total_items}) 🕷️ 补抓正文: {news_item.title}")
                        try:
                            content = await crawler_service.crawl_content(news_item.url)
                            if content:
                                news_item.content = content
                        except Exception as e:
                            logger.error(f"   ({index}/{total_items}) ⚠️ 补抓失败: {e}")

                    text = news_item.summary or news_item.content or ""
                    logger.debug(f"   ({index}/{total_items}) 🧠 分析中: {news_item.title}")
                    return await ai_service.analyze_sentiment(news_item.title, text)
                except Exception as e:
                    logger.error(f"   ({index}/{total_items}) ⚠️ 分析失败 ({news_item.title}): {e}")
                    return None

        count = 0
        for i in range(0, total_items, batch_size):
            batch = items_to_process[i : i + batch_size]
            current_end = min(i + batch_size, total_items)
            logger.info(f"   🚀 正在分析: {i + 1}-{current_end}/{total_items} (本批: {len(batch)})...")

            tasks = [analyze_task(n, i + idx + 1) for idx, n in enumerate(batch)]
            results = await asyncio.gather(*tasks)

            for news, res in zip(batch, results):
                if res:
                    news.sentiment_score = res["score"]
                    news.sentiment_label = res["label"]
                    news.category = res.get("category", "其他")
                    news.keywords = res.get("keywords", [])
                    news.entities = res.get("entities", [])
                    db.add(news)
                    count += 1

            await db.commit()
        logger.info(f"✅ 深度分析完成，共更新 {count} 条")


async def cleanup_old_data() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 清理过期且低热度的数据，控制数据库体量
    """

    enabled = bool(getattr(settings, "DATA_CLEANUP_ENABLED", False))
    if not enabled:
        logger.info("⏩ 数据清理已关闭，跳过低热新闻自动删除")
        return

    min_heat = float(getattr(settings, "DATA_CLEANUP_MIN_HEAT", 1.0) or 0.0)
    protect_days = max(0, int(getattr(settings, "DATA_CLEANUP_PROTECT_DAYS", 3) or 0))
    logger.info(f"🧹 开始清理低热新闻: protect_days={protect_days}, min_heat={min_heat}")
    async with AsyncSessionLocal() as db:
        deadline = datetime.now() - timedelta(days=protect_days)
        stmt = delete(News).where(News.publish_date < deadline, News.heat_score < min_heat)
        result = await db.execute(stmt)
        await db.commit()
        logger.info(f"🗑️ 已删除 {result.rowcount} 条低热新闻数据")


async def run_pipeline_task(generate_daily: bool = True, run_topic_task: bool = True) -> None:
    """
    输入:
    - `generate_daily`: 是否在流程中生成每日大盘报表
    - `run_topic_task`: 是否在流程中运行专题追踪任务

    输出:
    - 无

    作用:
    - 串联抓取、入库、聚类、分析、摘要与清理的全流程任务
    """

    try:
        logger.info(f"🚀 开始新一轮全流程任务 (generate_daily={generate_daily}, run_topic_task={run_topic_task})...")
        news_items = await crawler_service.fetch_all_sources()
        await crawler_service.save_raw_news(news_items)

        await cluster_service.execute_clustering()

        await auto_batch_analyze_new_news()

        await auto_generate_summaries_top_n()
        await auto_generate_summaries_categories_top_n()

        await auto_analyze_sentiment_top_n()

        if generate_daily:
            await report_service.generate_and_cache_global_report("daily")

        if run_topic_task:
            try:
                await topic_service.refresh_topics()
            except AIConfigurationError:
                raise
            except Exception as e:
                logger.error(f"❌ 专题刷新异常: {e}")
        else:
            logger.info("⏩ 跳过专题刷新 (未到配置的时间间隔)")

        await cleanup_old_data()

        logger.info("✅ 本轮全流程任务结束")
    except AIConfigurationError:
        raise
    except Exception as e:
        logger.error(f"❌ 任务执行异常: {e}")
    finally:
        try:
            report_service.clear_local_cache()
        except Exception:
            pass
        gc.collect()
        _malloc_trim()


async def scheduled_task() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 定时调度入口：按固定间隔运行全流程，并在特定时刻生成日报/周报/月报
    """

    logger.info("⏰ 定时任务调度器启动...")

    state = _load_scheduler_state()
    last_periodic_run = state.get("last_periodic_run", datetime.min)
    last_topic_run = state.get("last_topic_run", datetime.min)
    last_daily_final = state.get("last_daily_final", None)
    last_weekly_final = state.get("last_weekly_final", None)
    last_monthly_final = state.get("last_monthly_final", None)

    if last_periodic_run != datetime.min:
        logger.info(f"📅 恢复调度状态: 上次运行于 {last_periodic_run}")

    while True:
        try:
            rss_mb = _read_rss_mb()
            cgroup_mb = _read_cgroup_memory_mb()
            if rss_mb is not None or cgroup_mb is not None:
                logger.debug(f"📈 内存状态: rss={rss_mb or 0:.1f}MB, cgroup={cgroup_mb or 0:.1f}MB")

            if not await check_db_connection():
                logger.warning("⚠️ 数据库连接异常，定时任务暂停运行，等待恢复...")
                await asyncio.sleep(60)
                continue

            if not (settings.DATABASE_URL or "").strip():
                logger.warning("⚠️ 未配置 DATABASE_URL，定时任务暂停运行")
                await asyncio.sleep(60)
                continue
            now = datetime.now()

            interval_seconds = settings.SCHEDULE_INTERVAL_MINUTES * 60
            if (now - last_periodic_run).total_seconds() >= interval_seconds:
                topic_interval_seconds = settings.TOPIC_SCHEDULE_INTERVAL_HOURS * 3600
                should_run_topics = (now - last_topic_run).total_seconds() >= topic_interval_seconds
                result = await task_manager.start(
                    "pipeline",
                    lambda: run_pipeline_task(generate_daily=True, run_topic_task=should_run_topics),
                    progress="定时全流程执行中",
                )
                if result.get("status") != "success":
                    logger.warning(f"⚠️ 定时全流程未完成: {result}")
                    await asyncio.sleep(30)
                    continue

                last_periodic_run = datetime.now()
                if should_run_topics:
                    last_topic_run = datetime.now()

                _save_scheduler_state(
                    last_periodic_run,
                    last_topic_run,
                    last_daily_final,
                    last_weekly_final,
                    last_monthly_final,
                )
                logger.info("🔄 全流程任务完成，5秒后重启服务以释放内存...")
                schedule_restart(delay_seconds=5)
                return

            if now.hour == 23 and now.minute == 58:
                if last_daily_final != now.date():
                    logger.info("⏰ [Schedule] 触发每日最终报表 (23:58)...")
                    await report_service.generate_and_cache_global_report("daily")
                    last_daily_final = now.date()
                    _save_scheduler_state(last_periodic_run, last_topic_run, last_daily_final, last_weekly_final, last_monthly_final)
                    gc.collect()

            if now.weekday() == 6 and now.hour == 23 and now.minute == 55:
                if last_weekly_final != now.date():
                    logger.info("⏰ [Schedule] 触发每周最终报表 (周日 23:55)...")
                    await report_service.generate_and_cache_global_report("weekly")
                    last_weekly_final = now.date()
                    _save_scheduler_state(last_periodic_run, last_topic_run, last_daily_final, last_weekly_final, last_monthly_final)
                    gc.collect()

            tomorrow = now + timedelta(days=1)
            if tomorrow.day == 1 and now.hour == 23 and now.minute == 50:
                if last_monthly_final != now.date():
                    logger.info("⏰ [Schedule] 触发每月最终报表 (月末 23:50)...")
                    await report_service.generate_and_cache_global_report("monthly")
                    last_monthly_final = now.date()
                    _save_scheduler_state(last_periodic_run, last_topic_run, last_daily_final, last_weekly_final, last_monthly_final)
                    gc.collect()

        except AIConfigurationError as e:
            logger.error(f"🛑 配置错误: {e} 请检查 config.yaml 是否配置正确")
            logger.warning("⚠️ 系统将进入维护模式，每 5 分钟自动重启服务检查一次...")
            await asyncio.sleep(300)
            
            # 重新加载配置
            from app.core.config import reload_settings
            reload_settings()
            
            # 重新加载 AI 服务中的配置引用
            from app.services.ai_service import ai_service
            ai_service.reload_config()
            
            logger.info("🔄 配置已尝试重新加载")
            
            continue

        except Exception as e:
            logger.error(f"❌ 调度循环异常: {e}")

        await asyncio.sleep(30)


async def run_manual() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 手动触发一次抓取与分析任务，并补齐日报/周报/月报缓存
    """

    logger.info("🚀 手动任务开始...")
    try:
        items = await crawler_service.fetch_all_sources()
        await crawler_service.save_raw_news(items)
        await cluster_service.execute_clustering()
        await auto_generate_summaries_top_n()
        await auto_generate_summaries_categories_top_n()
        await auto_analyze_sentiment_top_n()

        await report_service.generate_and_cache_global_report("daily")
        await report_service.generate_and_cache_global_report("weekly")
        await report_service.generate_and_cache_global_report("monthly")

        logger.info("✅ 手动任务结束")
    finally:
        try:
            await topic_service.refresh_topics()
        except Exception as e:
            logger.error(f"❌ 专题刷新异常: {e}")
        
        gc.collect()


async def reanalyze_all_categories() -> Dict:
    """
    输入:
    - 无

    输出:
    - 任务状态与成功更新条数

    作用:
    - 对全量新闻逐条调用 AI 进行重新分析，用于修复历史分类或策略调整
    """

    logger.info("🔄 开始全量数据重新分析任务...")

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(News.id))
        all_ids = result.scalars().all()

        logger.info(f"   📊 待处理新闻总数: {len(all_ids)}")

        sem = asyncio.Semaphore(5)

        async def analyze_task(news_item):
            async with sem:
                try:
                    text = news_item.summary or news_item.content or ""
                    res = await ai_service.analyze_sentiment(news_item.title, text)
                    if res:
                        news_item.sentiment_score = res["score"]
                        news_item.sentiment_label = res["label"]
                        news_item.category = res.get("category", "其他")
                        news_item.keywords = res["keywords"]
                        news_item.entities = res["entities"]
                        return True
                except Exception as e:
                    logger.error(f"   ⚠️ 分析失败 ({news_item.title}): {e}")
                return False

        tasks = []
        batch_size = 50
        count = 0

        for i in range(0, len(all_ids), batch_size):
            batch_ids = all_ids[i : i + batch_size]

            async with AsyncSessionLocal() as db:
                result = await db.execute(select(News).options(defer(News.embedding)).where(News.id.in_(batch_ids)))
                current_batch_news = result.scalars().all()

                batch_tasks = [analyze_task(n) for n in current_batch_news]
                results = await asyncio.gather(*batch_tasks)

                try:
                    await db.commit()
                    success_count = sum(1 for r in results if r)
                    count += success_count
                    logger.debug(f"   处理批次 {i} - {i + batch_size}，成功: {success_count}")
                except Exception as e:
                    logger.error(f"   ❌ 批次提交失败: {e}")
                    await db.rollback()
                
                # 主动回收内存
                del current_batch_news
                del results
                gc.collect()

        logger.info(f"✅ 全量重分析完成，共更新 {count} 条")
        return {"status": "finished", "count": count}


async def background_analyze_all() -> None:
    """
    输入:
    - 无

    输出:
    - 无

    作用:
    - 以批处理方式补全历史新闻的情感与关键词，并在结束后刷新报表缓存
    """

    logger.info("🚀 开始全量情感分析任务（处理所有未分析的历史数据）...")

    total_processed = 0
    batch_size = 50

    while True:
        processed_in_batch = 0
        async with AsyncSessionLocal() as db:
            stmt = select(News).options(defer(News.embedding)).where(News.keywords.is_(None)).limit(batch_size)
            result = await db.execute(stmt)
            items = result.scalars().all()

            if not items:
                stmt = select(News).options(defer(News.embedding)).limit(batch_size * 5)
                result = await db.execute(stmt)
                all_candidates = result.scalars().all()
                items = [
                    n for n in all_candidates if not n.keywords or n.keywords == [] or n.keywords == "[]"
                ][:batch_size]

            if not items:
                logger.info("   ⚠️ 未发现更多待分析数据")
                break

            logger.info(f"   📦 本批次处理 {len(items)} 条...")

            sem = asyncio.Semaphore(10)

            async def analyze_task(news_item):
                async with sem:
                    try:
                        text = news_item.summary or news_item.content or ""
                        if len(text) < 10:
                            return {
                                "score": 50,
                                "label": "中立",
                                "category": "其他",
                                "keywords": ["无内容"],
                                "entities": [],
                            }
                        return await ai_service.analyze_sentiment(news_item.title, text)
                    except Exception as e:
                        logger.error(f"   ⚠️ 分析失败 ({news_item.id}): {e}")
                        return None

            tasks = [analyze_task(n) for n in items]
            results = await asyncio.gather(*tasks)

            for news, res in zip(items, results):
                if res:
                    news.sentiment_score = res["score"]
                    news.sentiment_label = res["label"]
                    news.category = res.get("category", "其他")
                    news.keywords = res["keywords"]
                    news.entities = res["entities"]
                    db.add(news)
                    processed_in_batch += 1
                else:
                    news.keywords = ["分析失败"]
                    db.add(news)

            await db.commit()
            
            # 主动回收内存
            del items
            del results
            gc.collect()
            total_processed += processed_in_batch
            logger.info(f"   ✅ 已更新 {processed_in_batch} 条，累计 {total_processed} 条")

        await asyncio.sleep(1)

    await report_service.generate_and_cache_global_report("daily")
    await report_service.generate_and_cache_global_report("weekly")
    await report_service.generate_and_cache_global_report("monthly")
    logger.info(f"🎉 全量分析任务结束，共处理 {total_processed} 条")
