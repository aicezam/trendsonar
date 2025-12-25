import asyncio
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sqlalchemy import text
from app.core.database import AsyncSessionLocal

async def reset_topics():
    """
    重置专题数据脚本
    作用：
    1. 删除所有专题 (topics 表)
    2. 级联删除所有时间轴条目 (topic_timeline_items 表)
    3. 重置相关的主键序列 (ID 从 1 开始)
    注意：不会删除 news 表中的原始新闻数据，这些新闻将变为“未归类”状态，可被再次聚合。
    """
    print("🗑️  开始清理专题历史数据...")
    async with AsyncSessionLocal() as db:
        try:
            # 1. 删除所有专题 (Cascade 会自动删除 timeline items)
            print("   - 正在删除所有专题记录...")
            await db.execute(text("DELETE FROM topics"))
            
            # 2. 重置自增 ID (PostgreSQL)
            try:
                print("   - 正在重置 ID 序列...")
                await db.execute(text("ALTER SEQUENCE topics_id_seq RESTART WITH 1"))
                await db.execute(text("ALTER SEQUENCE topic_timeline_items_id_seq RESTART WITH 1"))
            except Exception as e:
                print(f"   (ID 序列重置跳过或失败: {e})")

            await db.commit()
            print("✅ 专题数据已成功重置。")
            
        except Exception as e:
            await db.rollback()
            print(f"❌ 重置失败: {e}")

if __name__ == "__main__":
    # Windows 下 asyncio 策略调整
    # 移除 WindowsSelectorEventLoopPolicy 以支持 Playwright 子进程
    # if sys.platform == "win32":
    #     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(reset_topics())
