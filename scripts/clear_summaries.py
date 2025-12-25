import asyncio
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.getcwd())

from sqlalchemy import text
from app.core.database import get_sessionmaker

async def clear_summaries():
    print("正在连接数据库...")
    try:
        session_maker = get_sessionmaker()
    except Exception as e:
        print(f"初始化数据库连接失败: {e}")
        return

    async with session_maker() as session:
        print("开始清理数据...")
        
        try:
            # 1. 清空新闻摘要
            print("清空 News.summary ...")
            await session.execute(text("UPDATE news SET summary = NULL"))
            
            # 2. 删除所有时间轴条目
            print("删除所有 TopicTimelineItem ...")
            await session.execute(text("DELETE FROM topic_timeline_items"))
            
            # 3. 删除所有专题
            print("删除所有 Topic ...")
            await session.execute(text("DELETE FROM topics"))
            
            await session.commit()
            print("清理完成！所有摘要、专题和时间轴数据已重置。")
        except Exception as e:
            print(f"清理过程中出错: {e}")
            await session.rollback()

if __name__ == "__main__":
    # 在 Windows 上运行 asyncio 可能需要这个策略设置
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(clear_summaries())
