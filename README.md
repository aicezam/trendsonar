# TrendSonar

当前版本：**v0.2.8**

TrendSonar 是一个面向新闻热点聚合、事件去重、专题追踪、舆情报告和新闻智能体问答的 Web 工具。它会从配置的新闻源中持续抓取内容，结合 Embedding、OpenAI-compatible 大模型、Crawl4AI/Playwright 正文补抓与结构化分析能力，对新闻进行聚类、摘要、分类、情感分析、关键词实体提取、专题整理和报告生成。

项目适合用于搭建个人或团队内部的资讯观察台，例如跟踪行业动态、观察公共事件进展、沉淀关键词报告、生成每日热点简报，或通过智能体按自然语言检索本地新闻库。AI 分析结果依赖新闻源质量、模型能力、提示词和数据积累时间，建议作为辅助阅读与分析工具使用，重要结论仍应回看原文核验。

## 在线演示

体验地址：[https://ainews.izam.cn](https://ainews.izam.cn)

## 功能概览

### 新闻采集与处理

- 通过 `data/news_sources.json` 配置多个新闻源，支持启用状态、来源权重和地址维护。
- 兼容 RSS/XML、JSON 接口和部分网页类热点源。
- 支持通过管理后台卡片式新增、编辑、删除、测试新闻源。
- 记录新闻源健康状态，包括最近抓取结果、测试结果、失败次数和错误信息。
- 使用 Crawl4AI/Playwright 补抓正文，支持动态页面等待、超时、重试和并发控制。
- 支持微博 Cookie、忽略域名、关注关键词过滤等采集辅助配置。

### 热点列表与语义搜索

- 首页按热度或时间展示新闻，支持分页、时间范围、分类、地区和来源筛选。
- 支持 `today`、`24h`、`3d`、`7d`、`30d`、`week`、`month`、`year`、`all` 及自定义日期范围。
- 关键词搜索会优先使用向量召回，并结合文本匹配提升检索可用性。
- 新闻详情弹窗展示摘要、来源、关键词、实体、情感、关联报道和相似新闻。
- 支持生成热点新闻图片和智能体新闻卡片图片，便于分享或归档。

### AI 分析与聚类

- 自动为热点新闻生成 AI 摘要，并在正文不足时尽量使用来源摘要兜底。
- 自动补全分类、地区、情感倾向、关键词和实体。
- 使用 Embedding 相似度与 AI 核验对同一事件多来源报道进行去重聚合。
- 支持主力模型、备用模型和按功能配置的 AI 路由，例如摘要、情感、聚类、专题、报告、对话。
- 支持在管理后台测试 Embedding、主模型和备用模型连通性。

### 专题追踪

- 自动从近期高热新闻中发现候选事件簇，并通过 AI 审核生成专题。
- 支持专题列表、专题详情、时间轴、相关新闻和专题趋势仪表盘。
- 支持手动创建、改名、删除专题，并在后台扫描匹配相关新闻。
- 支持刷新专题综述、刷新单个时间轴节点摘要。
- 可通过配置调节专题召回池、候选簇数量、AI 审核批次、相似度阈值、质量等级、最低新闻数和来源数。

### 报告与图表

- 支持综合报告和关键词报告，可按时间、分类、地区、来源和样本数量筛选。
- 提供来源分布、词云、情感分布、正负面关键词、热度趋势、相关新闻和词项共现网络。
- 支持日报、周报、月报缓存，以及历史报告读取和删除。
- AI 报告支持流式输出，关键词深度报告会围绕事件演变、观点光谱、风险机会和后续观察生成 Markdown 内容。

### 新闻智能体

- 基于 PydanticAI 构建新闻智能体，支持连续对话和工具调用事件流。
- 可调用内置工具查询热点新闻、语义搜索新闻、读取新闻详情、查询专题、读取专题详情、获取报告数据、创建关键词报告、创建事件专题、分析词项趋势、网页搜索、网页正文抓取和生成新闻图片。
- 支持管理端新增自定义 HTTP 工具，配置参数、执行器、提示词提示和启用状态。
- 自定义工具支持 GET/POST、URL/Query/Header/Body 模板、结果路径截取和返回体压缩。
- 对网页抓取和自定义工具做基础 URL 安全校验，避免访问本机、内网或高风险元数据地址。

### 管理后台

- 后台地址：`/admin`。
- 支持登录态 Cookie，管理员密码通过 `.env` 或环境变量 `ADMIN_PASSWORD` 设置。
- 可在线维护运行配置、新闻源、提示词、后台任务、日志和智能体工具。
- 支持查看当天内存日志、历史日志文件、任务状态，以及手动触发抓取分析和历史数据补全。
- 修改 `config.yaml` 后会触发服务重启以重新加载配置。

## 技术栈

- Web 框架：FastAPI、Starlette、Jinja2
- 数据库：SQLAlchemy Async，默认 SQLite，可配置 PostgreSQL
- AI 接入：OpenAI-compatible API、`openai` SDK、`pydantic-ai`
- 向量能力：Embedding API，默认示例使用 SiliconFlow
- 爬虫能力：aiohttp、BeautifulSoup、Crawl4AI、Playwright
- 前端图表：ECharts
- 图片生成：Pillow
- 部署：Docker / Docker Compose

## 项目结构

```text
TrendSonar/
├── app/
│   ├── api/              # FastAPI API 路由
│   ├── core/             # 配置、数据库、日志、提示词默认值
│   ├── models/           # SQLAlchemy 数据模型
│   ├── services/         # 采集、聚类、报告、专题、智能体等业务服务
│   └── utils/            # 配置读写、搜索、图片、网页工具等通用能力
├── data/                 # 运行数据、新闻源、提示词和工具配置
├── docker/               # Docker 示例配置
├── docs/images/          # README 截图
├── static/               # 前端静态资源
├── templates/            # 页面模板
├── main.py               # 应用入口
├── config.yaml           # 主配置文件
└── requirements.txt      # Python 依赖
```

## 快速开始（Docker Compose）

推荐使用 Docker Compose 部署。启动前先准备运行配置和新闻源文件。

### 1. 准备目录

```bash
mkdir -p data
```

复制或参考仓库内示例文件：

- `docker/data/config.yaml` -> `data/config.yaml`
- `docker/data/news_sources.json` -> `data/news_sources.json`

至少需要配置：

- `DATABASE_URL`：数据库连接，默认可用 `sqlite+aiosqlite:///data/trendsonar.db`
- `SILICONFLOW_API_KEY` / `SILICONFLOW_BASE_URL` / `EMBEDDING_MODEL`：Embedding 配置
- `MAIN_AI_API_KEY` / `MAIN_AI_BASE_URL` / `MAIN_AI_MODEL`：主力生成模型
- `BACKUP_AI_API_KEY` / `BACKUP_AI_BASE_URL` / `BACKUP_AI_MODEL`：备用生成模型
- `ADMIN_PASSWORD`：管理员密码，建议通过环境变量设置

### 2. 创建 `docker-compose.yml`

```yaml
version: '3.8'

services:
  trendsonar:
    image: instarsea/trendsonar
    container_name: trendsonar
    restart: always
    ports:
      - "8193:8193"
    volumes:
      - ./data/config.yaml:/app/config.yaml
      - ./data:/app/data
    environment:
      - TZ=Asia/Shanghai
      - ADMIN_PASSWORD=your_secure_password
```

### 3. 启动服务

```bash
docker-compose up -d
```

启动后访问：

- 首页：`http://localhost:8193`
- 专题页：`http://localhost:8193/topics`
- 报告页：`http://localhost:8193/report`
- 管理后台：`http://localhost:8193/admin`

## Docker CLI 部署

```bash
docker run -d \
  --name trendsonar \
  -p 8193:8193 \
  -v /path/to/your/data/config.yaml:/app/config.yaml \
  -v /path/to/your/data:/app/data \
  -e TZ=Asia/Shanghai \
  -e ADMIN_PASSWORD=your_secure_password \
  instarsea/trendsonar
```

请将 `/path/to/your/data` 替换为实际数据目录。Windows 路径挂载时可使用类似 `D:/trendsonar/data:/app/data` 的形式。

## 本地源码运行

建议使用 Python 3.11。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Linux / Debian 环境可使用：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

正文抓取依赖 Playwright/Chromium。本地首次运行如遇浏览器依赖缺失，可执行：

```bash
python -m playwright install chromium
python -m playwright install-deps chromium
```

也可以通过命令行指定配置文件：

```bash
python main.py --config /path/to/config.yaml
```

## 配置说明

配置读取优先级为：初始化参数、`config.yaml`、环境变量、`.env`、文件密钥。默认配置文件为项目根目录 `config.yaml`，也可以通过环境变量 `TRENDSONAR_CONFIG` 或启动参数 `--config` 指定。

常用配置项：

| 配置项 | 说明 |
| --- | --- |
| `APP_NAME` | 页面标题和系统名称 |
| `PORT` | 服务端口，默认 `8193` |
| `LOG_LEVEL` / `LOG_RETENTION_DAYS` | 日志等级和日志文件保留天数 |
| `DATABASE_URL` | 数据库连接，支持 SQLite 和 PostgreSQL |
| `WEIBO_COOKIE` | 微博正文抓取所需 Cookie |
| `CRAWLER_CONCURRENCY` | 正文补抓并发数 |
| `CRAWLER_*` | 正文抓取等待、超时、重试和最小长度配置 |
| `SILICONFLOW_*` / `EMBEDDING_MODEL` | Embedding 服务配置 |
| `MAIN_AI_*` | 主力生成模型配置 |
| `BACKUP_AI_*` | 备用生成模型配置 |
| `AI_ROUTE` | 各功能模块使用 `main` 或 `backup` 模型 |
| `SCHEDULE_INTERVAL_MINUTES` | 自动全流程任务间隔 |
| `AUTO_SUMMARY_TOP_N` | 自动摘要新闻数量 |
| `AUTO_ANALYSIS_TOP_N` | 自动深度分析新闻数量 |
| `CLUSTERING_THRESHOLD` | 新闻聚类相似度阈值 |
| `FOLLOW_KEYWORDS` | 关注关键词，逗号分隔；为空则不过滤 |
| `FOLLOW_KEYWORDS_THRESHOLD` | 关注关键词向量相似度阈值 |
| `NEWS_CATEGORIES` | 新闻分类列表 |
| `IGNORED_DOMAINS` | 忽略域名列表 |
| `DATA_CLEANUP_*` | 低热历史新闻自动清理配置 |
| `TOPIC_*` | 专题生成、匹配、更新时间窗和质量控制 |
| `TOPIC_DISCOVERY_*` | v0.2.8 专题候选簇发现和 AI 批量审核参数 |

`AI_ROUTE` 示例：

```yaml
AI_ROUTE:
  SUMMARY: "main"
  SENTIMENT: "backup"
  KEYWORDS: "backup"
  CLUSTERING: "backup"
  TOPIC_NAME: "backup"
  TOPIC_EVAL: "backup"
  TOPIC_MATCH: "backup"
  TOPIC_TIMELINE: "backup"
  TOPIC_OVERVIEW: "backup"
  TOPIC_INITIAL_SUMMARY: "main"
  REPORT: "backup"
  CHAT: "backup"
```

## 新闻源配置

新闻源位于 `data/news_sources.json`，基础结构如下：

```json
[
  {
    "name": "来源名称",
    "weight": 1.0,
    "address": "https://example.com/rss-or-api",
    "enabled": true
  }
]
```

字段说明：

- `name`：来源名称，会显示在列表和报告中。
- `weight`：来源权重，会影响热度计算。
- `address`：RSS、XML、JSON 或可解析的新闻接口地址。
- `enabled`：是否启用该来源。

管理后台的新闻源测试不会写入数据库，适合在正式保存前检查抓取结果和正文补抓效果。

## 常用 API

| 接口 | 说明 |
| --- | --- |
| `GET /api/app_info` | 应用名称和版本 |
| `GET /api/news` | 新闻列表、筛选和语义搜索 |
| `GET /api/news/top` | 热点新闻 TopN |
| `GET /api/news/{news_id}` | 新闻详情 |
| `GET /api/news/{news_id}/similar` | 相似新闻 |
| `POST /api/generate_summary/{news_id}` | 为单条新闻生成摘要 |
| `GET /api/news_image` | 生成热点新闻图片 |
| `GET /api/topics/list` | 专题列表 |
| `GET /api/topics/{topic_id}` | 专题详情 |
| `GET /api/topics/{topic_id}/trends` | 专题趋势数据 |
| `POST /api/report/generate` | 生成报告 |
| `GET /api/report/analysis` | 报告分析数据 |
| `GET /api/report/term-analysis` | 词项分析 |
| `GET /api/chat` | 基于新闻库的 RAG 问答 |
| `GET /api/agent/chat` | 智能体工具调用问答流 |
| `POST /api/trigger_crawl` | 管理员手动触发全流程 |

管理端接口需要登录或管理员鉴权，包括配置读写、新闻源维护、日志查看、任务状态、AI 连通性测试、智能体工具维护等。

## 自动任务

应用启动后会初始化数据库并启动定时任务。默认流程包括：

1. 抓取全部启用新闻源。
2. 保存新新闻并更新新闻源健康状态。
3. 对近期开窗新闻进行聚类去重。
4. 批量补全分类、地区、情感、关键词和实体。
5. 为热点新闻生成 AI 摘要。
6. 生成或刷新日报缓存。
7. 按专题间隔刷新专题。
8. 按配置清理低热历史数据。

此外，调度器会在特定时间生成每日、每周和每月最终报告缓存。全流程任务完成后，服务会按当前逻辑尝试重启以释放内存。

## 使用建议

- 初始运行时数据量少，聚类、专题和报告效果会比较有限，建议运行一段时间后再评估质量。
- 新闻源质量直接影响结果。若某来源经常失败，可在后台查看健康状态并单独测试。
- 动态页面正文补抓会占用更多内存，低配机器建议将 `CRAWLER_CONCURRENCY` 控制在 `1-2`。
- 聚类阈值过低可能误合并，过高可能漏合并；专题质量等级越高，生成数量越少但更稳。
- 关键词报告和智能体问答只基于已入库新闻与可调用工具，不代表完整互联网信息。
- 涉及法律、医疗、投资、公共安全等高风险判断时，请以原文和权威来源为准。

## 推荐新闻源

如果需要扩展 RSS 或热点来源，可以参考：

- [Hot News](https://github.com/orz-ai/hot_news)：每日热点新闻聚合。
- [NewsNow](https://github.com/ourongxing/newsnow)：多平台热榜聚合，提供部分 RSS/API 接口。
- [RSSHub](https://github.com/DIYgod/RSSHub)：为许多网站生成 RSS。
- [AnyFeeder](https://plink.anyfeeder.com/)：RSS 源聚合服务。

## 界面预览

### 热点新闻列表

<img src="docs/images/index.png" alt="热点新闻列表" width="100%">

### 专题追踪

<img src="docs/images/topic.png" alt="专题追踪" width="100%">

### 深度报告

<img src="docs/images/baobiao1.png" alt="报告预览1" width="100%">
<img src="docs/images/baobiao2.png" alt="报告预览2" width="100%">

## 更新日志

- **v0.2.8**：补强新闻智能体能力，新增网页搜索、网页正文抓取、新闻图片生成和管理端自定义 HTTP 工具；优化新闻源卡片管理、健康状态展示、AI 连通性测试、日志查看、专题候选簇发现与 AI 批量审核参数；强化报告词项分析、专题趋势和新闻详情体验。
- **v0.2.7**：优化新闻详情、相似新闻召回、报告交互和管理端配置体验。
- **v0.2.6**：重构 UI 视觉样式，优化性能、提示词和专题生成逻辑。
- **v0.2.5**：优化搜索与向量召回，专题模块新增关键词趋势分析，报告页新增关键词分析，优化 UI 交互和新闻详情弹窗。
- **v0.2.1**：专题模块新增报告能力，包括词云、来源分布、情感分析、相关新闻和关键词趋势等。
- **v0.2.0**：优化 token 消耗、日志显示和专题重复生成问题。
- **v0.1.7**：优化 token 消耗、聚合流程和首页筛选交互。
- **v0.1.6**：优化关键词深度分析交互，支持在管理后台自定义提示词。
- **v0.1.5**：优化专题生成逻辑，支持手动新增、编辑、删除专题。
- **v0.1.4**：新增专题质量审核等级配置，优化专题生成逻辑。
- **v0.1.3**：优化内存占用和专题追踪复核逻辑。
- **v0.1.2**：优化配置异常时的定时任务流程。
- **v0.1.1**：修复部分鉴权问题。
- **v0.1.0**：初始版本，发布至 Docker Hub。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=aicezam/trendsonar&type=date&legend=top-left)](https://www.star-history.com/#aicezam/trendsonar&type=date&legend=top-left)

## 许可证

本项目基于 [MIT License](LICENSE) 开源。
