// 本文件驱动知识图谱页面：基于 Sigma v3 渲染 + graphology ForceAtlas2(Web Worker)力导向布局。
// 设计要点：
// - 节点屏幕尺寸恒定(zoomToSizeRatioFunction 恒为 1)，缩放只改变节点间距，露出更多邻居与标签。
// - 力导向布局跑在 Worker，主线程不阻塞，避免卡顿。
// - 选中走 reducer 高亮邻域，不再使用影子光晕节点。
(function () {
    "use strict";

    const SigmaCtor = window.Sigma && (window.Sigma.Sigma || window.Sigma.default || window.Sigma);
    const GraphCtor = window.graphology && (window.graphology.Graph || window.graphology.default || window.graphology);
    const Lib = window.graphologyLibrary || {};

    const state = {
        graph: null,
        renderer: null,
        fa2: null,
        fa2Timer: 0,
        pinRaf: 0,
        autoExpandTimer: 0,
        range: "24h",
        selectedTerm: "",
        loading: false,
        detailLoading: false,
        expanded: new Set(),
        expandQueue: new Set(),
        newsPager: {
            term: "",
            page: 1,
            pageSize: 10,
            hasMore: false,
            loading: false,
            observer: null
        },
        dragged: null,
        hovered: "",
        highlightNodes: null,
        highlightEdges: null
    };

    const els = {
        canvas: document.getElementById("graphCanvas"),
        empty: document.getElementById("graphEmpty"),
        status: document.getElementById("graphStatus"),
        miniStats: document.getElementById("graphMiniStats"),
        rangeTabs: document.getElementById("graphRangeTabs"),
        refreshBtn: document.getElementById("graphRefreshBtn"),
        resetBtn: document.getElementById("graphResetBtn"),
        category: document.getElementById("graphCategoryFilter"),
        region: document.getElementById("graphRegionFilter"),
        source: document.getElementById("graphSourceFilter"),
        search: document.getElementById("graphSearchInput"),
        searchBtn: document.getElementById("graphSearchBtn"),
        sidePanel: document.getElementById("graphSidePanel"),
        sideEmpty: document.getElementById("graphSideEmpty"),
        sideContent: document.getElementById("graphSideContent"),
        nodeTitle: document.getElementById("graphNodeTitle"),
        nodeMetrics: document.getElementById("graphNodeMetrics"),
        trendBars: document.getElementById("graphTrendBars"),
        neighbors: document.getElementById("graphNeighborList"),
        newsList: document.getElementById("graphNewsList"),
        expandBtn: document.getElementById("graphExpandBtn"),
        closeDetailBtn: document.getElementById("graphCloseDetailBtn"),
        detailModal: document.getElementById("newsDetailModal"),
        detailCloseBtn: document.getElementById("detailCloseBtn"),
        detailTitle: document.getElementById("detailTitle"),
        detailBody: document.getElementById("detailBody")
    };

    function escapeHtml(value) {
        return String(value || "")
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    function setStatus(text, loading) {
        if (!els.status) return;
        els.status.textContent = text;
        els.status.classList.toggle("loading", !!loading);
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function isDarkTheme() {
        return document.documentElement.getAttribute("data-theme") === "dark";
    }

    // 将后端 color 映射为更深的标签描边色，保证标签在浅色画布上可读。
    function labelColor(color) {
        const map = {
            "#0f172a": "#0f172a",
            "#7c3aed": "#6d28d9",
            "#2563eb": "#1d4ed8",
            "#0891b2": "#0e7490",
            "#10b981": "#047857",
            "#ef4444": "#dc2626",
            "#64748b": "#334155"
        };
        return map[String(color || "").toLowerCase()] || "#334155";
    }

    // 把任意颜色(hex / rgb / rgba)转换为带指定透明度的 rgba。
    // 用于“淡化”节点：保留其本来的色相，只降低不透明度，而不是统一刷成灰白。
    function withAlpha(color, alpha) {
        const c = String(color || "").trim();
        let m = c.match(/^#([0-9a-fA-F]{6})$/);
        if (m) {
            const n = parseInt(m[1], 16);
            return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
        }
        m = c.match(/^#([0-9a-fA-F]{3})$/);
        if (m) {
            const r = parseInt(m[1][0] + m[1][0], 16);
            const g = parseInt(m[1][1] + m[1][1], 16);
            const b = parseInt(m[1][2] + m[1][2], 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        m = c.match(/rgba?\(([^)]+)\)/i);
        if (m) {
            const parts = m[1].split(",").map(s => s.trim());
            return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
        }
        return c;
    }

    function relationColor(type, alpha) {
        const opacity = clamp(Number(alpha || 0.2), 0.05, 0.95);
        const map = {
            entity_cooccurrence: [37, 99, 235],
            entity_topic: [124, 58, 237],
            risk: [239, 68, 68],
            positive: [16, 185, 129],
            cooccurrence: [100, 116, 139]
        };
        const rgb = map[String(type || "cooccurrence")] || map.cooccurrence;
        return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${opacity})`;
    }

    // 默认(未选中)状态下边的颜色：保留关系色相，加适度透明度，
    // 浅色模式略实、深色模式略透，既能看清又不喧宾夺主。
    function defaultEdgeAlpha() {
        return isDarkTheme() ? 0.3 : 0.42;
    }

    // 选中邻域之外的边：进一步淡化但仍保留关系色相，不直接隐藏也不刷白。
    function dimEdgeAlpha() {
        return isDarkTheme() ? 0.16 : 0.24;
    }

    function currentFilters() {
        const params = new URLSearchParams();
        params.set("range", state.range);
        if (els.category.value) params.set("category", els.category.value);
        if (els.region.value) params.set("region", els.region.value);
        if (els.source.value) params.set("source", els.source.value);
        return params;
    }

    async function fetchJson(url) {
        const resp = await fetch(url, { cache: "no-store" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    }

    function sentimentClass(label) {
        const v = String(label || "").trim();
        if (v === "正面") return "sent-pos";
        if (v === "负面") return "sent-neg";
        return "sent-neu";
    }

    function safeExternalUrl(url) {
        const raw = typeof url === "string" ? url.trim() : "";
        if (!raw || raw === "#" || raw.startsWith("#")) return "";
        const candidate = raw.startsWith("www.") ? `https://${raw}` : raw;
        try {
            const parsed = new URL(candidate, window.location.origin);
            return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "";
        } catch (e) {
            return "";
        }
    }

    function renderDetailSourceLink(source, url) {
        const label = escapeHtml(source || "未知来源");
        return url
            ? `<a class="detail-source-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
            : `<span>${label}</span>`;
    }

    function renderDetailTitleLink(title, url) {
        const label = escapeHtml(title || "新闻详情");
        return url
            ? `<a class="news-detail-title-link" href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">${label}</a>`
            : label;
    }

    function renderDetailTags(items, emptyText) {
        const values = (items || []).filter(x => typeof x === "string" && x.trim()).slice(0, 16);
        if (!values.length) return `<div class="detail-empty">${emptyText}</div>`;
        return values.map(x => `<button type="button" class="detail-chip detail-chip-btn" data-term="${escapeHtml(x)}">${escapeHtml(x)}</button>`).join("");
    }

    function renderRelatedSources(items) {
        if (!items || !items.length) return `<div class="detail-empty">暂无关联报道</div>`;
        return items.map((item, idx) => `
            <div class="detail-source-row">
                <div class="detail-source-index">${idx + 1}</div>
                <div class="detail-source-main">
                    <div class="detail-source-name">${escapeHtml(item.name || "未知来源")}</div>
                    <a href="${escapeHtml(item.url || "#")}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title || item.url || "查看原文")}</a>
                </div>
            </div>
        `).join("");
    }

    function renderRelatedTopics(items) {
        if (!items || !items.length) return `<div class="detail-empty">暂未关联专题</div>`;
        return items.map(item => `
            <a class="detail-topic-link" href="/topics/${Number(item.id || 0)}">
                <div class="detail-topic-name">${escapeHtml(item.name || "未命名专题")}</div>
                <div class="detail-topic-meta">热度 ${Number(item.heat_score || 0).toFixed(1)} · ${item.updated_time ? new Date(item.updated_time).toLocaleString("zh-CN") : "-"}</div>
                ${item.matched_timeline && item.matched_timeline.content ? `<div class="detail-topic-snippet">${escapeHtml(item.matched_timeline.content)}</div>` : ""}
            </a>
        `).join("");
    }

    function renderDetailNewsList(items, emptyText) {
        if (!items || !items.length) return `<div class="detail-empty">${emptyText}</div>`;
        return items.map(item => {
            const time = item.time ? new Date(item.time).toLocaleString("zh-CN") : "-";
            const itemId = Number(item.id || 0);
            const titleHtml = itemId > 0
                ? `<button class="detail-inline-link" type="button" onclick="openNewsDetail(${itemId}, { replace: true })">${escapeHtml(item.title || "(无标题)")}</button>`
                : `<a href="${escapeHtml(item.url || "#")}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title || "(无标题)")}</a>`;
            return `
                <div class="detail-list-item">
                    ${titleHtml}
                    <div class="detail-item-meta">
                        <span>${escapeHtml(item.source || "未知来源")}</span>
                        <span>${time}</span>
                        <span>热度 ${Number(item.heat || 0).toFixed(1)}</span>
                    </div>
                </div>
            `;
        }).join("");
    }

    function closeNewsDetail() {
        if (!els.detailModal) return;
        els.detailModal.classList.remove("show");
        els.detailModal.setAttribute("aria-hidden", "true");
        document.body.classList.remove("modal-open");
    }

    function renderNewsDetail(data) {
        const news = data.news || {};
        const status = data.content_status || {};
        const time = news.time ? new Date(news.time).toLocaleString("zh-CN") : "-";
        const newsUrl = safeExternalUrl(news.url);
        const sourceHtml = renderDetailSourceLink(news.source, newsUrl);
        const summary = news.summary
            ? (window.marked ? window.marked.parse(news.summary) : escapeHtml(news.summary))
            : `<div class="detail-empty">暂无摘要，可在新闻卡片中生成摘要。</div>`;
        const topicSection = data.topics && data.topics.length ? `
            <div class="detail-section">
                <div class="detail-section-title">相关专题</div>
                <div class="detail-topic-list">${renderRelatedTopics(data.topics)}</div>
            </div>
        ` : "";
        const sourceSection = data.related_sources && data.related_sources.length ? `
            <div class="detail-section">
                <div class="detail-section-title">关联报道来源</div>
                <div class="detail-source-list">${renderRelatedSources(data.related_sources)}</div>
            </div>
        ` : "";

        els.detailTitle.innerHTML = renderDetailTitleLink(news.title, newsUrl);
        els.detailBody.innerHTML = `
            <div class="detail-meta-line">
                ${sourceHtml}
                <span>${time}</span>
                <span class="meta-heat-val">热度 ${Number(news.heat || 0).toFixed(1)}</span>
                <span class="meta-sent-val ${sentimentClass(news.sentiment_label)}">${escapeHtml(news.sentiment_label || "未分析")}${news.sentiment_score !== undefined ? "：" + news.sentiment_score : ""}</span>
            </div>
            <div class="detail-status-grid">
                <div><strong>${status.has_summary ? "已有摘要" : "暂无摘要"}</strong><span>新闻摘要</span></div>
                <div><strong>${status.related_source_count || 0}</strong><span>关联报道</span></div>
            </div>
            <div class="detail-section">
                <div class="detail-section-title">
                    <span>新闻摘要</span>
                    <button class="btn-link" onclick="genSummary(${Number(news.id || 0)})">重新生成</button>
                </div>
                <div class="detail-summary markdown-body" id="summary-${Number(news.id || 0)}">${summary}</div>
            </div>
            <div class="detail-two-col">
                <div class="detail-section">
                    <div class="detail-section-title">关键词</div>
                    <div class="detail-chip-row">${renderDetailTags(news.keywords, "暂无关键词")}</div>
                </div>
                <div class="detail-section">
                    <div class="detail-section-title">实体</div>
                    <div class="detail-chip-row">${renderDetailTags(news.entities, "暂无实体")}</div>
                </div>
            </div>
            ${topicSection}
            <div class="detail-section" id="similarNewsSection">
                <div class="detail-section-title">可能相似的报道</div>
                <div id="similarNewsList" class="detail-list">
                    <div class="detail-empty">查询中……</div>
                </div>
            </div>
            ${sourceSection}
        `;
        els.detailBody.querySelectorAll("[data-term]").forEach(el => {
            el.addEventListener("click", () => {
                const term = el.dataset.term || el.textContent;
                closeNewsDetail();
                if (state.graph && state.graph.hasNode(term)) selectNode(term);
                else expandNode(term).then(() => selectNode(term));
            });
        });
    }

    async function loadSimilarNews(id) {
        const listEl = document.getElementById("similarNewsList");
        if (!listEl) return;
        try {
            const data = await fetchJson(`/api/news/${id}/similar`);
            listEl.innerHTML = renderDetailNewsList(data.data || [], "暂无相似报道");
        } catch (e) {
            listEl.innerHTML = `<div class="detail-empty">相似报道加载失败：${escapeHtml(e.message)}</div>`;
        }
    }

    window.openNewsDetail = async function (id, options) {
        const newsId = Number(id || 0);
        if (!newsId || state.detailLoading || !els.detailModal) return;
        if (options && options.replace) closeNewsDetail();
        state.detailLoading = true;
        els.detailTitle.textContent = "新闻详情";
        els.detailBody.innerHTML = `<div class="detail-loading">加载中...</div>`;
        els.detailModal.classList.add("show");
        els.detailModal.setAttribute("aria-hidden", "false");
        document.body.classList.add("modal-open");
        try {
            const data = await fetchJson(`/api/news/${newsId}`);
            renderNewsDetail(data);
            loadSimilarNews(newsId);
        } catch (e) {
            els.detailBody.innerHTML = `<div class="detail-empty">加载失败：${escapeHtml(e.message)}</div>`;
        } finally {
            state.detailLoading = false;
        }
    };

    window.genSummary = async function (id) {
        const newsId = Number(id || 0);
        const box = document.getElementById(`summary-${newsId}`);
        if (!newsId || !box) return;
        const original = box.innerHTML;
        box.innerHTML = `<span class="summary-loading">正在生成摘要...</span>`;
        try {
            const resp = await fetch(`/api/generate_summary/${newsId}`, { method: "POST", cache: "no-store" });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            if (!data.summary) throw new Error("摘要为空");
            box.innerHTML = window.marked ? window.marked.parse(data.summary) : escapeHtml(data.summary);
        } catch (e) {
            box.innerHTML = original;
            window.alert(`生成失败：${e.message}`);
        }
    };

    async function loadFilters() {
        const targets = [
            ["/api/categories", els.category, "全部分类"],
            ["/api/regions", els.region, "全部地区"],
            ["/api/sources", els.source, "全部来源"]
        ];
        await Promise.all(targets.map(async ([url, select, emptyText]) => {
            if (!select) return;
            try {
                const data = await fetchJson(url);
                const values = Array.isArray(data) ? data : [];
                select.innerHTML = `<option value="">${emptyText}</option>` + values
                    .map(item => `<option value="${escapeHtml(item)}">${escapeHtml(item)}</option>`)
                    .join("");
            } catch (e) {
                select.innerHTML = `<option value="">${emptyText}</option>`;
            }
        }));
    }

    // 将后端节点尺寸压缩到稳定的屏幕像素范围(因节点尺寸恒定，这里就是最终半径基准)。
    function visualSize(node) {
        const raw = Number(node.size || 5);
        const role = String(node.role || "node").toLowerCase();
        const boost = role === "center" ? 6 : role === "hub" ? 2.4 : 0;
        return Number(clamp(3.4 + raw * 0.52 + boost, 4, 16).toFixed(2));
    }

    function nodeAttributes(node, layer) {
        const color = String(node.color || "#0891b2");
        return {
            label: node.label || node.term || node.id,
            term: node.term || node.id,
            term_type: node.term_type || node.type || "keyword",
            sentiment_group: node.sentiment_group || "neutral",
            role: String(node.role || "node").toLowerCase(),
            layer: layer || "base",
            x: Number(node.x || 0),
            y: Number(node.y || 0),
            size: visualSize(node),
            color,
            labelColor: labelColor(color),
            weight: Number(node.weight || 0),
            news_count: Number(node.news_count || 0),
            heat_score: Number(node.heat_score || 0),
            sentiment_avg: Number(node.sentiment_avg || 0),
            last_seen_at: node.last_seen_at || ""
        };
    }

    function initGraph() {
        if (!SigmaCtor || !GraphCtor || !els.canvas) {
            setStatus("图谱运行库未加载", false);
            return false;
        }
        if (state.fa2) { state.fa2.kill(); state.fa2 = null; }
        if (state.renderer) state.renderer.kill();
        state.graph = new GraphCtor({ type: "undirected", multi: false, allowSelfLoops: false });
        state.renderer = new SigmaCtor(state.graph, els.canvas, {
            // —— 关键：节点屏幕尺寸恒定，缩放只改变节点间距 ——
            zoomToSizeRatioFunction: () => 1,
            itemSizesReference: "screen",
            minCameraRatio: 0.05,
            maxCameraRatio: 8,
            // 标签按屏幕密度显示：放大后单位屏幕内节点变少，自然露出更多标签
            labelDensity: 0.7,
            labelGridCellSize: 64,
            labelRenderedSizeThreshold: 5,
            labelFont: "Microsoft YaHei, PingFang SC, Arial, sans-serif",
            labelWeight: "700",
            labelSize: 12,
            labelColor: { attribute: "labelColor" },
            defaultEdgeColor: "rgba(100, 116, 139, 0.42)",
            renderEdgeLabels: true,
            edgeLabelFont: "Microsoft YaHei, PingFang SC, Arial, sans-serif",
            edgeLabelSize: 10,
            edgeLabelColor: { color: "#64748b" },
            enableEdgeEvents: false,
            zIndex: true,
            nodeReducer,
            edgeReducer
        });
        bindSigmaEvents();
        return true;
    }

    // —— ForceAtlas2 力导向布局(Web Worker) ——
    function runLayout(durationMs) {
        if (!state.graph || state.graph.order === 0 || !Lib.FA2Layout) return;
        stopLayout();
        if (state.fa2) { state.fa2.kill(); state.fa2 = null; }
        let settings = {};
        try {
            settings = Lib.layoutForceAtlas2 ? Lib.layoutForceAtlas2.inferSettings(state.graph) : {};
        } catch (e) {
            settings = {};
        }
        settings.adjustSizes = true;          // 按节点尺寸防重叠
        settings.barnesHutOptimize = state.graph.order > 180;
        settings.gravity = 0.6;
        settings.scalingRatio = 14;
        settings.slowDown = 1 + Math.log(Math.max(2, state.graph.order));
        try {
            state.fa2 = new Lib.FA2Layout(state.graph, { settings });
            state.fa2.start();
            window.clearTimeout(state.fa2Timer);
            state.fa2Timer = window.setTimeout(stopLayout, durationMs || 2200);
        } catch (e) {
            state.fa2 = null;
        }
    }

    function stopLayout() {
        window.clearTimeout(state.fa2Timer);
        state.fa2Timer = 0;
        if (state.fa2 && state.fa2.isRunning && state.fa2.isRunning()) state.fa2.stop();
        stopPin();
    }

    // —— 展开时的钉住式布局 ——
    // worker 版 FA2 异步写坐标，会和“钉住”逻辑抢着改坐标导致漂移。
    // 展开时改用同步 layoutForceAtlas2.assign 分批跑：每批跑几次迭代，
    // 跑完立刻把已有节点坐标写回(钉住)，只让新节点被力推开。用 rAF 串起多批以获得平滑动画。
    function runPinnedLayout(pinned, newNodes) {
        const graph = state.graph;
        const renderer = state.renderer;
        if (!graph || !renderer) return;
        const assign = Lib.layoutForceAtlas2 && Lib.layoutForceAtlas2.assign;
        if (!assign) { renderer.refresh(); return; }
        stopLayout();
        let settings = {};
        try {
            settings = Lib.layoutForceAtlas2.inferSettings(graph);
        } catch (e) { settings = {}; }
        settings.adjustSizes = true;
        settings.gravity = 0.5;
        settings.scalingRatio = 16;
        settings.barnesHutOptimize = graph.order > 180;

        const restore = () => {
            pinned.forEach((pos, id) => {
                if (!graph.hasNode(id)) return;
                graph.setNodeAttribute(id, "x", pos.x);
                graph.setNodeAttribute(id, "y", pos.y);
            });
        };

        const totalBatches = 24;
        const iterPerBatch = 6;
        let batch = 0;
        function step() {
            try {
                assign(graph, { iterations: iterPerBatch, settings });
            } catch (e) { /* 布局失败则停止 */ batch = totalBatches; }
            restore(); // 每批结束钉住已有节点，确保它们坐标不动
            renderer.refresh({ skipIndexation: true });
            batch += 1;
            if (batch < totalBatches) {
                state.pinRaf = window.requestAnimationFrame(step);
            } else {
                state.pinRaf = 0;
            }
        }
        state.pinRaf = window.requestAnimationFrame(step);
    }

    function stopPin() {
        if (state.pinRaf) {
            window.cancelAnimationFrame(state.pinRaf);
            state.pinRaf = 0;
        }
    }

    // 节点入场补间：新节点从极小尺寸平滑放大到目标尺寸，避免“突然出现”。
    function tweenAppear(nodeIds, duration) {
        const graph = state.graph;
        const renderer = state.renderer;
        if (!graph || !renderer || !nodeIds.length) return;
        const targets = [];
        nodeIds.forEach(id => {
            if (!graph.hasNode(id)) return;
            const target = Number(graph.getNodeAttribute(id, "size") || 6);
            graph.setNodeAttribute(id, "size", Math.max(0.6, target * 0.12));
            targets.push({ id, target });
        });
        if (!targets.length) return;
        const dur = duration || 360;
        const start = performance.now();
        function step(now) {
            const t = clamp((now - start) / dur, 0, 1);
            const ease = 1 - Math.pow(1 - t, 3); // easeOutCubic
            targets.forEach(({ id, target }) => {
                if (!graph.hasNode(id)) return;
                const min = Math.max(0.6, target * 0.12);
                graph.setNodeAttribute(id, "size", min + (target - min) * ease);
            });
            renderer.refresh({ skipIndexation: true });
            if (t < 1) window.requestAnimationFrame(step);
        }
        window.requestAnimationFrame(step);
    }

    function mergeGraphData(payload, options) {
        if (!state.graph) return;
        const graph = state.graph;
        const center = options && options.center;
        const layer = options && options.layer ? options.layer : "base";
        const centerAttrs = center && graph.hasNode(center) ? graph.getNodeAttributes(center) : null;
        const newNodes = [];
        // 展开时记录所有已存在节点的当前坐标，稍后在布局期间钉住，确保它们不被 FA2 推动。
        const pinned = center ? new Map() : null;
        if (pinned) {
            graph.forEachNode(id => {
                pinned.set(id, { x: graph.getNodeAttribute(id, "x"), y: graph.getNodeAttribute(id, "y") });
            });
        }

        const incoming = payload.nodes || [];
        // 统计本次将要新增的邻居数量，用于把它们均匀排布成一圈(bloom 散开)。
        let pendingNew = 0;
        incoming.forEach(node => {
            const id = String(node.id || node.term || node.label || "");
            if (id && id !== center && !graph.hasNode(id)) pendingNew += 1;
        });
        const ringRadius = 36 + pendingNew * 1.4;
        let ringIndex = 0;

        incoming.forEach(node => {
            const id = String(node.id || node.term || node.label || "");
            if (!id) return;
            const exists = graph.hasNode(id);
            const attrs = nodeAttributes(node, exists && graph.getNodeAttribute(id, "layer") === "base" ? "base" : layer);
            if (exists) {
                // 已存在的节点保留当前位置，避免展开时全图跳动
                attrs.x = graph.getNodeAttribute(id, "x");
                attrs.y = graph.getNodeAttribute(id, "y");
                graph.mergeNodeAttributes(id, attrs);
            } else {
                if (centerAttrs && id !== center) {
                    // 新邻居围绕中心点排成一圈再交给 FA2 扩散，避免在远处突然冒出。
                    const angle = (ringIndex / Math.max(1, pendingNew)) * Math.PI * 2;
                    ringIndex += 1;
                    const jitter = 0.85 + Math.random() * 0.3;
                    attrs.x = centerAttrs.x + Math.cos(angle) * ringRadius * jitter;
                    attrs.y = centerAttrs.y + Math.sin(angle) * ringRadius * jitter;
                }
                graph.addNode(id, attrs);
                newNodes.push(id);
            }
        });

        (payload.edges || payload.links || []).forEach(edge => {
            const source = String(edge.source || "");
            const target = String(edge.target || "");
            if (!source || !target || source === target) return;
            if (!graph.hasNode(source) || !graph.hasNode(target)) return;
            const key = edge.id || [source, target].sort((a, b) => a.localeCompare(b)).join("__");
            const attrs = {
                size: clamp(Number(edge.size || 0.8), 0.5, 4),
                weight: Number(edge.weight || edge.value || 1),
                label: edge.relation_label || edge.label || "共同出现",
                relation_type: edge.relation_type || "cooccurrence",
                color: relationColor(edge.relation_type || "cooccurrence", defaultEdgeAlpha()),
                layer: center ? "expanded" : "base"
            };
            // 无向图中同一对节点只允许一条边；不同接口可能为同一对节点生成不同 key，
            // 因此按端点判断是否已存在，避免 addEdgeWithKey 抛出重复边异常。
            if (graph.hasEdge(source, target)) graph.mergeEdgeAttributes(graph.edge(source, target), attrs);
            else graph.addEdgeWithKey(key, source, target, attrs);
        });

        updateMiniStats(payload.summary || {});
        els.empty.classList.toggle("is-hidden", graph.order > 0);
        if (pinned && pinned.size) {
            // 展开：钉住已有节点(含中心点)，只让新邻居在周围散开。
            runPinnedLayout(pinned, newNodes);
            if (newNodes.length) tweenAppear(newNodes, 380);
        } else {
            // 首屏/刷新：正常跑 worker 力导向铺开全图。
            runLayout(2400);
        }
    }

    function updateMiniStats(summary) {
        if (!els.miniStats || !state.graph) return;
        const cached = summary.cached ? "<span>缓存</span>" : "";
        els.miniStats.innerHTML = `
            <span>节点 ${state.graph.order}</span>
            <span>关系 ${state.graph.size}</span>
            <span>样本 ${Number(summary.news_scanned || 0)}</span>
            ${cached}
        `;
    }

    // —— 选中节点时计算高亮邻域 ——
    function computeHighlight(term) {
        if (!state.graph || !state.graph.hasNode(term)) {
            state.highlightNodes = null;
            state.highlightEdges = null;
            return;
        }
        const nodes = new Set([term]);
        const edges = new Set();
        state.graph.forEachEdge(term, (edge, _attrs, src, tgt) => {
            nodes.add(src);
            nodes.add(tgt);
            edges.add(edge);
        });
        state.highlightNodes = nodes;
        state.highlightEdges = edges;
    }

    function clearHighlight() {
        state.highlightNodes = null;
        state.highlightEdges = null;
        if (state.renderer) state.renderer.refresh();
    }

    function nodeReducer(node, attrs) {
        const res = { ...attrs };
        if (state.hovered === node) {
            res.highlighted = true;
            res.forceLabel = true;
        }
        const isMain = attrs.role === "center" || attrs.role === "hub";
        if (isMain) res.forceLabel = true;
        if (state.highlightNodes) {
            if (state.highlightNodes.has(node)) {
                res.zIndex = node === state.selectedTerm ? 3 : 2;
                res.forceLabel = true;
                if (node === state.selectedTerm) {
                    // 仅放大选中节点，不设 highlighted，避免 Sigma 给标签套白色高亮底框。
                    res.size = attrs.size * 1.25;
                }
            } else {
                // 邻域之外的节点：保留本来的色相，只降低不透明度，避免“全部刷白”。
                res.color = withAlpha(attrs.color, isDarkTheme() ? 0.32 : 0.4);
                // 主节点(中心/枢纽)的文字保留并按本来的标签色降透明度，不变白、不消失。
                if (isMain) {
                    res.labelColor = withAlpha(attrs.labelColor || attrs.color, isDarkTheme() ? 0.5 : 0.55);
                    res.forceLabel = true;
                } else {
                    res.label = "";
                    res.forceLabel = false;
                }
                res.zIndex = 0;
            }
        }
        return res;
    }

    function edgeReducer(edge, attrs) {
        const res = { ...attrs };
        const type = attrs.relation_type || "cooccurrence";
        if (state.highlightEdges) {
            if (state.highlightEdges.has(edge)) {
                res.color = relationColor(type, 0.85);
                res.size = Math.max(Number(attrs.size || 1), 1.6);
                res.forceLabel = Number(attrs.weight || 0) >= 3;
                res.zIndex = 1;
            } else {
                // 邻域之外的边：保留关系色相，进一步降透明度，仍然可见。
                res.color = relationColor(type, dimEdgeAlpha());
                res.label = "";
                res.hidden = false;
            }
        } else {
            // 默认状态：所有边都展示，使用带透明度的关系色。
            res.color = relationColor(type, defaultEdgeAlpha());
            res.label = "";
        }
        return res;
    }

    function bindSigmaEvents() {
        const renderer = state.renderer;

        renderer.on("clickNode", ({ node }) => {
            // 单击即选中并展开：加载该节点周围的所有子节点与连接边。
            selectNode(node);
            expandNode(node);
        });
        renderer.on("doubleClickNode", ({ node, event }) => {
            if (event && event.preventSigmaDefault) event.preventSigmaDefault();
            selectNode(node);
            expandNode(node);
        });
        renderer.on("clickStage", () => {
            state.selectedTerm = "";
            clearHighlight();
        });

        renderer.on("enterNode", ({ node }) => {
            state.hovered = node;
            els.canvas.style.cursor = "pointer";
            renderer.refresh({ skipIndexation: true });
        });
        renderer.on("leaveNode", () => {
            state.hovered = "";
            els.canvas.style.cursor = "";
            renderer.refresh({ skipIndexation: true });
        });

        // —— 拖拽节点 ——
        renderer.on("downNode", ({ node }) => {
            state.dragged = node;
            state.graph.setNodeAttribute(node, "highlighted", true);
        });
        const captor = renderer.getMouseCaptor();
        captor.on("mousemovebody", event => {
            if (!state.dragged) return;
            const pos = renderer.viewportToGraph(event);
            state.graph.setNodeAttribute(state.dragged, "x", pos.x);
            state.graph.setNodeAttribute(state.dragged, "y", pos.y);
            event.preventSigmaDefault();
            event.original.preventDefault();
            event.original.stopPropagation();
        });
        const endDrag = () => {
            if (!state.dragged) return;
            state.graph.removeNodeAttribute(state.dragged, "highlighted");
            state.dragged = null;
        };
        captor.on("mouseup", endDrag);
        captor.on("mouseleave", endDrag);

        // 相机变化(缩放/平移)后触发自动展开判断
        renderer.getCamera().on("updated", scheduleAutoExpand);
    }

    async function loadOverview() {
        if (!state.graph && !initGraph()) return;
        state.loading = true;
        state.selectedTerm = "";
        state.expanded.clear();
        state.expandQueue.clear();
        state.highlightNodes = null;
        state.highlightEdges = null;
        resetNewsPager("");
        els.sideContent.classList.add("is-hidden");
        els.sideEmpty.classList.remove("is-hidden");
        setStatus("正在加载图谱...", true);
        els.empty.classList.add("is-hidden");
        try {
            stopLayout();
            state.graph.clear();
            const params = currentFilters();
            params.set("limit", "120");
            params.set("edge_limit", "320");
            const data = await fetchJson(`/api/graph/overview?${params.toString()}`);
            mergeGraphData(data, { layer: "base" });
            const summary = data.summary || {};
            setStatus(`已加载 ${summary.node_count || state.graph.order} 个节点、${summary.edge_count || state.graph.size} 条关系`, false);
            if (state.renderer) state.renderer.getCamera().animatedReset({ duration: 260 });
        } catch (e) {
            setStatus(`加载失败：${e.message}`, false);
            els.empty.classList.remove("is-hidden");
        } finally {
            state.loading = false;
            flushExpandQueue();
        }
    }

    async function expandNode(term) {
        const cleanTerm = String(term || "").trim();
        if (!cleanTerm) return;
        if (state.loading) { state.expandQueue.add(cleanTerm); return; }
        if (state.expanded.has(cleanTerm)) return;
        state.loading = true;
        setStatus(`正在展开「${cleanTerm}」...`, true);
        try {
            const params = currentFilters();
            params.set("term", cleanTerm);
            params.set("limit", "40");
            params.set("edge_limit", "100");
            const data = await fetchJson(`/api/graph/expand?${params.toString()}`);
            mergeGraphData(data, { center: cleanTerm, layer: "expanded" });
            state.expanded.add(cleanTerm);
            if (state.selectedTerm === cleanTerm) {
                computeHighlight(cleanTerm);
                state.renderer.refresh();
            }
            setStatus(`已展开「${cleanTerm}」`, false);
        } catch (e) {
            setStatus(`展开失败：${e.message}`, false);
        } finally {
            state.loading = false;
            flushExpandQueue();
        }
    }

    function flushExpandQueue() {
        if (state.loading || !state.expandQueue.size) return;
        const next = state.expandQueue.values().next().value;
        state.expandQueue.delete(next);
        expandNode(next);
    }

    // —— 放大时自动展开：当缩放进入放大区间且视口内节点过少时，
    //    自动展开视口内“最密集且尚未展开”的节点，补充周围邻居。
    const AUTO_EXPAND_RATIO = 0.7;   // 相机 ratio 小于此值视为已放大
    const AUTO_EXPAND_MIN_VISIBLE = 6; // 视口内可见节点少于此值时触发

    function countVisibleNodes() {
        const renderer = state.renderer;
        const graph = state.graph;
        if (!renderer || !graph) return { count: 0, candidate: "" };
        const { width, height } = renderer.getDimensions();
        let count = 0;
        let candidate = "";
        let bestDegree = -1;
        graph.forEachNode(node => {
            // graphToViewport 把图坐标换算成视口像素，才能判断节点是否在屏幕内；
            // getNodeDisplayData 返回的是 framed 归一化坐标，不能直接比较像素。
            const p = renderer.graphToViewport(graph.getNodeAttributes(node));
            if (!p) return;
            if (p.x >= 0 && p.x <= width && p.y >= 0 && p.y <= height) {
                count += 1;
                if (!state.expanded.has(node)) {
                    const deg = graph.degree(node);
                    if (deg > bestDegree) { bestDegree = deg; candidate = node; }
                }
            }
        });
        return { count, candidate };
    }

    function maybeAutoExpand() {
        if (state.loading || !state.renderer || !state.graph) return;
        const ratio = state.renderer.getCamera().getState().ratio;
        if (ratio > AUTO_EXPAND_RATIO) return; // 仅在放大时触发
        const { count, candidate } = countVisibleNodes();
        if (count >= AUTO_EXPAND_MIN_VISIBLE || !candidate) return;
        expandNode(candidate);
    }

    function scheduleAutoExpand() {
        window.clearTimeout(state.autoExpandTimer);
        // 略长的去抖：等用户停止缩放/平移后再判断，避免连续触发造成节点接连弹出。
        state.autoExpandTimer = window.setTimeout(maybeAutoExpand, 420);
    }

    function centerNode(term) {
        if (!state.renderer || !state.graph || !state.graph.hasNode(term)) return;
        const camera = state.renderer.getCamera();
        // Sigma v3 相机工作在 framed(归一化)坐标系，必须用 getNodeDisplayData 取节点的
        // framed 坐标，而不是图的原始 x/y，否则相机会移动到空白处。
        const display = state.renderer.getNodeDisplayData(term);
        if (!display) return;
        const current = camera.getState();
        const nextRatio = clamp(current.ratio, 0.4, 1.1);
        camera.animate({ x: display.x, y: display.y, ratio: nextRatio }, { duration: 260 });
    }

    async function selectNode(term) {
        const cleanTerm = String(term || "").trim();
        if (!cleanTerm || !state.graph || !state.graph.hasNode(cleanTerm)) return;
        state.selectedTerm = cleanTerm;
        computeHighlight(cleanTerm);
        state.renderer.refresh();
        window.requestAnimationFrame(() => centerNode(cleanTerm));

        els.sideEmpty.classList.add("is-hidden");
        els.sideContent.classList.remove("is-hidden");
        els.nodeTitle.textContent = cleanTerm;
        els.nodeMetrics.innerHTML = `<div class="graph-detail-loading">正在加载分析...</div>`;
        els.trendBars.innerHTML = "";
        els.neighbors.innerHTML = "";
        els.newsList.innerHTML = "";
        try {
            const params = currentFilters();
            const data = await fetchJson(`/api/graph/node/${encodeURIComponent(cleanTerm)}?${params.toString()}`);
            renderNodeDetail(data);
        } catch (e) {
            els.nodeMetrics.innerHTML = `<div class="graph-detail-error">加载失败：${escapeHtml(e.message)}</div>`;
        }
    }

    function renderNodeDetail(data) {
        const summary = data.summary || {};
        els.nodeMetrics.innerHTML = `
            <div><strong>${Number(summary.related_count || 0)}</strong><span>相关新闻</span></div>
            <div><strong>${Number(summary.total_heat || 0).toFixed(1)}</strong><span>总热度</span></div>
            <div><strong>${Number(summary.avg_sentiment || 0).toFixed(1)}</strong><span>情绪均值</span></div>
        `;
        renderTrend(data.trend || {});
        renderNeighbors(data.neighbors || []);
        resetNewsPager(data.term || state.selectedTerm);
        loadMoreNodeNews();
    }

    function renderTrend(trend) {
        const dates = trend.dates || [];
        const counts = trend.count || [];
        if (!dates.length) {
            els.trendBars.innerHTML = `<div class="graph-muted">暂无趋势数据</div>`;
            return;
        }
        const recentDates = dates.slice(-18);
        const recentCounts = counts.slice(-18);
        const max = Math.max(1, ...recentCounts.map(Number));
        els.trendBars.innerHTML = recentDates.map((date, index) => {
            const value = Number(recentCounts[index] || 0);
            const height = Math.max(8, Math.round(value / max * 74));
            return `<span title="${escapeHtml(date)}：${value}" style="height:${height}px"></span>`;
        }).join("");
    }

    function renderNeighbors(items) {
        if (!items.length) {
            els.neighbors.innerHTML = `<div class="graph-muted">暂无关联节点</div>`;
            return;
        }
        els.neighbors.innerHTML = items.map(item => `
            <button type="button" data-term="${escapeHtml(item.name)}">
                ${escapeHtml(item.name)} <em>${Number(item.value || 0)}</em>
            </button>
        `).join("");
        els.neighbors.querySelectorAll("button[data-term]").forEach(btn => {
            btn.addEventListener("click", () => {
                const term = btn.dataset.term || "";
                if (state.graph && state.graph.hasNode(term)) selectNode(term);
                else expandNode(term).then(() => selectNode(term));
            });
        });
    }

    function resetNewsPager(term) {
        if (state.newsPager.observer) {
            state.newsPager.observer.disconnect();
            state.newsPager.observer = null;
        }
        state.newsPager.term = String(term || "").trim();
        state.newsPager.page = 1;
        state.newsPager.hasMore = false;
        state.newsPager.loading = false;
        els.newsList.innerHTML = state.newsPager.term ? `<div class="graph-news-loading">正在加载相关新闻...</div>` : "";
    }

    function renderNewsItems(items) {
        return items.map(item => {
            const time = item.time ? new Date(item.time).toLocaleString("zh-CN") : "-";
            const id = Number(item.id || 0);
            return `
                <article class="graph-news-item">
                    <button type="button" data-news-id="${id}">${escapeHtml(item.title || "(无标题)")}</button>
                    <div>
                        <span>${escapeHtml(item.source || "未知来源")}</span>
                        <span>${time}</span>
                        <span>热度 ${Number(item.heat || 0).toFixed(1)}</span>
                    </div>
                </article>
            `;
        }).join("");
    }

    function bindGraphNewsClicks() {
        els.newsList.querySelectorAll("button[data-news-id]").forEach(btn => {
            btn.addEventListener("click", () => {
                window.openNewsDetail(Number(btn.dataset.newsId || 0));
            });
        });
    }

    function renderNewsPage(items, append) {
        const html = renderNewsItems(items);
        if (append) {
            const sentinel = els.newsList.querySelector(".graph-news-sentinel");
            if (sentinel) sentinel.remove();
            els.newsList.insertAdjacentHTML("beforeend", html);
        } else {
            els.newsList.innerHTML = html || `<div class="graph-muted">暂无相关新闻</div>`;
        }
        bindGraphNewsClicks();
        renderNewsPagerState();
    }

    function renderNewsPagerState() {
        const old = els.newsList.querySelector(".graph-news-sentinel");
        if (old) old.remove();
        if (state.newsPager.loading) {
            els.newsList.insertAdjacentHTML("beforeend", `<div class="graph-news-sentinel">继续加载中...</div>`);
            return;
        }
        if (state.newsPager.hasMore) {
            els.newsList.insertAdjacentHTML("beforeend", `<div class="graph-news-sentinel" data-load-more="1">滚动加载更多</div>`);
            observeNewsSentinel();
        } else if (els.newsList.querySelector(".graph-news-item")) {
            els.newsList.insertAdjacentHTML("beforeend", `<div class="graph-news-sentinel muted">已显示全部</div>`);
        }
    }

    function observeNewsSentinel() {
        if (state.newsPager.observer) {
            state.newsPager.observer.disconnect();
            state.newsPager.observer = null;
        }
        const sentinel = els.newsList.querySelector("[data-load-more]");
        if (!sentinel) return;
        state.newsPager.observer = new IntersectionObserver(entries => {
            if (entries.some(entry => entry.isIntersecting)) loadMoreNodeNews();
        }, { root: els.sidePanel || null, rootMargin: "120px 0px", threshold: 0.1 });
        state.newsPager.observer.observe(sentinel);
    }

    async function loadMoreNodeNews() {
        const pager = state.newsPager;
        const term = pager.term || state.selectedTerm;
        if (!term || pager.loading) return;
        pager.loading = true;
        renderNewsPagerState();
        try {
            const params = currentFilters();
            params.set("page", String(pager.page));
            params.set("page_size", String(pager.pageSize));
            const data = await fetchJson(`/api/graph/node/${encodeURIComponent(term)}/news?${params.toString()}`);
            if (term !== (pager.term || state.selectedTerm)) {
                pager.loading = false;
                return;
            }
            const items = data.data || [];
            pager.hasMore = !!data.has_more;
            pager.page += 1;
            pager.loading = false;
            renderNewsPage(items, pager.page > 2);
        } catch (e) {
            pager.loading = false;
            els.newsList.innerHTML = `<div class="graph-detail-error">相关新闻加载失败：${escapeHtml(e.message)}</div>`;
            pager.hasMore = false;
            renderNewsPagerState();
        }
    }

    function bindUi() {
        els.rangeTabs.querySelectorAll("button[data-range]").forEach(btn => {
            btn.addEventListener("click", () => {
                els.rangeTabs.querySelectorAll("button").forEach(item => item.classList.remove("active"));
                btn.classList.add("active");
                state.range = btn.dataset.range || "24h";
                loadOverview();
            });
        });
        [els.category, els.region, els.source].forEach(select => {
            select.addEventListener("change", loadOverview);
        });
        els.refreshBtn.addEventListener("click", loadOverview);
        els.resetBtn.addEventListener("click", () => {
            state.selectedTerm = "";
            clearHighlight();
            if (state.renderer) state.renderer.getCamera().animatedReset({ duration: 260 });
        });
        els.searchBtn.addEventListener("click", () => {
            const term = els.search.value.trim();
            if (!term) return;
            if (state.graph && state.graph.hasNode(term)) selectNode(term);
            else expandNode(term).then(() => selectNode(term));
        });
        els.search.addEventListener("keydown", event => {
            if (event.key === "Enter") {
                event.preventDefault();
                els.searchBtn.click();
            }
        });
        els.expandBtn.addEventListener("click", () => {
            if (state.selectedTerm) expandNode(state.selectedTerm);
        });
        els.closeDetailBtn.addEventListener("click", () => {
            state.selectedTerm = "";
            clearHighlight();
            resetNewsPager("");
            els.sideContent.classList.add("is-hidden");
            els.sideEmpty.classList.remove("is-hidden");
        });
        if (els.detailCloseBtn) {
            els.detailCloseBtn.addEventListener("click", closeNewsDetail);
        }
        if (els.detailModal) {
            els.detailModal.addEventListener("click", event => {
                if (event.target === els.detailModal) closeNewsDetail();
            });
        }
        document.addEventListener("keydown", event => {
            if (event.key === "Escape" && els.detailModal && els.detailModal.classList.contains("show")) {
                closeNewsDetail();
            }
        });
        window.addEventListener("beforeunload", () => {
            if (state.fa2) state.fa2.kill();
        });
    }

    async function boot() {
        bindUi();
        if (!initGraph()) return;
        await loadFilters();
        await loadOverview();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", boot);
    } else {
        boot();
    }
})();
