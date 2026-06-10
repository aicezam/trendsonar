// 本文件承载专题列表页的筛选、分页和管理交互逻辑。
const topicsPageDataEl = document.getElementById("page-data");
const topicsPageData = topicsPageDataEl ? JSON.parse(topicsPageDataEl.textContent || "{}") : {};
const IS_ADMIN = !!topicsPageData.isAdmin;
const TOPIC_FALLBACK_CATEGORIES = topicsPageData.newsCategories || [];
    const TIME_OPTIONS = [
        { label: "全部时间", value: "all" },
        { label: "今日", value: "today" },
        { label: "24小时", value: "24h" },
        { label: "3天内", value: "3d" },
        { label: "7天内", value: "7d" },
        { label: "本周", value: "week" },
        { label: "本月", value: "month" },
        { label: "今年", value: "year" },
        { label: "其他", value: "other" }
    ];
    const state = {
        filter: {
            q: "",
            sortBy: "updated",
            dateOption: "all",
            startDate: "",
            endDate: "",
            minHeat: "0",
            category: "all",
            region: ""
        }
    };
    const els = {
        mobileFilter: document.getElementById("mobileTopicFilter"),
        desktopFilter: document.getElementById("desktopTopicFilter"),
        openDesktopFilter: document.getElementById("openDesktopFilter"),
        closeDesktopFilter: document.getElementById("closeDesktopFilter"),
        applyDesktopFilter: document.getElementById("applyDesktopFilter"),
        openMobileFilter: document.getElementById("openMobileFilter"),
        closeMobileFilter: document.getElementById("closeMobileFilter"),
        applyMobileFilter: document.getElementById("applyMobileFilter"),
        resetMobileFilter: document.getElementById("resetMobileFilter"),
        resetDesktopFilter: document.getElementById("resetDesktopFilter"),
        resetToolbarFilter: document.getElementById("resetToolbarFilter"),
        refreshTopicsBtn: document.getElementById("refreshTopicsBtn"),
        refreshMobileTopicsBtn: document.getElementById("refreshMobileTopicsBtn"),
        heroTotal: document.getElementById("heroTotal"),
        heroLoaded: document.getElementById("heroLoaded"),
        heroFilter: document.getElementById("heroFilter"),
        categoryTabs: document.getElementById("topicCategoryTabs"),
        mobileSearchInput: document.getElementById("mobileSearchInput"),
        mobileSearchBtn: document.getElementById("mobileSearchBtn"),
        desktopSearchInput: document.getElementById("desktopSearchInput"),
        mobileTimeOptions: document.getElementById("mobileTimeOptions"),
        desktopTimeSelect: document.getElementById("desktopTimeSelect"),
        drawerTimeSelect: document.getElementById("drawerTimeSelect"),
        mobileDateRange: document.getElementById("mobileDateRange"),
        desktopDateRange: document.getElementById("desktopDateRange"),
        drawerDateRange: document.getElementById("drawerDateRange"),
        mobileStartDate: document.getElementById("mobileStartDate"),
        mobileEndDate: document.getElementById("mobileEndDate"),
        desktopStartDate: document.getElementById("desktopStartDate"),
        desktopEndDate: document.getElementById("desktopEndDate"),
        drawerStartDate: document.getElementById("drawerStartDate"),
        drawerEndDate: document.getElementById("drawerEndDate"),
        mobileHeatSelect: document.getElementById("mobileHeatSelect"),
        desktopHeatSelect: document.getElementById("desktopHeatSelect"),
        drawerHeatSelect: document.getElementById("drawerHeatSelect"),
        mobileRegionContainer: document.getElementById("mobileRegionContainer"),
        desktopRegionSelect: document.getElementById("desktopRegionSelect"),
        sortBtns: {
            mobileUpdated: document.getElementById("mobileSortUpdated"),
            mobileHeat: document.getElementById("mobileSortHeat"),
            desktopUpdated: document.getElementById("desktopSortUpdated"),
            desktopHeat: document.getElementById("desktopSortHeat")
        }
    };

    function fmtTime(iso) {
        if (!iso) return "-";
        const d = new Date(iso);
        if (isNaN(d.getTime())) return iso;
        return d.toLocaleString();
    }

    function fmtShortDate(iso) {
        if (!iso) return "-";
        const d = new Date(iso);
        if (isNaN(d.getTime())) return iso;
        return d.toLocaleDateString("zh-CN", { month: "2-digit", day: "2-digit" });
    }

    function fmtRelative(iso) {
        if (!iso) return "-";
        const d = new Date(iso);
        if (isNaN(d.getTime())) return iso;
        const diff = Date.now() - d.getTime();
        const minute = 60 * 1000;
        const hour = 60 * minute;
        const day = 24 * hour;
        if (diff < hour) return `${Math.max(1, Math.floor(diff / minute))} 分钟前`;
        if (diff < day) return `${Math.floor(diff / hour)} 小时前`;
        if (diff < 7 * day) return `${Math.floor(diff / day)} 天前`;
        return fmtShortDate(iso);
    }

    function heatTone(score) {
        if (score >= 80) return "high";
        if (score >= 40) return "mid";
        return "low";
    }

    function heatLabel(value) {
        const num = Number(value || 0);
        return num > 0 ? `${num}+` : "不限";
    }

    function el(tag, attrs = {}, children = []) {
        const node = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
            if (k === "class") node.className = v;
            else if (k === "text") node.textContent = v;
            else if (k.startsWith("on")) node.addEventListener(k.substring(2).toLowerCase(), v);
            else node.setAttribute(k, v);
        }
        for (const c of children) node.appendChild(c);
        return node;
    }

    class MultiSelect {
        constructor(container, placeholder, options, selectedValues, onChange) {
            this.container = typeof container === "string" ? document.getElementById(container) : container;
            this.placeholder = placeholder;
            this.options = options || [];
            this.selected = new Set(selectedValues ? selectedValues.split(",").filter(x => x && x !== "all") : []);
            this.onChange = onChange;
            this.render();
        }

        render() {
            if (!this.container) return;
            this.container.innerHTML = "";
            const trigger = el("div", { class: "ms-trigger" });
            this.updateTriggerText(trigger);
            trigger.onclick = (e) => {
                e.stopPropagation();
                this.toggleDropdown();
            };
            this.container.appendChild(trigger);

            const dropdown = el("div", { class: "ms-dropdown" });
            this.dropdown = dropdown;
            if (this.options.length > 8) {
                const searchBox = el("div", { class: "ms-search" });
                const input = el("input", { placeholder: "搜索..." });
                input.onclick = (e) => e.stopPropagation();
                input.onkeyup = (e) => {
                    const term = e.target.value.toLowerCase();
                    dropdown.querySelectorAll(".ms-option").forEach(opt => {
                        opt.style.display = opt.textContent.toLowerCase().includes(term) ? "flex" : "none";
                    });
                };
                searchBox.appendChild(input);
                dropdown.appendChild(searchBox);
            }

            this.options.forEach(opt => {
                const row = el("div", { class: "ms-option" });
                const cb = el("input", { type: "checkbox" });
                cb.checked = this.selected.has(opt.value);
                const toggle = (checked) => {
                    if (checked) this.selected.add(opt.value);
                    else this.selected.delete(opt.value);
                    cb.checked = this.selected.has(opt.value);
                    this.updateTriggerText(trigger);
                    this.onChange(Array.from(this.selected).join(","));
                };
                row.onclick = (e) => {
                    e.stopPropagation();
                    toggle(!this.selected.has(opt.value));
                };
                cb.onclick = (e) => {
                    e.stopPropagation();
                    toggle(e.target.checked);
                };
                row.appendChild(cb);
                row.appendChild(document.createTextNode(opt.label));
                dropdown.appendChild(row);
            });
            this.container.appendChild(dropdown);
        }

        updateTriggerText(trigger) {
            if (this.selected.size === 0) {
                trigger.textContent = this.placeholder;
                trigger.classList.remove("has-value");
            } else if (this.selected.size === 1) {
                const val = Array.from(this.selected)[0];
                const opt = this.options.find(item => item.value === val);
                trigger.textContent = opt ? opt.label : val;
                trigger.classList.add("has-value");
            } else {
                trigger.textContent = `${this.placeholder} (${this.selected.size})`;
                trigger.classList.add("has-value");
            }
            trigger.title = trigger.textContent;
        }

        toggleDropdown() {
            document.querySelectorAll(".ms-dropdown.show").forEach(d => {
                if (d !== this.dropdown) d.classList.remove("show");
            });
            this.dropdown.classList.toggle("show");
        }

        updateSelected(value) {
            this.selected = new Set(value ? value.split(",").filter(x => x && x !== "all") : []);
            this.render();
        }
    }

    let msRegion, msMobileRegion;

    function isMobile() {
        return window.innerWidth <= 720;
    }

    function getEffectiveDate() {
        return state.filter.dateOption === "other" ? "" : state.filter.dateOption;
    }

    function resetFilters() {
        state.filter = {
            q: "",
            sortBy: "updated",
            dateOption: "all",
            startDate: "",
            endDate: "",
            minHeat: "0",
            category: "all",
            region: ""
        };
        els.mobileSearchInput.value = "";
        els.desktopSearchInput.value = "";
        if (msRegion) msRegion.updateSelected("");
        if (msMobileRegion) msMobileRegion.updateSelected("");
        syncFilterUI();
        loadTopics(1);
    }

    function renderSkeletons(list) {
        list.innerHTML = "";
        for (let i = 0; i < 4; i += 1) {
            const card = el("div", { class: "topic-card topic-card-skeleton" });
            card.innerHTML = `
                <div class="skeleton-line wide"></div>
                <div class="skeleton-line mid"></div>
                <div class="skeleton-line"></div>
            `;
            list.appendChild(card);
        }
    }

    function syncFilterUI() {
        const showRange = state.filter.dateOption === "other";
        els.mobileDateRange.classList.toggle("is-hidden", !showRange);
        els.desktopDateRange.classList.toggle("is-hidden", !showRange);

        for (const btn of Array.from(els.mobileTimeOptions.children)) {
            btn.classList.toggle("active", btn.dataset.val === state.filter.dateOption);
        }
        els.desktopTimeSelect.value = state.filter.dateOption;
        els.drawerTimeSelect.value = state.filter.dateOption;

        els.mobileStartDate.value = state.filter.startDate;
        els.mobileEndDate.value = state.filter.endDate;
        els.desktopStartDate.value = state.filter.startDate;
        els.desktopEndDate.value = state.filter.endDate;
        els.drawerStartDate.value = state.filter.startDate;
        els.drawerEndDate.value = state.filter.endDate;
        els.mobileHeatSelect.value = state.filter.minHeat || "0";
        els.desktopHeatSelect.value = state.filter.minHeat || "0";
        els.drawerHeatSelect.value = state.filter.minHeat || "0";
        if (els.drawerDateRange) els.drawerDateRange.classList.toggle("is-hidden", !showRange);

        Object.values(els.sortBtns).forEach(btn => {
            btn.classList.toggle("active", btn.dataset.sort === state.filter.sortBy);
        });

        if (els.categoryTabs) {
            Array.from(els.categoryTabs.children).forEach(btn => {
                const isAll = state.filter.category === "all" || !state.filter.category;
                btn.classList.toggle("active", (btn.dataset.category === "all" && isAll) || btn.dataset.category === state.filter.category);
            });
        }
        if (els.heroFilter) els.heroFilter.textContent = heatLabel(state.filter.minHeat);
    }

    function setTimeOption(value) {
        state.filter.dateOption = value;
        if (value !== "other") {
            state.filter.startDate = "";
            state.filter.endDate = "";
        }
        syncFilterUI();
        if (!isMobile() && value !== "other") loadTopics(1);
    }

    function setSort(value) {
        state.filter.sortBy = value;
        syncFilterUI();
        if (!isMobile()) loadTopics(1);
    }

    function renderTimeOptions() {
        els.mobileTimeOptions.innerHTML = "";
        TIME_OPTIONS.forEach(opt => {
            const btn = el("button", {
                class: "topics-filter-btn",
                text: opt.label,
                type: "button",
                "data-val": opt.value,
                onClick: () => setTimeOption(opt.value)
            });
            els.mobileTimeOptions.appendChild(btn);
        });

        els.desktopTimeSelect.innerHTML = "";
        els.drawerTimeSelect.innerHTML = "";
        TIME_OPTIONS.forEach(opt => {
            const item = el("option", { value: opt.value, text: opt.label });
            els.desktopTimeSelect.appendChild(item);
            els.drawerTimeSelect.appendChild(el("option", { value: opt.value, text: opt.label }));
        });
    }

    function fetchTopicCategories() {
        const fallback = ["全部", ...TOPIC_FALLBACK_CATEGORIES];
        renderTopicCategoryTabs(fallback);
    }

    function renderTopicCategoryTabs(categories) {
        if (!els.categoryTabs) return;
        els.categoryTabs.innerHTML = "";
        categories.forEach(cat => {
            const val = cat === "全部" ? "all" : cat;
            const btn = el("div", { class: "category-tab-item", text: cat, "data-category": val });
            btn.onclick = () => {
                state.filter.category = val;
                syncFilterUI();
                loadTopics(1);
                const scrollLeft = btn.offsetLeft - (els.categoryTabs.offsetWidth / 2) + (btn.offsetWidth / 2);
                els.categoryTabs.scrollTo({ left: scrollLeft, behavior: "smooth" });
            };
            els.categoryTabs.appendChild(btn);
        });
        syncFilterUI();
    }

    async function fetchTopicRegions() {
        try {
            const resp = await fetch("/api/regions", { cache: "no-store" });
            if (!resp.ok) throw new Error(resp.status);
            const list = await resp.json();
            const options = list.map(r => ({ label: r, value: r }));
            msRegion = new MultiSelect(els.desktopRegionSelect, "全部地区", options, state.filter.region, (val) => {
                state.filter.region = val;
                if (msMobileRegion) msMobileRegion.updateSelected(val);
            });
            msMobileRegion = new MultiSelect(els.mobileRegionContainer, "全部地区", options, state.filter.region, (val) => {
                state.filter.region = val;
                if (msRegion) msRegion.updateSelected(val);
            });
        } catch (e) {
            msRegion = new MultiSelect(els.desktopRegionSelect, "加载失败", [], "", () => {});
            msMobileRegion = new MultiSelect(els.mobileRegionContainer, "加载失败", [], "", () => {});
        }
    }

    function bindFilterEvents() {
        els.openMobileFilter.onclick = () => els.mobileFilter.classList.add("show");
        els.closeMobileFilter.onclick = () => els.mobileFilter.classList.remove("show");
        els.mobileFilter.onclick = (e) => {
            if (e.target === els.mobileFilter) els.mobileFilter.classList.remove("show");
        };
        els.openDesktopFilter.onclick = () => els.desktopFilter.classList.add("show");
        els.closeDesktopFilter.onclick = () => els.desktopFilter.classList.remove("show");
        els.desktopFilter.onclick = (e) => {
            if (e.target === els.desktopFilter) els.desktopFilter.classList.remove("show");
        };
        els.resetMobileFilter.onclick = () => {
            els.mobileFilter.classList.remove("show");
            resetFilters();
        };
        els.resetDesktopFilter.onclick = () => {
            els.desktopFilter.classList.remove("show");
            resetFilters();
        };
        els.resetToolbarFilter.onclick = resetFilters;
        els.refreshTopicsBtn.onclick = () => loadTopics(1);
        els.refreshMobileTopicsBtn.onclick = () => loadTopics(1);

        const runSearch = (value) => {
            state.filter.q = (value || "").trim();
            els.mobileSearchInput.value = state.filter.q;
            els.desktopSearchInput.value = state.filter.q;
            loadTopics(1);
        };
        const onSearchKey = (e) => {
            if (e.key === "Enter") runSearch(e.target.value);
        };
        els.mobileSearchInput.onkeyup = onSearchKey;
        els.desktopSearchInput.onkeyup = onSearchKey;
        els.mobileSearchBtn.onclick = () => runSearch(els.mobileSearchInput.value);

        els.desktopTimeSelect.onchange = (e) => setTimeOption(e.target.value);
        els.drawerTimeSelect.onchange = (e) => setTimeOption(e.target.value);
        els.mobileHeatSelect.onchange = (e) => {
            state.filter.minHeat = e.target.value;
            els.desktopHeatSelect.value = state.filter.minHeat;
            els.drawerHeatSelect.value = state.filter.minHeat;
        };
        els.desktopHeatSelect.onchange = (e) => {
            state.filter.minHeat = e.target.value;
            els.mobileHeatSelect.value = state.filter.minHeat;
            els.drawerHeatSelect.value = state.filter.minHeat;
            loadTopics(1);
        };
        els.drawerHeatSelect.onchange = (e) => {
            state.filter.minHeat = e.target.value;
            syncFilterUI();
        };

        const onDateChange = (e, field) => {
            state.filter[field] = e.target.value;
            syncFilterUI();
            if (!isMobile() && state.filter.dateOption === "other" && state.filter.startDate && state.filter.endDate) {
                loadTopics(1);
            }
        };
        els.mobileStartDate.onchange = (e) => onDateChange(e, "startDate");
        els.mobileEndDate.onchange = (e) => onDateChange(e, "endDate");
        els.desktopStartDate.onchange = (e) => onDateChange(e, "startDate");
        els.desktopEndDate.onchange = (e) => onDateChange(e, "endDate");
        els.drawerStartDate.onchange = (e) => onDateChange(e, "startDate");
        els.drawerEndDate.onchange = (e) => onDateChange(e, "endDate");

        Object.values(els.sortBtns).forEach(btn => {
            btn.onclick = () => setSort(btn.dataset.sort);
        });

        els.applyMobileFilter.onclick = () => {
            state.filter.q = els.mobileSearchInput.value.trim();
            els.desktopSearchInput.value = state.filter.q;
            els.mobileFilter.classList.remove("show");
            loadTopics(1);
        };
        els.applyDesktopFilter.onclick = () => {
            state.filter.q = els.desktopSearchInput.value.trim();
            els.mobileSearchInput.value = state.filter.q;
            els.desktopFilter.classList.remove("show");
            loadTopics(1);
        };

        document.addEventListener("click", (e) => {
            if (!e.target.closest(".ms-container")) {
                document.querySelectorAll(".ms-dropdown.show").forEach(d => d.classList.remove("show"));
            }
        });
        document.addEventListener("keydown", (e) => {
            if (e.key === "Escape") {
                els.mobileFilter.classList.remove("show");
                els.desktopFilter.classList.remove("show");
            }
        });
    }

    async function deleteTopic(e, id, name) {
        e.stopPropagation();
        if (!confirm(`确定要删除专题 "${name}" 吗？此操作不可恢复。`)) return;

        try {
            const resp = await fetch(`/api/topics/${id}`, { method: "DELETE" });
            if (!resp.ok) throw new Error("删除失败");
            loadTopics(); // 刷新列表
        } catch (err) {
            alert("删除失败: " + err.message);
        }
    }

    async function manualCreateTopic() {
        const name = prompt("【手动创建专题】\n\n注意：\n1. 创建后系统将立即在后台扫描过去几天的新闻进行匹配。\n2. 专题名称越精准，匹配效果越好。\n\n请输入新专题名称（如：俄乌冲突事件全纪录）：");
        if (!name) return;
        if (name.trim().length < 2) {
            alert("名称太短");
            return;
        }

        try {
            document.getElementById("addTopicBtn").disabled = true;
            document.getElementById("addTopicBtn").textContent = "创建中...";

            const resp = await fetch("/api/topics/manual_create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name.trim() })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "请求失败");
            }

            alert("专题创建成功。\n\n系统正在后台根据新专题名称和向量扫描相关新闻，请稍后刷新查看结果。");
            loadTopics();
        } catch (e) {
            alert("创建失败: " + e.message);
        } finally {
            const btn = document.getElementById("addTopicBtn");
            btn.disabled = false;
            btn.textContent = "+ 手动创建专题";
        }
    }

    async function editTopic(e, id, oldName) {
        e.stopPropagation();
        const newName = prompt(
            `【修改专题名称】\n\n当前名称：${oldName}\n\n注意：\n修改名称将重新生成该专题的 AI 向量，并触发重新扫描。\n这可能会导致：\n1. 新闻匹配逻辑发生变化\n2. 部分旧新闻可能不再匹配（取决于后续逻辑）\n3. 系统负载短暂上升\n\n请输入新的名称：`,
            oldName
        );

        if (!newName || newName === oldName) return;
        if (newName.trim().length < 2) {
            alert("名称太短");
            return;
        }

        try {
            const resp = await fetch(`/api/topics/${id}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: newName.trim() })
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "请求失败");
            }

            alert("修改成功。\n\n系统已重新生成向量并触发后台重新扫描，请稍后查看更新结果。");
            loadTopics();
        } catch (e) {
            alert("修改失败: " + e.message);
        }
    }

    if (IS_ADMIN) {
        const btn = document.getElementById("addTopicBtn");
        btn.classList.remove("is-hidden");
        btn.onclick = manualCreateTopic;
    }

    let currentPage = 1;
    const pageSize = 10;
    let isLoading = false;
    let hasMore = true;

    function renderTopicCard(t) {
        const card = el("div", { class: "topic-card" });
        card.tabIndex = 0;
        card.setAttribute("role", "link");
        card.addEventListener("click", () => {
            window.location.href = `/topics/${t.id}`;
        });
        card.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                window.location.href = `/topics/${t.id}`;
            }
        });

        const header = el("div", { class: "topic-card-head" });

        const nameWrap = el("div", { class: "topic-title-wrap" });
        nameWrap.appendChild(el("div", { class: "topic-name", text: t.name || "未命名专题" }));
        nameWrap.appendChild(el("div", { class: "topic-subline", text: `更新于 ${fmtRelative(t.updated_time)}` }));
        header.appendChild(nameWrap);

        const heat = Number(t.heat_score || 0);
        const heatBadge = el("div", { class: `topic-heat ${heatTone(heat)}` });
        heatBadge.innerHTML = `<span>热度 ${heat.toFixed(1)}</span>`;
        header.appendChild(heatBadge);

        if (IS_ADMIN) {
            const actions = el("div", { class: "topic-actions" });
            actions.appendChild(el("button", {
                text: "编辑",
                class: "topic-action-btn",
                onClick: (e) => editTopic(e, t.id, t.name)
            }));

            actions.appendChild(el("button", {
                text: "删除",
                class: "topic-action-btn danger",
                onClick: (e) => deleteTopic(e, t.id, t.name)
            }));
            actions.addEventListener("keydown", (e) => e.stopPropagation());
            header.appendChild(actions);
        }
        card.appendChild(header);

        const meta = el("div", { class: "meta" });
        meta.appendChild(el("span", { class: "topic-time-text", text: `开始 ${fmtTime(t.start_time)}` }));
        meta.appendChild(el("span", { class: "topic-time-text", text: `更新 ${fmtTime(t.updated_time)}` }));
        card.appendChild(meta);

        card.appendChild(el("div", { class: "topic-summary", text: t.summary || "暂无概览，等待更多相关新闻沉淀。" }));
        const footer = el("div", { class: "topic-card-footer" });
        footer.appendChild(el("span", { class: "topic-card-hint", text: "查看时间轴、综述与相关新闻" }));
        footer.appendChild(el("span", { class: "topic-arrow", text: "→" }));
        card.appendChild(footer);
        return card;
    }

    function renderStats(data) {
        const total = Number(data.total || 0);
        const loaded = document.getElementById("list").children.length;
        if (els.heroTotal) els.heroTotal.textContent = total;
        if (els.heroLoaded) els.heroLoaded.textContent = loaded;
        if (els.heroFilter) els.heroFilter.textContent = heatLabel(state.filter.minHeat);
    }

    async function loadTopics(page = 1) {
        if (isLoading) return;
        isLoading = true;

        const loading = document.getElementById("loading");
        const empty = document.getElementById("empty");
        const list = document.getElementById("list");
        const loadingMore = document.getElementById("loading-more");
        const noMoreData = document.getElementById("no-more-data");

        if (page === 1) {
            loading.classList.remove("is-hidden");
            empty.classList.add("is-hidden");
            empty.textContent = "暂无专题。请等待全流程任务运行后自动生成。";
            renderSkeletons(list);
            loadingMore.classList.add("is-hidden");
            noMoreData.classList.add("is-hidden");
            hasMore = true;
        } else {
            loadingMore.classList.remove("is-hidden");
        }

        try {
            const params = new URLSearchParams({
                page,
                size: pageSize,
                q: state.filter.q,
                sort_by: state.filter.sortBy,
                date: getEffectiveDate()
            });
            params.append("min_heat", state.filter.minHeat || "0");
            if (state.filter.category && state.filter.category !== "all") params.append("category", state.filter.category);
            if (state.filter.region) params.append("region", state.filter.region);
            if (state.filter.dateOption === "other") {
                if (state.filter.startDate) params.append("start_date", state.filter.startDate);
                if (state.filter.endDate) params.append("end_date", state.filter.endDate);
            }
            const resp = await fetch(`/api/topics/list?${params.toString()}`, { method: "GET", cache: "no-store" });
            if (!resp.ok) throw new Error("请求失败");
            const data = await resp.json();
            const items = (data && data.items) || [];

            if (page === 1) {
                loading.classList.add("is-hidden");
                if (!items.length) {
                    list.innerHTML = "";
                    renderStats(data);
                    empty.textContent = state.filter.q || Number(state.filter.minHeat || 0) > 0 || state.filter.dateOption !== "all" || state.filter.category !== "all" || state.filter.region
                        ? "没有符合条件的专题。"
                        : "暂无专题。请等待全流程任务运行后自动生成。";
                    empty.classList.remove("is-hidden");
                    hasMore = false;
                    isLoading = false;
                    return;
                }
            } else {
                loadingMore.classList.add("is-hidden");
            }

            if (page === 1) list.innerHTML = "";

            if (items.length < pageSize) {
                hasMore = false;
                if (items.length > 0 || page > 1) {
                     noMoreData.classList.remove("is-hidden");
                }
            } else {
                hasMore = true;
            }

            for (const t of items) {
                list.appendChild(renderTopicCard(t));
            }
            renderStats(data);

            currentPage = page;

        } catch (e) {
            console.error(e);
            if (page === 1) {
                loading.classList.add("is-hidden");
                empty.classList.remove("is-hidden");
                empty.textContent = "加载失败，请稍后重试。";
            }
        } finally {
            isLoading = false;
        }
    }

    renderTimeOptions();
    bindFilterEvents();
    syncFilterUI();
    fetchTopicCategories();
    fetchTopicRegions();

    // 滚动到底部时继续加载专题。
    window.addEventListener('scroll', () => {
        if (isLoading || !hasMore) return;
        if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100) {
            loadTopics(currentPage + 1);
        }
    });

    loadTopics(1);
