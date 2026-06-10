// 本文件用于承载全站基础交互，包括主题切换、图片容错和应用名称刷新。
(function () {
    window.addEventListener("error", function (event) {
        if (event.target && event.target.tagName === "IMG") {
            if (event.target.dataset.failed) return;
            event.target.dataset.failed = "true";
            event.target.style.display = "none";
        }
    }, true);

    try {
        const key = "ts_theme";
        const btn = document.getElementById("themeToggle");
        if (btn) {
            function setLabel(theme) {
                const isDark = theme === "dark";
                btn.innerHTML = isDark
                    ? '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" aria-hidden="true"><circle cx="12" cy="12" r="4.2" stroke="currentColor" stroke-width="2"/><path d="M12 2.8v2.1M12 19.1v2.1M4.2 4.2l1.5 1.5M18.3 18.3l1.5 1.5M2.8 12h2.1M19.1 12h2.1M4.2 19.8l1.5-1.5M18.3 5.7l1.5-1.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
                    : '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" aria-hidden="true"><path d="M20.2 14.8A7.2 7.2 0 0 1 9.2 3.8 8.7 8.7 0 1 0 20.2 14.8Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>';
                btn.setAttribute("aria-label", isDark ? "切换为浅色模式" : "切换为深色模式");
                btn.setAttribute("title", btn.getAttribute("aria-label") || "");
            }

            function getCurrentTheme() {
                const theme = (document.documentElement.dataset.theme || "").trim();
                if (theme === "dark" || theme === "light") return theme;
                const prefersDark = !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
                return prefersDark ? "dark" : "light";
            }

            setLabel(getCurrentTheme());
            btn.addEventListener("click", function () {
                const nextTheme = getCurrentTheme() === "dark" ? "light" : "dark";
                document.documentElement.dataset.theme = nextTheme;
                try {
                    localStorage.setItem(key, nextTheme);
                } catch (error) {}
                setLabel(nextTheme);
            });
        }
    } catch (error) {}

    async function refreshAppName() {
        try {
            const response = await fetch("/api/app_info", { method: "GET", cache: "no-store" });
            if (!response.ok) return;
            const data = await response.json().catch(function () { return {}; });
            const appName = String(data.app_name || "").trim();
            if (!appName) return;

            const logoText = document.querySelector(".logo span:last-child");
            if (logoText && logoText.textContent !== appName) {
                logoText.textContent = appName;
            }

            const separatorIndex = document.title.indexOf(" - ");
            if (separatorIndex >= 0) {
                document.title = appName + document.title.slice(separatorIndex);
            } else {
                document.title = appName;
            }
        } catch (error) {}
    }

    refreshAppName();
})();
