const FAVORITES_KEY = "crypto-favorites";
const REFRESH_INTERVAL_MS = 30000;

const els = {
    status: document.getElementById("status"),
    table: document.getElementById("cryptoTable"),
    body: document.getElementById("cryptoBody"),
    search: document.getElementById("searchBox"),
    allTab: document.getElementById("allTab"),
    favTab: document.getElementById("favTab"),
};

let allCoins = [];
let currentTab = "all";
let favorites = loadFavorites();

function loadFavorites() {
    try {
        return JSON.parse(localStorage.getItem(FAVORITES_KEY)) || [];
    } catch {
        return [];
    }
}

function saveFavorites() {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites));
}

function toggleFavorite(symbol) {
    const idx = favorites.indexOf(symbol);
    if (idx === -1) {
        favorites.push(symbol);
    } else {
        favorites.splice(idx, 1);
    }
    saveFavorites();
    renderCoins();
}
window.toggleFavorite = toggleFavorite;

async function fetchCoins() {
    try {
        const res = await fetch("https://api4.binance.com/api/v3/ticker/24hr");
        if (!res.ok) throw new Error("네트워크 응답이 올바르지 않습니다.");

        const data = await res.json();
        allCoins = data
            .filter((coin) => coin.symbol.endsWith("USDT") && parseFloat(coin.lastPrice) > 0)
            .sort((a, b) => a.symbol.localeCompare(b.symbol));

        els.status.classList.add("hidden");
        els.table.classList.remove("hidden");
        renderCoins();
    } catch (err) {
        els.status.textContent = `데이터를 불러오지 못했습니다: ${err.message}`;
    }
}

function getVisibleCoins() {
    const keyword = els.search.value.trim().toUpperCase();
    let visible = allCoins.filter((coin) => coin.symbol.includes(keyword));

    if (currentTab === "favorites") {
        visible = visible.filter((coin) => favorites.includes(coin.symbol));
    }
    return visible;
}

function formatNumber(value) {
    return Number(value).toLocaleString(undefined, { maximumFractionDigits: 8 });
}

function renderCoins() {
    const coins = getVisibleCoins();
    els.body.innerHTML = "";

    if (coins.length === 0) {
        els.body.innerHTML = `<tr><td colspan="6" class="empty">표시할 항목이 없습니다.</td></tr>`;
        return;
    }

    const rows = coins.map((coin) => {
        const changePercent = parseFloat(coin.priceChangePercent);
        const direction = changePercent >= 0 ? "up" : "down";
        const sign = changePercent >= 0 ? "+" : "";
        const isFav = favorites.includes(coin.symbol);

        return `
            <tr>
                <td>
                    <button class="fav-btn" onclick="toggleFavorite('${coin.symbol}')">
                        ${isFav ? "★" : "☆"}
                    </button>
                </td>
                <td class="symbol">${coin.symbol}</td>
                <td>${formatNumber(coin.lastPrice)}</td>
                <td class="${direction}">${sign}${changePercent.toFixed(2)}%</td>
                <td>${formatNumber(coin.highPrice)}</td>
                <td>${formatNumber(coin.lowPrice)}</td>
            </tr>
        `;
    });

    els.body.innerHTML = rows.join("");
}

function switchTab(tab) {
    currentTab = tab;
    els.allTab.classList.toggle("active", tab === "all");
    els.favTab.classList.toggle("active", tab === "favorites");
    renderCoins();
}
window.switchTab = switchTab;

els.search.addEventListener("input", renderCoins);

fetchCoins();
setInterval(fetchCoins, REFRESH_INTERVAL_MS);
