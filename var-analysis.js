(function () {
    "use strict";

    const data = window.VAR_DATA;
    if (!data) return;

    const byId = id => document.getElementById(id);
    const percent = value => `${(value * 100).toFixed(1)}%`;
    const dollars = value => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(value);
    const compactDate = iso => new Date(`${iso}T00:00:00`).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
    const baseline = data.portfolio.investment;
    const historicalBacktest = data.backtests.historical;
    const parametricBacktest = data.backtests.parametric;

    byId("hero-breach-count").textContent = historicalBacktest.breaches;
    byId("hero-test-days").textContent = historicalBacktest.observations;
    byId("observation-count").textContent = data.period.observations.toLocaleString("en-US");

    function riskAt(confidence, method) {
        return data.risk[String(confidence)][method];
    }

    const risk95 = data.risk["0.95"];
    byId("historical-var").textContent = dollars(risk95.historical.var_return * baseline);
    byId("parametric-var").textContent = dollars(risk95.parametric.var_return * baseline);
    byId("monte-carlo-var").textContent = dollars(risk95.monte_carlo.var_return * baseline);
    byId("tail-var").textContent = dollars(risk95.historical.var_return * baseline);
    byId("tail-es").textContent = dollars(risk95.historical.expected_shortfall_return * baseline);
    byId("max-drawdown").textContent = percent(Math.abs(data.performance.drawdown.max_drawdown));

    byId("historical-breach-rate").textContent = percent(historicalBacktest.breach_rate);
    byId("historical-breaches").textContent = historicalBacktest.breaches;
    byId("historical-kupiec").textContent = `Kupiec p-value: ${historicalBacktest.kupiec.p_value.toFixed(3)}`;
    byId("parametric-breach-rate").textContent = percent(parametricBacktest.breach_rate);
    byId("parametric-breaches").textContent = parametricBacktest.breaches;
    byId("parametric-kupiec").textContent = `Kupiec p-value: ${parametricBacktest.kupiec.p_value.toFixed(3)}`;

    function renderAllocation() {
        const colors = ["#244d6d", "#3b6a88", "#6e8796", "#9c4d43", "#b33b32", "#87947c", "#b99555"];
        byId("allocation-bar").innerHTML = data.portfolio.assets.map((asset, index) =>
            `<span class="allocation-segment" style="width:${data.portfolio.weights[index] * 100}%;background:${colors[index]}" title="${asset}: ${percent(data.portfolio.weights[index])}"></span>`
        ).join("");
        byId("allocation-key").innerHTML = data.portfolio.assets.map((asset, index) =>
            `<span><i style="background:${colors[index]}"></i>${asset} ${percent(data.portfolio.weights[index])}</span>`
        ).join("");
    }

    function renderWorstDays() {
        byId("worst-days").innerHTML = data.worst_days.map(day =>
            `<div class="worst-day"><span>${compactDate(day.date)}</span><span>${percent(day.return)}</span></div>`
        ).join("");
    }

    function renderDistribution() {
        const width = 820;
        const height = 390;
        const margin = { top: 24, right: 24, bottom: 54, left: 58 };
        const returns = data.returns;
        const min = Math.min(...returns);
        const max = Math.max(...returns);
        const bins = 42;
        const binWidth = (max - min) / bins;
        const counts = Array(bins).fill(0);
        returns.forEach(value => {
            const index = Math.min(bins - 1, Math.floor((value - min) / binWidth));
            counts[index] += 1;
        });
        const threshold = -risk95.historical.var_return;
        const yMax = Math.max(...counts);
        const x = value => margin.left + ((value - min) / (max - min)) * (width - margin.left - margin.right);
        const y = value => height - margin.bottom - value / yMax * (height - margin.top - margin.bottom);
        const bars = counts.map((count, index) => {
            const left = min + index * binWidth;
            const right = left + binWidth;
            const fill = right <= threshold ? "#b33b32" : "#7890a0";
            return `<rect x="${x(left) + 1}" y="${y(count)}" width="${Math.max(1, x(right) - x(left) - 2)}" height="${height - margin.bottom - y(count)}" fill="${fill}"/>`;
        }).join("");
        const grid = [];
        for (let index = 0; index <= 4; index += 1) {
            const value = min + (max - min) * index / 4;
            grid.push(`<line x1="${x(value)}" y1="${margin.top}" x2="${x(value)}" y2="${height - margin.bottom}" stroke="#dedbd2"/>`);
            grid.push(`<text x="${x(value)}" y="${height - 25}" text-anchor="middle" fill="#68727a" font-size="12">${percent(value)}</text>`);
        }
        byId("distribution-chart").innerHTML = `
            <svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="distribution-title distribution-desc">
                <title id="distribution-title">Distribution of daily portfolio returns</title>
                <desc id="distribution-desc">A histogram of 1,064 daily returns. Observations below the 95 percent historical Value at Risk threshold are red.</desc>
                ${grid.join("")}
                ${bars}
                <line x1="${x(threshold)}" y1="${margin.top}" x2="${x(threshold)}" y2="${height - margin.bottom}" stroke="#8b2f2a" stroke-width="2" stroke-dasharray="6 5"/>
                <text x="${x(threshold) + 7}" y="${margin.top + 15}" fill="#8b2f2a" font-size="12" font-weight="600">95% VaR</text>
                <text x="${width / 2}" y="${height - 3}" text-anchor="middle" fill="#42515c" font-size="13">Daily portfolio return</text>
            </svg>`;
    }

    function renderBacktest() {
        const series = historicalBacktest.series;
        const width = 820;
        const height = 390;
        const margin = { top: 24, right: 24, bottom: 55, left: 62 };
        const maxLoss = Math.max(...series.map(point => Math.max(point.loss, point.var)), 0.04);
        const x = index => margin.left + index / (series.length - 1) * (width - margin.left - margin.right);
        const y = value => height - margin.bottom - Math.max(0, value) / maxLoss * (height - margin.top - margin.bottom);
        const path = series.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point.var).toFixed(1)}`).join(" ");
        const breaches = series.map((point, index) => point.breach ? `<circle cx="${x(index)}" cy="${y(point.loss)}" r="3.8" fill="#b33b32"/>` : "").join("");
        const grid = [];
        for (let index = 0; index <= 4; index += 1) {
            const value = maxLoss * index / 4;
            grid.push(`<line x1="${margin.left}" y1="${y(value)}" x2="${width - margin.right}" y2="${y(value)}" stroke="#dedbd2"/>`);
            grid.push(`<text x="${margin.left - 10}" y="${y(value) + 4}" text-anchor="end" fill="#68727a" font-size="12">${percent(value)}</text>`);
        }
        const dateIndexes = [0, Math.floor(series.length / 3), Math.floor(series.length * 2 / 3), series.length - 1];
        const dateLabels = dateIndexes.map(index => `<text x="${x(index)}" y="${height - 25}" text-anchor="middle" fill="#68727a" font-size="12">${new Date(`${series[index].date}T00:00:00`).getFullYear()}</text>`).join("");
        byId("backtest-chart").innerHTML = `
            <svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="backtest-svg-title backtest-svg-desc">
                <title id="backtest-svg-title">Rolling historical Value at Risk and breaches</title>
                <desc id="backtest-svg-desc">The Value at Risk threshold changes through time based on the previous 252 returns. Red points show the 47 days when the following loss exceeded the threshold.</desc>
                ${grid.join("")}
                <path d="${path}" fill="none" stroke="#193b57" stroke-width="2.5"/>
                ${breaches}
                ${dateLabels}
                <text x="15" y="${height / 2}" text-anchor="middle" fill="#42515c" font-size="13" transform="rotate(-90 15 ${height / 2})">One-day loss</text>
            </svg>`;
    }

    let confidence = 0.95;
    const investmentSlider = byId("investment-slider");

    function updateCalculator() {
        const investment = Number(investmentSlider.value);
        byId("investment-value").textContent = dollars(investment);
        const methods = {
            historical: ["calc-historical-var", "calc-historical-es"],
            parametric: ["calc-parametric-var", "calc-parametric-es"],
            monte_carlo: ["calc-monte-var", "calc-monte-es"]
        };
        Object.entries(methods).forEach(([method, ids]) => {
            const result = riskAt(confidence, method);
            byId(ids[0]).textContent = dollars(result.var_return * investment);
            byId(ids[1]).textContent = dollars(result.expected_shortfall_return * investment);
        });
        byId("calculator-caption").textContent = `One-day loss estimates at ${confidence * 100}% confidence.`;
    }

    document.querySelectorAll("[data-confidence]").forEach(button => {
        button.addEventListener("click", () => {
            confidence = Number(button.dataset.confidence);
            document.querySelectorAll("[data-confidence]").forEach(item => item.classList.toggle("active", item === button));
            updateCalculator();
        });
    });
    investmentSlider.addEventListener("input", updateCalculator);

    renderAllocation();
    renderWorstDays();
    renderDistribution();
    renderBacktest();
    updateCalculator();
}());
