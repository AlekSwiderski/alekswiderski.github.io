(function () {
    "use strict";

    const data = window.PORTFOLIO_DATA;
    if (!data) return;

    const percent = value => `${(value * 100).toFixed(1)}%`;
    const number = value => value.toFixed(2);
    const byId = id => document.getElementById(id);

    byId("observation-count").textContent = data.period.observations.toLocaleString("en-US");

    function allocationRows(portfolio) {
        return portfolio.weights
            .map((weight, index) => ({ weight, ticker: data.tickers[index] }))
            .filter(item => item.weight >= 0.005)
            .sort((a, b) => b.weight - a.weight)
            .map(item => `<div class="allocation-row"><span>${item.ticker}</span><strong>${percent(item.weight)}</strong></div>`)
            .join("");
    }

    function fillPortfolio(prefix, portfolio) {
        byId(`${prefix}-return`).textContent = percent(portfolio.return);
        byId(`${prefix}-risk`).textContent = percent(portfolio.volatility);
        const allocation = byId(`${prefix}-allocation`);
        if (allocation) allocation.innerHTML = allocationRows(portfolio);
    }

    fillPortfolio("max", data.max_sharpe);
    fillPortfolio("min", data.min_volatility);
    fillPortfolio("equal", data.equal_weight);

    function renderAssetTable() {
        byId("asset-table-body").innerHTML = data.tickers.map((ticker, index) => `
            <tr>
                <td class="ticker">${ticker}</td>
                <td>${data.names[index]}</td>
                <td>${percent(data.annual_returns[index])}</td>
                <td>${percent(data.annual_volatility[index])}</td>
            </tr>`).join("");
    }

    function renderCorrelation() {
        const cells = ['<div class="matrix-cell matrix-label" aria-hidden="true"></div>'];
        data.tickers.forEach(ticker => cells.push(`<div class="matrix-cell matrix-label">${ticker}</div>`));
        data.correlation.forEach((row, rowIndex) => {
            cells.push(`<div class="matrix-cell matrix-label">${data.tickers[rowIndex]}</div>`);
            row.forEach(value => {
                const strength = Math.max(0, Math.min(1, value));
                const alpha = 0.07 + strength * 0.58;
                const textColor = strength > 0.66 ? "#ffffff" : "#172432";
                cells.push(`<div class="matrix-cell" style="background:rgba(47,100,142,${alpha});color:${textColor}">${value.toFixed(2)}</div>`);
            });
        });
        byId("correlation-matrix").innerHTML = cells.join("");
    }

    function renderFrontier() {
        const width = 820;
        const height = 430;
        const margin = { top: 30, right: 42, bottom: 58, left: 68 };
        const allPoints = [
            ...data.frontier.map(point => ({ x: point.volatility, y: point.return })),
            ...data.annual_returns.map((value, index) => ({ x: data.annual_volatility[index], y: value }))
        ];
        const xMin = Math.floor(Math.min(...allPoints.map(point => point.x)) * 100) / 100 - 0.01;
        const xMax = Math.ceil(Math.max(...allPoints.map(point => point.x)) * 100) / 100 + 0.01;
        const yMin = 0;
        const yMax = Math.ceil(Math.max(...allPoints.map(point => point.y)) * 100) / 100 + 0.01;
        const x = value => margin.left + ((value - xMin) / (xMax - xMin)) * (width - margin.left - margin.right);
        const y = value => height - margin.bottom - ((value - yMin) / (yMax - yMin)) * (height - margin.top - margin.bottom);
        const ticks = 5;
        const grid = [];

        for (let index = 0; index <= ticks; index += 1) {
            const xValue = xMin + (xMax - xMin) * index / ticks;
            const yValue = yMin + (yMax - yMin) * index / ticks;
            grid.push(`<line x1="${x(xValue)}" y1="${margin.top}" x2="${x(xValue)}" y2="${height - margin.bottom}" stroke="#ded9cf"/>`);
            grid.push(`<text x="${x(xValue)}" y="${height - 28}" text-anchor="middle" fill="#66717b" font-size="12">${percent(xValue)}</text>`);
            grid.push(`<line x1="${margin.left}" y1="${y(yValue)}" x2="${width - margin.right}" y2="${y(yValue)}" stroke="#ded9cf"/>`);
            grid.push(`<text x="${margin.left - 12}" y="${y(yValue) + 4}" text-anchor="end" fill="#66717b" font-size="12">${percent(yValue)}</text>`);
        }

        const frontierPath = data.frontier.map((point, index) => `${index ? "L" : "M"}${x(point.volatility).toFixed(1)},${y(point.return).toFixed(1)}`).join(" ");
        const assets = data.tickers.map((ticker, index) => `
            <g>
                <circle cx="${x(data.annual_volatility[index])}" cy="${y(data.annual_returns[index])}" r="5" fill="#2f648e" stroke="#fbfaf7" stroke-width="2"/>
                <text x="${x(data.annual_volatility[index]) + 8}" y="${y(data.annual_returns[index]) - 8}" fill="#17324d" font-size="12" font-weight="600">${ticker}</text>
            </g>`).join("");
        const markers = [
            { label: "Lowest risk", portfolio: data.min_volatility },
            { label: "Highest Sharpe", portfolio: data.max_sharpe },
            { label: "Equal weight", portfolio: data.equal_weight }
        ].map((item, index) => {
            const fill = index === 2 ? "#a85132" : "#17324d";
            const dx = index === 0 ? 10 : -10;
            const anchor = index === 0 ? "start" : "end";
            return `<g>
                <circle cx="${x(item.portfolio.volatility)}" cy="${y(item.portfolio.return)}" r="7" fill="${fill}" stroke="#fbfaf7" stroke-width="3"/>
                <text x="${x(item.portfolio.volatility) + dx}" y="${y(item.portfolio.return) + 20}" text-anchor="${anchor}" fill="#172432" font-size="12" font-weight="600">${item.label}</text>
            </g>`;
        }).join("");

        byId("frontier-chart").innerHTML = `
            <svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="frontier-svg-title frontier-svg-desc">
                <title id="frontier-svg-title">Efficient frontier for seven ETFs</title>
                <desc id="frontier-svg-desc">Annualised volatility is shown on the horizontal axis and annualised arithmetic return on the vertical axis. QQQ has the highest individual return. The maximum Sharpe portfolio lies between QQQ and gold.</desc>
                ${grid.join("")}
                <path d="${frontierPath}" fill="none" stroke="#a85132" stroke-width="3"/>
                ${assets}
                ${markers}
                <text x="${width / 2}" y="${height - 4}" text-anchor="middle" fill="#43515d" font-size="13">Annualised volatility</text>
                <text x="16" y="${height / 2}" text-anchor="middle" fill="#43515d" font-size="13" transform="rotate(-90 16 ${height / 2})">Annualised return</text>
            </svg>`;
    }

    let weights = [...data.equal_weight.weights];

    function portfolioMetrics(currentWeights) {
        const expectedReturn = currentWeights.reduce((sum, weight, index) => sum + weight * data.annual_returns[index], 0);
        let variance = 0;
        currentWeights.forEach((leftWeight, leftIndex) => {
            currentWeights.forEach((rightWeight, rightIndex) => {
                variance += leftWeight * rightWeight * data.covariance[leftIndex][rightIndex];
            });
        });
        const volatility = Math.sqrt(variance);
        return {
            return: expectedReturn,
            volatility,
            sharpe: (expectedReturn - data.method.risk_free_rate) / volatility
        };
    }

    function rebalance(changedIndex, newWeight) {
        const chosen = Math.max(0, Math.min(1, newWeight));
        const remaining = 1 - chosen;
        const otherTotal = weights.reduce((sum, weight, index) => index === changedIndex ? sum : sum + weight, 0);

        if (otherTotal > 0) {
            weights = weights.map((weight, index) => index === changedIndex ? chosen : weight * remaining / otherTotal);
        } else {
            const share = remaining / (weights.length - 1);
            weights = weights.map((weight, index) => index === changedIndex ? chosen : share);
        }
        renderCalculator();
    }

    function renderCalculator() {
        const controls = byId("weight-controls");
        if (!controls.children.length) {
            controls.innerHTML = data.tickers.map((ticker, index) => `
                <label class="weight-control">
                    <span class="weight-label"><span>${ticker}</span><small>${data.names[index]}</small></span>
                    <input type="range" min="0" max="100" step="0.1" value="${weights[index] * 100}" data-weight-index="${index}" aria-label="${ticker} allocation">
                    <output class="weight-value" id="weight-value-${index}">${percent(weights[index])}</output>
                </label>`).join("");
            controls.querySelectorAll("input").forEach(input => {
                input.addEventListener("input", event => rebalance(Number(event.target.dataset.weightIndex), Number(event.target.value) / 100));
            });
        }

        weights.forEach((weight, index) => {
            const input = controls.querySelector(`[data-weight-index="${index}"]`);
            input.value = (weight * 100).toFixed(1);
            byId(`weight-value-${index}`).textContent = percent(weight);
        });

        const metrics = portfolioMetrics(weights);
        byId("custom-return").textContent = percent(metrics.return);
        byId("custom-risk").textContent = percent(metrics.volatility);
        byId("custom-sharpe").textContent = number(metrics.sharpe);
        byId("custom-total").textContent = percent(weights.reduce((sum, weight) => sum + weight, 0));
    }

    document.querySelectorAll("[data-preset]").forEach(button => {
        button.addEventListener("click", () => {
            weights = [...data[button.dataset.preset].weights];
            renderCalculator();
        });
    });

    renderAssetTable();
    renderCorrelation();
    renderFrontier();
    renderCalculator();
}());
