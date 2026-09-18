(function () {
    const plot = { x: 70, y: 22, width: 800, height: 190 };
    let currentPeriod = "live";

    function pointsFor(series, min, max, rangeStart, rangeEnd) {
        return series.map(({ time, value }) => {
            const x = plot.x + ((time - rangeStart) / (rangeEnd - rangeStart)) * plot.width;
            const bounded = Math.max(min, Math.min(max, value));
            const y = plot.y + plot.height - ((bounded - min) / (max - min)) * plot.height;
            return [x, y];
        });
    }

    function updateTimeAxis(chart, rangeStart, rangeEnd) {
        const svg = chart.querySelector("svg");
        if (!svg) return;
        svg.querySelectorAll("text.axis-label").forEach((label) => {
            if (Number(label.getAttribute("y")) > plot.y + plot.height + 10) label.remove();
        });

        const intervals = currentPeriod === "week" ? 7 : 6;
        const formatter = new Intl.DateTimeFormat("pt-BR", currentPeriod === "week"
            ? { day: "2-digit", month: "2-digit" }
            : { hour: "2-digit", minute: "2-digit", hour12: false });
        for (let index = 0; index <= intervals; index++) {
            const fraction = index / intervals;
            const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
            label.setAttribute("class", "axis-label");
            label.setAttribute("x", (plot.x + fraction * plot.width).toFixed(1));
            label.setAttribute("y", String(plot.y + plot.height + 30));
            label.setAttribute("text-anchor", index === 0 ? "start" : index === intervals ? "end" : "middle");
            label.textContent = formatter.format(new Date(rangeStart + fraction * (rangeEnd - rangeStart)));
            svg.appendChild(label);
        }
    }

    function updateCharts(payload) {
        const samples = Array.isArray(payload)
            ? (currentPeriod === "live" ? payload : [])
            : (currentPeriod === "live" ? payload.samples : payload[currentPeriod]);
        if (!Array.isArray(samples)) return;
        const rangeEnd = Date.now();
        const hours = currentPeriod === "live" ? 1 : currentPeriod === "today" ? 24 : 24 * 7;
        const rangeStart = rangeEnd - hours * 60 * 60 * 1000;

        document.querySelectorAll(".live-chart").forEach((chart) => {
            const key = chart.dataset.key;
            const min = Number(chart.dataset.min);
            const max = Number(chart.dataset.max);
            const unit = chart.dataset.unit || "";
            const series = samples
                .map((sample) => ({ time: Date.parse(sample.timestamp), value: Number(sample[key]) }))
                .filter(({ time, value }) => Number.isFinite(time) && Number.isFinite(value)
                    && time >= rangeStart && time <= rangeEnd)
                .sort((left, right) => left.time - right.time);
            const polyline = chart.querySelector(".metric-line");
            const lastPoint = chart.querySelector(".last-point");
            const lastLabel = chart.querySelector(".last-label");
            const current = chart.querySelector(".live-current");
            updateTimeAxis(chart, rangeStart, rangeEnd);

            if (series.length < 2) {
                polyline.setAttribute("points", "");
                lastPoint.setAttribute("r", "0");
                lastLabel.textContent = "";
                current.textContent = "--";
                return;
            }

            const points = pointsFor(series, min, max, rangeStart, rangeEnd);

            polyline.setAttribute("points", points.map((point) => point.join(",")).join(" "));
            const last = series[series.length - 1].value;
            const [lastX, lastY] = points[points.length - 1];
            lastPoint.setAttribute("r", "8");
            lastPoint.setAttribute("cx", lastX.toFixed(1));
            lastPoint.setAttribute("cy", lastY.toFixed(1));
            lastLabel.setAttribute("x", (lastX - 10).toFixed(1));
            lastLabel.setAttribute("y", (lastY - 14).toFixed(1));
            lastLabel.textContent = `${last.toFixed(2)}${unit}`;
            current.textContent = `${last.toFixed(2)}${unit}`;
        });
    }

    function refresh() {
        fetch(`/edgebox_history.json?t=${Date.now()}`, { cache: "no-store" })
            .then((response) => response.json())
            .then(updateCharts)
            .catch(() => {});
    }

    document.querySelectorAll(".history-period").forEach((button) => {
        button.textContent = { live: "Última hora", today: "24 horas", week: "7 dias" }[button.dataset.period] || button.textContent;
        button.addEventListener("click", () => {
            currentPeriod = button.dataset.period;
            document.querySelectorAll(".history-period").forEach((item) => item.classList.remove("active"));
            button.classList.add("active");
            refresh();
        });
    });

    refresh();
    setInterval(refresh, 10000);
})();
