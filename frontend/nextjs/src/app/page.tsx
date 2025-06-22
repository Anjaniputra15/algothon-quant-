"use client";
import React, { useState } from "react";
import Plot from "react-plotly.js";

const strategies = ["momentum", "mean_reversion", "ensemble"];

export default function Home() {
  const [strategy, setStrategy] = useState(strategies[0]);
  const [loading, setLoading] = useState(false);
  const [pnl, setPnl] = useState<number[]>([]);
  const [drawdown, setDrawdown] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Placeholder demo data
  const demoPnl = Array.from({ length: 100 }, (_, i) => Math.sin(i / 10) * 1000 + i * 10);
  const demoDrawdown = demoPnl.map((v, i, arr) => Math.min(0, v - Math.max(...arr.slice(0, i + 1))));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/backtest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy }),
      });
      const data = await res.json();
      // TODO: parse data.output to extract pnl/drawdown arrays
      setPnl(demoPnl);
      setDrawdown(demoDrawdown);
    } catch (err: any) {
      setError(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-900 text-gray-100 flex flex-col items-center justify-center p-8">
      <h1 className="text-3xl font-bold mb-6">Algothon Quant Backtester</h1>
      <form onSubmit={handleSubmit} className="flex flex-col items-center gap-4 mb-8">
        <label className="text-lg">Select Strategy:</label>
        <select
          className="bg-gray-800 text-gray-100 p-2 rounded"
          value={strategy}
          onChange={e => setStrategy(e.target.value)}
        >
          {strategies.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
        <button
          type="submit"
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded font-semibold"
          disabled={loading}
        >
          {loading ? "Running..." : "Run Backtest"}
        </button>
      </form>
      {error && <div className="text-red-400 mb-4">{error}</div>}
      <div className="w-full max-w-3xl bg-gray-800 rounded-lg p-6 shadow-lg">
        <h2 className="text-xl font-semibold mb-4">Cumulative P&L</h2>
        <Plot
          data={[
            {
              x: pnl.map((_, i) => i),
              y: pnl,
              type: "scatter",
              mode: "lines",
              name: "P&L",
              line: { color: "#22d3ee" },
            },
          ]}
          layout={{
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#f1f5f9" },
            xaxis: { title: "Day" },
            yaxis: { title: "Cumulative P&L" },
            autosize: true,
            height: 300,
          }}
          useResizeHandler
          style={{ width: "100%" }}
        />
        <h2 className="text-xl font-semibold mt-8 mb-4">Drawdown</h2>
        <Plot
          data={[
            {
              x: drawdown.map((_, i) => i),
              y: drawdown,
              type: "scatter",
              mode: "lines",
              name: "Drawdown",
              line: { color: "#f87171" },
            },
          ]}
          layout={{
            paper_bgcolor: "#1e293b",
            plot_bgcolor: "#1e293b",
            font: { color: "#f1f5f9" },
            xaxis: { title: "Day" },
            yaxis: { title: "Drawdown" },
            autosize: true,
            height: 300,
          }}
          useResizeHandler
          style={{ width: "100%" }}
        />
      </div>
    </main>
  );
} 