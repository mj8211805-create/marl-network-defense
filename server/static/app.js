// CyberMARL Frontend Operations Logic

let autoSimInterval = null;
let networkGraph = null;
let trainRewardChart = null;
let trainMitigationChart = null;
let benchmarkRadarChart = null;
let benchmarkBarChart = null;

// Tab Switching
function switchTab(tabId) {
  ['simulation', 'training', 'benchmark'].forEach(t => {
    const view = document.getElementById(`view-${t}`);
    const btn = document.getElementById(`tab-${t}`);
    if (view && btn) {
      if (t === tabId) {
        view.classList.remove('hidden');
        btn.classList.add('border-cyan-400', 'text-cyan-400');
        btn.classList.remove('border-transparent', 'text-slate-400');
      } else {
        view.classList.add('hidden');
        btn.classList.remove('border-cyan-400', 'text-cyan-400');
        btn.classList.add('border-transparent', 'text-slate-400');
      }
    }
  });

  if (tabId === 'simulation' && !networkGraph) {
    initTopologyGraph();
  }
}

// Initialize Vis.js Network Graph
async function initTopologyGraph() {
  const container = document.getElementById('topologyNetworkGraph');
  if (!container) return;

  try {
    const res = await fetch('/api/topology');
    const graphData = await res.json();
    renderTopology(graphData);
  } catch (err) {
    console.error('Failed to load topology:', err);
  }
}

function renderTopology(graphData) {
  const container = document.getElementById('topologyNetworkGraph');
  if (!container) return;

  const nodes = new vis.DataSet(graphData.nodes.map(n => ({
    id: n.id,
    label: n.label,
    color: { background: n.color, border: '#ffffff' },
    shape: n.id === 'gw-01' ? 'diamond' : (n.id.startsWith('dmz') || n.id.startsWith('srv') ? 'box' : 'dot'),
    size: 18,
    font: { color: '#090d16', size: 10, face: 'monospace' }
  })));

  const edges = new vis.DataSet(graphData.edges.map(e => ({
    from: e.from,
    to: e.to,
    color: { color: '#334155' },
    arrows: 'to'
  })));

  const options = {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -40, centralGravity: 0.01, springLength: 70 }
    },
    interaction: { hover: true }
  };

  networkGraph = new vis.Network(container, { nodes, edges }, options);
}

// Update Defense Model Label
function updateDefenseAlgorithm() {
  const select = document.getElementById('defenseAlgoSelect');
  const labelMap = {
    'marl': 'Cooperative MARL (MADQN)',
    'single_rl': 'Centralized Single RL',
    'supervised_ml': 'Supervised ML (Random Forest)',
    'anomaly_detector': 'Isolation Forest Anomaly',
    'rule_based': 'Traditional Rule-Based IDS'
  };
  document.getElementById('activeModelLabel').innerText = labelMap[select.value] || 'Cooperative MARL';
}

// Step Simulation Once
async function stepSimulation() {
  const method = document.getElementById('defenseAlgoSelect').value;

  try {
    const res = await fetch('/api/simulate/step', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ defense_method: method })
    });
    const data = await res.json();
    updateSimulationUI(data);
  } catch (err) {
    console.error('Step error:', err);
  }
}

// Toggle Auto Simulation
function toggleContinuousSimulation() {
  const btn = document.getElementById('toggleAutoSimBtn');
  const icon = document.getElementById('autoSimIcon');
  const text = document.getElementById('autoSimText');

  if (autoSimInterval) {
    clearInterval(autoSimInterval);
    autoSimInterval = null;
    icon.className = 'fa-solid fa-play';
    text.innerText = 'Start Auto Simulation';
    btn.classList.remove('from-rose-500', 'to-red-500');
    btn.classList.add('from-cyan-500', 'to-emerald-500');
  } else {
    autoSimInterval = setInterval(stepSimulation, 600);
    icon.className = 'fa-solid fa-pause';
    text.innerText = 'Pause Simulation';
    btn.classList.remove('from-cyan-500', 'to-emerald-500');
    btn.classList.add('from-rose-500', 'to-red-500');
  }
}

// Reset Environment Simulation
async function resetSimulation() {
  if (autoSimInterval) toggleContinuousSimulation();
  try {
    const res = await fetch('/api/simulate/reset', { method: 'POST' });
    const data = await res.json();
    initTopologyGraph();
    document.getElementById('simLogFeed').innerHTML = `<div class="text-cyan-400">Environment reset to pristine state.</div>`;
  } catch (e) {
    console.error(e);
  }
}

// Update UI with Step Telemetry
function updateSimulationUI(data) {
  // Update Matrix Action Cards
  const threatEl = document.getElementById('liveActiveThreat');
  const threatDet = document.getElementById('liveThreatDetails');
  
  threatEl.innerText = data.active_attack;
  if (data.active_attack === 'Benign') {
    threatEl.className = 'text-lg font-bold text-emerald-400 font-mono mt-1';
    threatDet.innerText = 'Normal enterprise baseline traffic';
  } else {
    threatEl.className = 'text-lg font-bold text-rose-400 font-mono mt-1';
    threatDet.innerText = data.mitigated ? 'MITIGATED BY DEFENSE AGENTS' : 'UNMITIGATED - DAMAGING ASSETS';
  }

  document.getElementById('livePerimeterAction').innerText = data.actions_executed.perimeter;
  document.getElementById('liveInternalAction').innerText = data.actions_executed.internal;
  document.getElementById('liveHostAction').innerText = data.actions_executed.host;

  // Header & Sidebar Stats
  const healthPercent = (data.health_ratio * 100).toFixed(1);
  document.getElementById('headerHealthQoS').innerText = `${healthPercent}%`;
  
  const mitRate = data.stats.total_attacks_generated > 0 
    ? ((data.stats.total_attacks_mitigated / data.stats.total_attacks_generated) * 100).toFixed(1)
    : '0.0';
  document.getElementById('headerMitigationRate').innerText = `${mitRate}%`;

  document.getElementById('metricSimStep').innerText = data.step;
  document.getElementById('metricMitigatedGen').innerText = `${data.stats.total_attacks_mitigated} / ${data.stats.total_attacks_generated}`;
  document.getElementById('metricTpfp').innerText = `${data.stats.true_positives} / ${data.stats.false_positives}`;
  document.getElementById('metricFn').innerText = data.stats.false_negatives;
  document.getElementById('metricStepReward').innerText = data.rewards.global.toFixed(2);

  // Update Topology Graph Nodes
  if (networkGraph && data.topology) {
    data.topology.nodes.forEach(n => {
      networkGraph.body.data.nodes.update({
        id: n.id,
        color: { background: n.color, border: '#ffffff' }
      });
    });
  }

  // Append to Action Log Feed
  const log = document.getElementById('simLogFeed');
  const line = document.createElement('div');
  line.className = 'text-[11px] font-mono flex justify-between border-b border-slate-800/40 pb-1';
  line.innerHTML = `
    <span>Step #${data.step}: <strong class="${data.active_attack==='Benign'?'text-emerald-400':'text-rose-400'}">${data.active_attack}</strong></span>
    <span class="text-cyan-400">[P:${data.actions_executed.perimeter} | I:${data.actions_executed.internal} | H:${data.actions_executed.host}]</span>
  `;
  log.prepend(line);
  if (log.children.length > 20) log.lastChild.remove();
}

// Start MARL Training Run
async function startMARLTraining() {
  const episodes = parseInt(document.getElementById('trainEpisodesSelect').value) || 100;
  const btn = document.getElementById('runTrainBtn');
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Training Policy...</span>`;

  try {
    const res = await fetch('/api/train/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ algorithm: 'marl', episodes: episodes, steps_per_episode: 40 })
    });
    const data = await res.json();
    renderTrainingCharts(data.history);
  } catch (err) {
    alert('Training error: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-dumbbell"></i> <span>Start Training Run</span>`;
  }
}

function renderTrainingCharts(history) {
  const epLabels = history.episode_rewards.map((_, i) => i + 1);

  // 1. Reward Curve
  const ctxReward = document.getElementById('trainRewardChart');
  if (trainRewardChart) trainRewardChart.destroy();
  trainRewardChart = new Chart(ctxReward, {
    type: 'line',
    data: {
      labels: epLabels,
      datasets: [{
        label: 'Episode Cumulative Reward',
        data: history.episode_rewards,
        borderColor: '#00ff9d',
        backgroundColor: 'rgba(0, 255, 157, 0.1)',
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
      },
      plugins: { legend: { labels: { color: '#f8fafc' } } }
    }
  });

  // 2. Mitigation Rate Curve
  const ctxMit = document.getElementById('trainMitigationChart');
  if (trainMitigationChart) trainMitigationChart.destroy();
  trainMitigationChart = new Chart(ctxMit, {
    type: 'line',
    data: {
      labels: epLabels,
      datasets: [{
        label: 'Mitigation Rate (%)',
        data: history.mitigation_rates.map(r => r * 100),
        borderColor: '#00f0ff',
        backgroundColor: 'rgba(0, 240, 255, 0.1)',
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
        y: { min: 0, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
      },
      plugins: { legend: { labels: { color: '#f8fafc' } } }
    }
  });
}

// Run Comparative Benchmark
async function runComparativeBenchmark() {
  const btn = document.getElementById('runBenchmarkBtn');
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> <span>Evaluating Baselines...</span>`;

  try {
    const res = await fetch('/api/benchmark/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ episodes: 10, steps_per_episode: 40 })
    });
    const report = await res.json();
    renderBenchmarkResults(report);
  } catch (e) {
    alert('Benchmark error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-bolt"></i> <span>Execute Full Benchmark</span>`;
  }
}

function renderBenchmarkResults(report) {
  const methods = [
    report.marl_system,
    report.single_rl_baseline,
    report.supervised_ml_baseline,
    report.anomaly_detector_baseline,
    report.rule_based_ids_baseline
  ];

  // Render Table Rows
  const tbody = document.getElementById('benchmarkTableBody');
  tbody.innerHTML = methods.map((m, i) => `
    <tr class="hover:bg-slate-800/40 transition-all ${i===0?'bg-emerald-950/20 font-bold':''}">
      <td class="p-3 ${i===0?'text-emerald-400':(i===1?'text-yellow-400':'text-cyan-400')}">${m.method_name}</td>
      <td class="p-3 text-white">${m.f1_score.toFixed(4)}</td>
      <td class="p-3 text-slate-300">${m.precision.toFixed(4)}</td>
      <td class="p-3 text-slate-300">${m.recall.toFixed(4)}</td>
      <td class="p-3 text-rose-400">${m.false_positive_rate.toFixed(1)}%</td>
      <td class="p-3 text-emerald-400">${m.network_availability_qos.toFixed(1)}%</td>
      <td class="p-3 text-slate-300">${m.mean_time_to_mitigate_steps.toFixed(2)}</td>
      <td class="p-3 text-slate-400">${m.avg_inference_latency_ms.toFixed(3)} ms</td>
    </tr>
  `).join('');

  // Render Radar Chart
  const radarCtx = document.getElementById('benchmarkRadarChart');
  if (benchmarkRadarChart) benchmarkRadarChart.destroy();
  benchmarkRadarChart = new Chart(radarCtx, {
    type: 'radar',
    data: {
      labels: ['F1-Score (%)', 'Precision (%)', 'Recall (%)', 'QoS Availability (%)', 'FP Suppression (%)'],
      datasets: [
        {
          label: 'Cooperative MARL',
          data: [report.marl_system.f1_score * 100, report.marl_system.precision * 100, report.marl_system.recall * 100, report.marl_system.network_availability_qos, 100 - report.marl_system.false_positive_rate],
          borderColor: '#00ff9d',
          backgroundColor: 'rgba(0, 255, 157, 0.2)'
        },
        {
          label: 'Single-Agent RL',
          data: [report.single_rl_baseline.f1_score * 100, report.single_rl_baseline.precision * 100, report.single_rl_baseline.recall * 100, report.single_rl_baseline.network_availability_qos, 100 - report.single_rl_baseline.false_positive_rate],
          borderColor: '#fbbf24',
          backgroundColor: 'rgba(251, 191, 36, 0.2)'
        },
        {
          label: 'Supervised ML',
          data: [report.supervised_ml_baseline.f1_score * 100, report.supervised_ml_baseline.precision * 100, report.supervised_ml_baseline.recall * 100, report.supervised_ml_baseline.network_availability_qos, 100 - report.supervised_ml_baseline.false_positive_rate],
          borderColor: '#00f0ff',
          backgroundColor: 'rgba(0, 240, 255, 0.2)'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          grid: { color: '#1e293b' },
          angleLines: { color: '#1e293b' },
          pointLabels: { color: '#94a3b8', font: { family: 'monospace', size: 10 } },
          ticks: { backdropColor: 'transparent', color: '#64748b' }
        }
      },
      plugins: { legend: { labels: { color: '#f8fafc', font: { family: 'monospace' } } } }
    }
  });

  // Render Bar Chart
  const barCtx = document.getElementById('benchmarkBarChart');
  if (benchmarkBarChart) benchmarkBarChart.destroy();
  benchmarkBarChart = new Chart(barCtx, {
    type: 'bar',
    data: {
      labels: methods.map(m => m.method_name),
      datasets: [
        {
          label: 'F1-Score (%)',
          data: methods.map(m => m.f1_score * 100),
          backgroundColor: 'rgba(0, 255, 157, 0.8)'
        },
        {
          label: 'Network Availability (%)',
          data: methods.map(m => m.network_availability_qos),
          backgroundColor: 'rgba(0, 240, 255, 0.8)'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } },
        y: { min: 0, max: 100, grid: { color: '#1e293b' }, ticks: { color: '#94a3b8' } }
      },
      plugins: { legend: { labels: { color: '#f8fafc' } } }
    }
  });
}

// On load
window.addEventListener('DOMContentLoaded', () => {
  initTopologyGraph();
  stepSimulation();
});
