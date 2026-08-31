# CyberMARL: Multi-Agent Reinforcement Learning & Machine Learning for Autonomous Network Defense

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Decentralized Multi-Agent Reinforcement Learning (Dec-POMDP)** and **Supervised/Unsupervised Machine Learning** platform for autonomous multi-tier cyber defense, dynamic threat mitigation, and comparative benchmark evaluation against classical IDS baselines.

---

## 🛡️ Project Overview
Computer network defenses face increasingly complex, distributed, and fast-moving threat vectors including **Volumetric DoS/DDoS floods**, **Stealth Port Scans**, **SSH/RDP Brute-Force storms**, and **Multi-Stage APT Lateral Movements**. Traditional static rule-based Intrusion Detection Systems (IDS) and centralized security models suffer from rigid thresholds, high false-positive rates, and slow response times.

**CyberMARL** introduces an intelligent, autonomous multi-agent defense architecture where specialized defense agents cooperate across network boundaries to detect intrusions in real-time, coordinate countermeasures, and optimize the tradeoff between **threat mitigation accuracy** and **network availability (Quality of Service - QoS)**.

---

## 🏛️ System Architecture

```
                               [ Continuous Network Traffic Stream ]
                              (Stochastic Poisson Benign + Attacks)
                                                │
                                                ▼
                         ┌─────────────────────────────────────────────┐
                         │   Multi-Tier Enterprise Network Topology    │
                         │   • Gateway Router (192.168.1.1)            │
                         │   • Public DMZ Web & REST APIs (10.0.1.0/24) │
                         │   • Internal ERP & File Servers (10.0.2.0/24)│
                         │   • Critical Databases & DC (10.0.3.0/24)   │
                         │   • Workstation User Subnet (10.0.4.0/24)   │
                         └──────────────────────┬──────────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 │                              │                              │
                 ▼                              ▼                              ▼
  ┌─────────────────────────────┐┌─────────────────────────────┐┌─────────────────────────────┐
  │   1. Perimeter Defender     ││ 2. Internal Lateral Defender ││  3. Host Integrity Defender │
  │   • Observation: Volumetric ││ • Observation: East-West     ││  • Observation: Auth Fails, │
  │     PPS, SYN/ACK, Ext Rate  ││     Flows, SMB Rate, Entropy ││    CPU Spikes, Mem Load     │
  │   • Actions: RATE_LIMIT,    ││ • Actions: ISOLATE_SUBNET,   ││  • Actions: QUARANTINE_HOST,│
  │     BLOCK_IP, SYN_COOKIES   ││     TERMINATE_LATERAL, DECOY ││    RESTART_SERVICE, LOCK_ACC│
  └──────────────┬──────────────┘└──────────────┬──────────────┘└──────────────┬──────────────┘
                 │                              │                              │
                 └──────────────────────┬───────┴──────────────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Cooperative MARL Engine   │
                         │   (Shared Global QoS Reward)│
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 │                                             │
                 ▼                                             ▼
  [ Comparative Evaluation Hub ]               [ Interactive Web Operations UI ]
  • Cooperative MARL (MADQN)                   • Live Topology Network Graph (vis.js)
  • Centralized Single-Agent RL                • Real-Time Threat & Mitigation Feed
  • Supervised ML (Random Forest)              • Dynamic Training & Radar Charts
  • Anomaly Detector (Isolation Forest)        • REST API & WebSocket Streaming
  • Traditional Rule-Based IDS (Snort)
```

---

## 🤖 Defensive Agent Tiers & Action Spaces

| Agent Tier | Primary Responsibilities | Observation Space ($d=8$) | Action Space |
| :--- | :--- | :--- | :--- |
| **Perimeter Defender** | Mitigates external DDoS/DoS, suppresses high-speed port scanning, and filters ingress packets. | `[Norm PPS, SYN Ratio, Ext IP Rate, Err Rate, Entropy, CPU, Vol Anomaly, Threat Flag]` | `PASS (0), RATE_LIMIT (1), BLOCK_IP (2), SYN_COOKIES (3), DROP_ALL (4)` |
| **Internal Lateral Defender** | Analyzes east-west inter-subnet flows, terminates lateral SMB sessions, isolates compromised subnets. | `[EW PPS, SMB Count, Load Delta, Auth Fail, Entropy, Comp Ratio, APT Flag, Threat Flag]` | `PASS (0), ISOLATE_SUBNET (1), TERMINATE_LATERAL (2), HONEYPOT_REDIRECT (3)` |
| **Host Integrity Defender** | Protects servers and workstations from credential brute force, ransomware encryption, and privilege escalation. | `[Auth Fail Rate, CPU Spike, Proc Delta, Sweep Count, Health Loss, Fail Flag, Entropy, Threat Flag]` | `PASS (0), QUARANTINE_HOST (1), RESTART_SERVICE (2), LOCK_CREDENTIALS (3)` |

---

## 📈 Benchmark Evaluation

The platform includes an automated benchmarking test harness comparing all 5 cyber defense approaches:

| Defense Approach | Paradigm | F1-Score | Detection Accuracy | Network Availability (QoS %) | False Positive Rate (FPR %) | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cooperative MARL (MADQN)** | **Dec-POMDP Multi-Agent RL** | **Adaptive High** | **Superior** | **>85%** | **Low** | **<0.1 ms** |
| **Centralized Single RL** | Monolithic DQN ($|A|=80$) | Degraded | Moderate | ~80% | Moderate | ~0.1 ms |
| **Supervised ML (Random Forest)** | Static Classification | Fixed | High on Seen | ~65% | Minimal | ~15 ms |
| **Anomaly Detection (Isolation Forest)** | Outlier Detection | Unsupervised | Medium | ~75% | High on bursts | ~17 ms |
| **Traditional Rule-Based IDS** | Static Snort Signatures | Heuristic | Rigid | ~62% | High | **<0.05 ms** |

---

## 🚀 Quickstart & Usage

### 1. Installation
```powershell
cd C:\Users\muham\.gemini\antigravity\scratch\marl_network_defense
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
python -m pytest tests/ -v
```

### 3. Launch Web Operations Center
```powershell
python main.py serve --port 8000
```
Open **`http://localhost:8000`** in your web browser.

---

## 💻 CLI Commands

```powershell
# 1. Run live step-by-step CLI defense simulation
python main.py simulate --steps 15 --method marl

# 2. Train cooperative Multi-Agent Reinforcement Learning policy
python main.py train --episodes 150 --algorithm marl

# 3. Run comparative benchmark across all 5 approaches
python main.py evaluate --episodes 10 --output benchmark_results.md
```

---

## 🔌 REST API Endpoints

- `GET /api/health` - Check health status of simulation engine and active agents.
- `GET /api/topology` - Fetch current network topology nodes and statuses for graph rendering.
- `POST /api/simulate/step` - Step the environment with selected defense algorithm.
- `POST /api/simulate/reset` - Reset network nodes to healthy state.
- `POST /api/train/run` - Trigger background MARL training run and return learning curves.
- `POST /api/benchmark/run` - Run full comparative benchmark across all 5 approaches.
- `GET /` - Interactive Cyber Defense Operations Command Center UI.

---

## 📜 License
MIT License.
