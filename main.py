"""Central CLI Entrypoint for CyberMARL Autonomous Network Defense Platform."""

import sys
from pathlib import Path

# Configure utf-8 encoding for Windows console if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import typer
import uvicorn
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import config
from environment.network_env import MultiAgentNetworkDefenseEnv
from agents.marl.multi_agent_dql import MultiAgentDQNSystem
from benchmark.train_runner import train_marl_system, train_single_agent_system
from benchmark.evaluator import CyberMARLEvaluator

app = typer.Typer(help="CyberMARL: Multi-Agent Reinforcement Learning for Autonomous Network Defense")
console = Console()


@app.command()
def train(
    episodes: int = typer.Option(100, help="Number of training episodes"),
    algorithm: str = typer.Option("marl", help="Algorithm: marl or single_rl"),
    steps: int = typer.Option(50, help="Steps per episode"),
    save_weights: Optional[str] = typer.Option(None, help="Directory to save trained weights")
):
    """Train autonomous RL defense agents in the simulated network environment."""
    console.print(Panel.fit(
        f"[bold cyan]CyberMARL Policy Training[/bold cyan]\n"
        f"Algorithm: [bold yellow]{algorithm.upper()}[/bold yellow] | Episodes: [bold green]{episodes}[/bold green] | Steps/Ep: [bold]{steps}[/bold]"
    ))

    if algorithm == "marl":
        console.print("[yellow][*][/yellow] Training cooperative Multi-Agent Deep Q-Learning system...")
        marl, history = train_marl_system(episodes=episodes, steps_per_episode=steps, save_weights_path=save_weights)
        final_rew = history["episode_rewards"][-1] if history["episode_rewards"] else 0.0
        final_mit = history["mitigation_rates"][-1] * 100.0 if history["mitigation_rates"] else 0.0
        final_qos = history["availability_scores"][-1] if history["availability_scores"] else 100.0

        console.print(f"[bold green][+][/bold green] Training Complete!")
        console.print(f"- Final Episode Cumulative Reward: [bold cyan]{final_rew:.2f}[/bold cyan]")
        console.print(f"- Final Attack Mitigation Rate: [bold green]{final_mit:.1f}%[/bold green]")
        console.print(f"- Network Availability (QoS): [bold green]{final_qos:.1f}%[/bold green]")
    else:
        console.print("[yellow][*][/yellow] Training centralized single-agent RL baseline...")
        single_agent, history = train_single_agent_system(episodes=episodes, steps_per_episode=steps)
        console.print("[bold green][+][/bold green] Single-Agent Training Complete!")


@app.command()
def evaluate(
    episodes: int = typer.Option(10, help="Evaluation test episodes per model"),
    output: Optional[str] = typer.Option(None, help="Save evaluation report to Markdown/JSON")
):
    """Run comparative benchmark evaluating MARL against Single-RL, Supervised ML, and Rule-Based IDS."""
    console.print(Panel.fit("[bold cyan]CyberMARL Comparative Benchmark Suite[/bold cyan]\n[dim]Evaluating 5 Autonomous Defense Approaches[/dim]"))

    evaluator = CyberMARLEvaluator()
    console.print("[yellow][*][/yellow] Running multi-episode rollouts across all defense paradigms...")
    report = evaluator.run_benchmark(episodes=episodes, steps_per_episode=50)

    # Render Rich Table
    table = Table(title="Comparative Cyber Defense Performance Matrix", show_header=True, header_style="bold magenta")
    table.add_column("Defense Approach", style="bold", width=30)
    table.add_column("F1-Score", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("FPR (%)", style="red", justify="right")
    table.add_column("QoS Availability", style="green", justify="right")
    table.add_column("MTTR", justify="right")
    table.add_column("Latency (ms)", justify="right")

    models = [
        report.marl_system,
        report.single_rl_baseline,
        report.supervised_ml_baseline,
        report.anomaly_detector_baseline,
        report.rule_based_ids_baseline
    ]

    for m in models:
        is_marl = m == report.marl_system
        name_style = "[bold green]" + m.method_name + "[/bold green]" if is_marl else m.method_name
        f1_style = f"[bold]{m.f1_score:.4f}[/bold]" if is_marl else f"{m.f1_score:.4f}"
        table.add_row(
            name_style,
            f1_style,
            f"{m.precision:.4f}",
            f"{m.recall:.4f}",
            f"{m.false_positive_rate:.1f}%",
            f"{m.network_availability_qos:.1f}%",
            f"{m.mean_time_to_mitigate_steps:.2f}",
            f"{m.avg_inference_latency_ms:.3f} ms"
        )

    console.print(table)
    console.print(f"\n[bold green][+][/bold green] {report.summary_analysis}")

    if output:
        out_p = Path(output)
        if out_p.suffix == ".md":
            out_p.write_text(evaluator.generate_markdown_table(report), encoding="utf-8")
        else:
            out_p.write_text(json.dumps(report.model_dump(mode="json"), indent=2), encoding="utf-8")
        console.print(f"[cyan]Results saved to {output}[/cyan]")


@app.command()
def simulate(
    steps: int = typer.Option(10, help="Number of steps to simulate"),
    method: str = typer.Option("marl", help="Defense method: marl, single_rl, supervised_ml, anomaly_detector, rule_based")
):
    """Run step-by-step CLI defense simulation."""
    console.print(Panel.fit(f"[bold cyan]Simulating Defense: {method.upper()}[/bold cyan]"))
    env = MultiAgentNetworkDefenseEnv(max_steps=steps)
    obs = env.reset()

    if method == "marl":
        model = MultiAgentDQNSystem()
    else:
        from agents.rule_baseline.snort_rules import RuleBasedIDSAgent
        model = RuleBasedIDSAgent()

    table = Table(title=f"Live Simulation Trace ({method})", show_header=True, header_style="bold cyan")
    table.add_column("Step", width=6)
    table.add_column("Threat", style="bold red")
    table.add_column("Perimeter Act", style="cyan")
    table.add_column("Internal Act", style="blue")
    table.add_column("Host Act", style="magenta")
    table.add_column("Mitigated?", style="bold")
    table.add_column("Health (QoS)", style="green")

    for st in range(steps):
        actions = model.select_actions(obs) if hasattr(model, "select_actions") else {"perimeter": 0, "internal": 0, "host": 0}
        next_obs, rew, done, info = env.step(actions)
        obs = next_obs

        p_names = ["PASS", "RATE_LIMIT", "BLOCK_IP", "SYN_COOKIES", "DROP_ALL"]
        i_names = ["PASS", "ISOLATE", "TERMINATE", "HONEYPOT"]
        h_names = ["PASS", "QUARANTINE", "RESTART", "LOCK_CREDS"]

        table.add_row(
            str(st+1),
            info["active_attack"],
            p_names[actions.get("perimeter", 0)],
            i_names[actions.get("internal", 0)],
            h_names[actions.get("host", 0)],
            "[green]YES[/green]" if info["mitigated"] else "[red]NO[/red]",
            f"{info['health_ratio']*100:.0f}%"
        )
        if done:
            break

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(config.HOST, help="Host to bind server"),
    port: int = typer.Option(config.PORT, help="Port to bind server")
):
    """Launch the CyberMARL Operations Center Web Dashboard and REST API."""
    console.print(Panel.fit(
        f"[bold cyan]Launching CyberMARL Defense Command Center[/bold cyan]\n"
        f"Server URL: [bold green]http://{host}:{port}[/bold green]\n"
        f"Interactive Web UI: [bold green]http://localhost:{port}[/bold green]\n"
        f"REST API Docs: [bold green]http://localhost:{port}/docs[/bold green]",
        border_style="cyan"
    ))
    uvicorn.run("server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
