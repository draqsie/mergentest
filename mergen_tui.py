import os

from rich.console import Console
from rich.panel import Panel

from mergen_core import MergenNeuromorphicEngine

console = Console()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_startup_banner():
    console.print("[dim]Mergen çekirdeği yükleniyor. İlk açılışta yerel durum dosyaları okunabilir.[/dim]")
    console.print(
        Panel(
            "[bold cyan]MERGEN[/bold cyan]\n"
            "Komutlar: help, status, summary, memory, reset, sleep, exit\n"
            "Akış: girdi -> iç durum -> eylem -> öğrenme -> kayıt",
            title="Komut Paneli",
        )
    )


def render_result(result):
    console.print(f"\n[bold white]MERGEN >[/bold white] {result['response']}\n")

    action_feedback = result.get("action_feedback", {})
    memory_digest = result.get("memory_digest", {})
    session_summary = result.get("session_summary")
    journal_summary = result.get("journal_summary")

    meta = (
        f"[dim]Durum: {result['duygu']} | "
        f"Eşik: {result['esik']:.2f} | "
        f"Zaman: {result['time']:.3f}s | "
        f"Kortikal Ateş: {result['cortical_spikes']} | "
        f"Farkındalık: {memory_digest.get('summary', 'yok')} | "
        f"Eylem: {action_feedback.get('action', result['motor_plan'].get('action', 'bilinmiyor'))} | "
        f"Sonuç: {action_feedback.get('outcome', 'yok')} | "
        f"Ödül: {result['reward']:.2f} | "
        f"dt: {result['latency']:.2f}s[/dim]"
    )
    console.print(meta)

    if session_summary:
        console.print(Panel(session_summary, title="Oturum Özeti", expand=False))
    if journal_summary:
        console.print(Panel(journal_summary, title="Plastisite Günlüğü", expand=False))


def run_tui():
    clear_screen()
    print_startup_banner()
    engine = MergenNeuromorphicEngine()
    user_name = console.input("\n[dim]Ağa bağlanan biyolojik gözlemci: [/dim]").strip() or "Kullanıcı"

    clear_screen()
    console.print(f"[bold]Bağlantı aktif:[/bold] {user_name}\n[dim]Sistem yerel çalışıyor. Komutlar için help yaz.[/dim]\n")

    while True:
        try:
            query = console.input(f"\n[bold green]{user_name} >[/bold green] ").strip()
            if query.lower() in {"q", "exit", "çıkış"}:
                break
            if not query:
                continue

            with console.status("[dim]Vektörler kodlanıyor, devreler güncelleniyor...[/dim]", spinner="bouncingBar"):
                result = engine.process(query, user_name)

            render_result(result)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    run_tui()
