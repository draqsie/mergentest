import time
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from mergen_core import MergenNeuromorphicEngine

console = Console()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_tui():
    clear_screen()
    console.print("[dim]MiniLM Model Yükleniyor... İlk açılışta lokal dosyalar indirilebilir...[/dim]")
    engine = MergenNeuromorphicEngine()
    
    clear_screen()
    console.print("[bold cyan]MERGEN RAG + LIF NÖRON + SAF NÖROMORFİK MİMARİ[/bold cyan]\n")
    user_name = console.input("[dim]Ağa Bağlanan Biyolojik Gözlemci: [/dim]").strip() or "Kullanıcı"

    clear_screen()
    console.print(f"[bold]Bağlantı Aktif:[/bold] {user_name}\n[dim]Sistem tamamen lokaldir. API kullanılmıyor.[/dim]\n")

    while True:
        try:
            query = console.input(f"\n[bold green]{user_name} >[/bold green] ")
            if query.lower() in ['q', 'exit', 'çıkış']:
                break
            if not query:
                continue

            with console.status("[dim]Vektörler kodlanıyor, nöronlar ateşleniyor...[/dim]", spinner="bouncingBar"):
                result = engine.process(query, user_name)

            # Yanıt
            console.print(f"\n[bold white]MERGEN >[/bold white] {result['response']}\n")
            
            # Teknik Metrikler
            duygu = result["duygu"]
            esik = result["esik"]
            snapshot = engine.snapshot()
            renk = "blue"
            if "Bilinçli" in duygu: renk = "cyan"
            elif "Refleks" in duygu: renk = "yellow"
            elif "Bilinçaltı" in duygu: renk = "white"
            elif "Rüya" in duygu: renk = "magenta"
            elif "Uyku" in duygu: renk = "magenta"

            neu = snapshot["neuromodulators"]
            meta = (
                f"[dim]Ağ Durumu: [{renk}]{duygu}[/{renk}] | "
                f"Eşik: {esik:.2f} | Zaman: {snapshot['time']:.3f}s | Kortikal Ateş: {result['cortical_spikes']} | "
                f"Anlam/Çelişki: {result['anlam_spikes']}/{result['celiski_spikes']} | "
                f"Farkındalık: {snapshot['awareness_level']:.2f} | Uyku: {snapshot['sleep_stage']} ({snapshot['sleep_pressure']:.2f}) | "
                f"Eylem: {snapshot['motor_action']} | Ödül: {snapshot['reward_trace']:.2f} | "
                f"DA/ACh/5-HT/NE: {neu['dopamine']:.2f}/{neu['acetylcholine']:.2f}/{neu['serotonin']:.2f}/{neu['norepinephrine']:.2f} | "
                f"dt: {result['latency']:.2f}s[/dim]"
            )
            console.print(meta)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_tui()
