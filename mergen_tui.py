import time
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from mergen_core import MergenPureBiologicalEngine

console = Console()
engine = MergenPureBiologicalEngine()

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def run_tui():
    clear_screen()
    console.print("[bold white]MERGEN[/bold white] [dim]v1.3 - Saf Korteks[/dim]\n")
    
    user_name = console.input("[dim]Gözlemci İmzası: [/dim]").strip() or "Kullanıcı"

    clear_screen()
    console.print(f"[dim]Bağlantı Kuruldu:[/dim] [bold]{user_name}[/bold]\n")

    while True:
        try:
            query = console.input(f"\n[bold]>[/bold] ")
            if query.lower() in ['q', 'exit', 'çıkış']:
                break
            if not query:
                continue

            with console.status("[dim]dt işleniyor...[/dim]", spinner="dots"):
                result = engine.process(query, user_name)

            # Yanıt
            console.print(f"\n[bold white]MERGEN:[/bold white] {result['response']}")
            
            # Ultra-Minimal Metrikler
            h_durum = result["homeostazi"]["anlam_durum"]
            esik = result["homeostazi"]["anlam_esik"]
            kavrayis = result["kavrayis"]
            
            durum_renk = "white"
            if "Stresli" in h_durum: durum_renk = "red"
            elif "Meraklı" in h_durum: durum_renk = "yellow"

            meta = f"[dim]Ağ: [{durum_renk}]{h_durum}[/{durum_renk}] | Eşik: {esik:.2f} | Kavrayış Şiddeti: %{kavrayis:.1f} | Süre: {result['latency']:.3f}s[/dim]"
            console.print(meta)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_tui()
