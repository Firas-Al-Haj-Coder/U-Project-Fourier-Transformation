"""Modul für Visualisierungs- und Plot-Hilfsfunktionen"""
import io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from src.fourier_series import fourier_series_coefficients, fourier_series_reconstruct


def plot_fourier_interactive(t: np.ndarray, f: np.ndarray, R_max: int = 40):
    """
    Interaktiver Plot für die Fourier-Approximation mit Schieberegler.
    Nutzt widgets.Image statt widgets.Output – verhindert Duplikate in VS Code zu 100%.
    """
    # 1. Fourier-Koeffizienten vorberechnen
    a0, A_k, B_k = fourier_series_coefficients(t, f, R_max)

    # 2. Slider erstellen
    slider = widgets.IntSlider(
        min=1, max=R_max, step=1, value=1, 
        description='Moden R:', continuous_update=False,
        layout=widgets.Layout(width='450px')
    )

    # 3. Bild-Widget erstellen (kein Output-Widget, kein Zell-Leck!)
    image_widget = widgets.Image(format='png', layout=widgets.Layout(width='100%'))

    # 4. Zeichenfunktion
    def _render(R: int):
        fFS = fourier_series_reconstruct(t, a0, A_k, B_k, R=R)
        mse = np.mean((f - fFS) ** 2)
        
        with plt.ioff():
            fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
            ax.plot(t, f, 'k--', lw=2.2, alpha=0.6, label=r'Original $f(t)$')
            ax.plot(t, fFS, color='crimson', lw=2.2, label=f'Fourier-Summe ($R = {R}$ Moden)')
            
            # Knickstellen markieren
            ax.axvline(0, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(-np.pi / 2, color='gray', linestyle=':', alpha=0.5)
            ax.axvline(np.pi / 2, color='gray', linestyle=':', alpha=0.5)
            
            ax.set_title(f'Fourier-Approximation mit $R = {R}$ Moden | Fehler (MSE): {mse:.2e}', fontsize=12)
            ax.set_xlabel(r'Zeit / Winkel $t$', fontsize=11)
            ax.set_ylabel(r'Amplitude $f(t)$', fontsize=11)
            ax.set_ylim(-0.15, 1.15)
            ax.set_xlim(t[0], t[-1])
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper right', framealpha=0.9)
            fig.tight_layout()
            
            # Direkt in Bytes umwandeln und an das Bild-Widget senden
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            
            # Bild aktualisieren (kein display, kein show!)
            image_widget.value = buf.getvalue()

    slider.observe(lambda change: _render(change['new']), names='value')

    # Initialen Plot einmal ins Bild laden
    _render(slider.value)

    # Nur den Slider und das Bild anzeigen
    display(widgets.VBox([slider, image_widget]))