"""Modul für Visualisierungs- und Plot-Hilfsfunktionen"""
import io
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from src.fourier_series import fourier_series_coefficients, fourier_series_reconstruct


def plot_fourier_interactive(t: np.ndarray, dt: float, f: np.ndarray, R_max: int = 40):
    """
    Interaktiver Plot für die Fourier-Approximation mit Schieberegler.
    Nutzt widgets.Image statt widgets.Output – verhindert Duplikate in VS Code zu 100%.
    """
    # 1. Fourier-Koeffizienten vorberechnen
    a0, A_k, B_k = fourier_series_coefficients(t, dt, f, R_max)

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


def plot_dft_magnitude_spectrum(
    f: np.ndarray, 
    c_hat: np.ndarray | None = None, 
    title_prefix: str = "Testsignal"
):
    """
    Visualisiert ein zeitdiskretes Signal und dessen zugehöriges DFT-Betragsspektrum im 1x2-Subplot.

    Parameter:
    ----------
    f : np.ndarray
        Signal im Zeitbereich (z. B. f = np.array([0, 1, 2, 3]))
    c_hat : np.ndarray, optional
        Berechnete komplexe DFT-Koeffizienten. Falls None, wird dft(f) berechnet.
    title_prefix : str, optional
        Präfix für die Plot-Titel (Standard: "Testsignal").

    Rückgabe:
    ---------
    fig, (ax1, ax2) : tuple
        Matplotlib Figure- und Axes-Objekte.
    """
    f = np.asarray(f, dtype=float)
    N = len(f)
    t_indices = np.arange(N)
    k_indices = np.arange(N)

    if c_hat is None:
        from src.dft import dft
        c_hat = dft(f)

    magnitudes = np.abs(c_hat)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Plot 1: Zeitsignal f (Abtastwerte) ---
    markerline, stemlines, baseline = ax1.stem(t_indices, f, linefmt='b-', markerfmt='bo', basefmt='k-')
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, 'markersize', 8)
    ax1.set_title(rf'{title_prefix}: Signal $\mathbf{{f}}$ im Zeitbereich ($N={N}$)', fontsize=12)
    ax1.set_xlabel('Zeitindex $n$ (bzw. $t_n$)', fontsize=11)
    ax1.set_ylabel('Signalwert $f_n$', fontsize=11)
    ax1.set_xticks(t_indices)
    ax1.set_ylim(min(np.min(f) - 0.5, -0.3), max(np.max(f) + 0.5, 3.5))
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Betragsspektrum |c_hat| (Diskrete Frequenzfunktion) ---
    if N == 4 and np.allclose(f, [0, 1, 2, 3]):
        colors = ['tab:red', 'tab:blue', 'tab:purple', 'tab:blue']
        bars = ax2.bar(k_indices, magnitudes, color=colors, width=0.45, alpha=0.85, edgecolor='black', lw=1.2)
        ax2.axhline(np.sqrt(8), color='gray', linestyle='--', alpha=0.6, label=r'$\sqrt{8} \approx 2.83$')
        ax2.axhline(6.0, color='red', linestyle=':', alpha=0.5, label=r'Gleichanteil $|\hat{c}_0|=6$')

        labels = [
            '6 (DC / Offset)', 
            r'$\sqrt{8} \approx 2.83$' + '\n(1. Harmonische)', 
            '2 (Nyquist)', 
            r'$\sqrt{8} \approx 2.83$' + '\n(Spiegelfrequenz)'
        ]
        for bar, label in zip(bars, labels):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.12, label, ha='center', va='bottom', fontsize=9.5, fontweight='bold')

        ax2.set_ylim(0, 7.5)
        ax2.legend(loc='upper right')
    else:
        bars = ax2.bar(k_indices, magnitudes, color='tab:blue', width=0.45, alpha=0.85, edgecolor='black', lw=1.2)
        for bar, mag in zip(bars, magnitudes):
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.05 * (np.max(magnitudes) if np.max(magnitudes) > 0 else 1), f"{mag:.2f}", ha='center', va='bottom', fontsize=9)
        ax2.set_ylim(0, np.max(magnitudes) * 1.25 if np.max(magnitudes) > 0 else 1)

    ax2.set_title(rf'Betragsverlauf $|\hat{{c}}_k|$ der Frequenzen ($N={N}$)', fontsize=12)
    ax2.set_xlabel('Frequenzindex $k$', fontsize=11)
    ax2.set_ylabel(r'Betrag $|\hat{c}_k|$', fontsize=11)
    ax2.set_xticks(k_indices)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()