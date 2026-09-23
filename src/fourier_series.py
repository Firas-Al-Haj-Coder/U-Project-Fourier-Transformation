"""
Modul für reelle Fourier-Reihen (Fourier Series).

Enthält Funktionen zur numerischen Berechnung der Fourier-Koeffizienten (Analyse)
und zur Rekonstruktion / Synthese der Fourier-Summe.
"""
import numpy as np


def fourier_series_coefficients(t: np.ndarray, dt: float, f: np.ndarray, R: int) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Berechnet die Fourier-Koeffizienten a0, A_k und B_k über numerische Quadratur.
    
    Rückgabe:
    ---------
    a0 : float
        Gleichanteil (Projektion auf 1)
    A_k : np.ndarray der Länge R
        Cosinus-Koeffizienten
    B_k : np.ndarray der Länge R
        Sinus-Koeffizienten
    """
    
    # ⟨f, 1⟩ / ‖cos(0·t)‖² = (1/2π) ∫ f(t)·1 dt = 1/2 · [(1/π) ∫ f(t)·1 dt] = a0 / 2
    # a0 = (1/π) ∫ f(t)·1 dt ≈ ∑ f(t)·1·dt * (1/π)
    a0 = np.sum(f * np.ones_like(t)) * dt * (1 / np.pi)
    
    # Anzahl der Harmonischen/Moden (Frequenzbereich)
    # Aus wie vielen verschiedenen Wellen setzt sich das Signal (aus Sinus/Cos) zusammen? (Auflösung : R)
    # Baue die Hutfunktion aus den ersten R Schwingungsmoden zusammen – von der 1-fachen bis zur R-fachen Frequenz pro Periode
    A_k = np.zeros(R)
    B_k = np.zeros(R)
    
    # Schrittweises Aufsummieren der Moden k = 1 bis R
    for k in range(1, R + 1):
        # Feste Grundkreisfrequenz w0 = π/L = 1 (bzw. 2π/T = 1).
        # Die k-te Harmonische vervielfacht diese (die Frequenz der k-ten Harmonischen): ω_k = k · w0 = k → Phase θ(t) = k · 1 · t
        # cos_mode = np.cos( (k * np.pi / L) * t)
        cos_mode = np.cos(k * t) # hängt von der Phase ab
        sin_mode = np.sin(k * t)

        # Inneres Produkt (Skalarprodukt) - wie stark f in atkuellen $k$-fache Schwingung im Signal vertreten ist. - ein skalar a_k, b_k pro überlagerte Kreisdrehung mit k Schwingungen (pro Mode)
        a_k = np.sum(f * cos_mode) * dt * (1 / np.pi)
        b_k = np.sum(f * sin_mode) * dt * (1 / np.pi) # immer 0 - f_gerade * sin_ungerade Funktionen ist immer 0
        
        A_k[k - 1] = a_k
        B_k[k - 1] = b_k
        
    return a0, A_k, B_k


def fourier_series_reconstruct(t: np.ndarray, a0: float, A_k: np.ndarray, B_k: np.ndarray, R: int | None = None) -> np.ndarray:
    """
    Rekonstruiert das Signal f_FS(t) aus den Fourier-Koeffizienten bis zur Ordnung R.
    """
    if R is None:
        R = len(A_k)
        
    # Start mit a0 / 2 (Gleichanteil)
    fFS = (a0 / 2) * np.ones_like(t)
    
    for k in range(1, R + 1):
        cos_mode = np.cos(k * t)
        sin_mode = np.sin(k * t)
        # Summe stückweise aufbauen (Signal f annähern)
        fFS = fFS + ((A_k[k - 1] * cos_mode) + (B_k[k - 1] * sin_mode))
        
    return fFS

