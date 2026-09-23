"""
Modul für Diskrete Fouriertransformation (DFT) und Inverse DFT (IDFT).

Wird in Notebook 02 für die matrixbasierte und algorithmische Transformation
im Frequenzbereich verwendet.

Nomenklatur-Standard:
- t: Zeit-Array der Länge N mit Zeitschritten t[n] = t_n (Spalten)
- f: Zeitsignal der Länge N mit Elementen f[n] = f(t_n)
- k: Frequenz-Array der Länge N mit Frequenzindizes k = 0, ..., N-1 (Zeilen)
- c_hat: Frequenzkoeffizienten der Länge N mit Elementen c_hat[k]
- F_N: DFT-Matrix der Dimension N x N
"""
import numpy as np


def dft_matrix(N: int) -> np.ndarray:
    """
    Erzeugt die komplexe DFT-Matrix F_N der Dimension N x N.
    
    Einträge: F_N[k, n] = exp(-2pi * i * k * t[n] / N)
    - Zeilen k: Frequenz-Harmonische (k = 0, ..., N-1)
    - Spalten: Diskrete Zeitpunkte t_n über das Zeit-Array t = np.arange(N)
    """
    t = np.arange(N)  # Zeit-Array: t[n] = t_n für n = 0, ..., N-1
    k = np.arange(N).reshape((N, 1))  # Frequenzen k
    return np.exp(-2j * np.pi * k * t / N)


def dft(f: np.ndarray) -> np.ndarray:
    """
    Berechnet die Diskrete Fouriertransformation (DFT) eines Signals f.
    Entspricht dem Basiswechsel c_hat = F_N @ f (Analyse: Zeit -> Frequenz).
    Komplexität: O(N^2)
    
    Parameter:
    ----------
    f : np.ndarray (1D)
        Reelles oder komplexes Eingangssignal der Länge N im Zeitbereich.
        Elemente f[n] repräsentieren Messwerte zum Zeitpunkt t_n.
        
    Rückgabe:
    ---------
    c_hat : np.ndarray (1D, komplex)
        Frequenzkoeffizienten (Spektrum) der Länge N.
    """
    f = np.asarray(f, dtype=complex)
    N = len(f)
    F_N = dft_matrix(N)
    return np.dot(F_N, f)


def idft(c_hat: np.ndarray) -> np.ndarray:
    """
    Berechnet die Inverse Diskrete Fouriertransformation (IDFT).
    Entspricht der Synthese f = (1/N) * F_N^* @ c_hat (Frequenz -> Zeit).
    Komplexität: O(N^2)
    
    Parameter:
    ----------
    c_hat : np.ndarray (1D, komplex)
        Frequenzkoeffizienten der Länge N.
        
    Rückgabe:
    ---------
    f : np.ndarray (1D, komplex)
        Rekonstruiertes Signal der Länge N im Zeitbereich mit Einträgen f[n].
    """
    c_hat = np.asarray(c_hat, dtype=complex)
    N = len(c_hat)
    t = np.arange(N)  # Zeit-Array: t[n] = t_n für n = 0, ..., N-1
    k = t.reshape((N, 1))
    # IDFT nutzt das positive Vorzeichen im Exponenten: exp(+2pi*i*k*t / N)
    F_inv = np.exp(2j * np.pi * k * t / N)
    return (1.0 / N) * np.dot(F_inv, c_hat)
