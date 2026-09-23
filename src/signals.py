"""
Signale und Testfunktionen für die Fourier-Analyse und Signalverarbeitung.
"""
import numpy as np


def hat_function(t: np.ndarray | float, h: float = np.pi / 2) -> np.ndarray:
    """
    Zentrierte Hutfunktion (periodisch fortgesetzt auf [-pi, pi]) (Dreieckssignal).
    
    Definition:
        f(t) = max(0, 1 - |t| / h)  für t in [-pi, pi)
        
    Parameter
    ---------
    t : float oder np.ndarray
        Zeitpunkte oder Phasenwinkel im Bogenmaß.
    h : float, optional
        Halbwertsbreite des Hutes (Standardwert: pi / 2).
        
    Rückgabe
    --------
    float oder np.ndarray
        Signalwerte im Bereich [0, 1].
    """
    return np.maximum(0.0, 1.0 - np.abs(t) / h)
