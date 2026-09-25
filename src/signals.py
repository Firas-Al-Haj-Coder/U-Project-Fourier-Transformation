"""
Signale und Testfunktionen für die Fourier-Analyse und Signalverarbeitung.

Dieses Modul stellt kanonische Testsignale verschiedener mathematischer Glattheitsklassen bereit:
  - C^-1: square_wave (Rechteckwelle, Sprungdiskontinuität) -> O(1/k) Abfall, Gibbs-Phänomen
  - C^0 : hat_function (Dreieck / Hutfunktion, Knick in 1. Ableitung) -> O(1/k^2) Abfall
  - C^1 : c1_parabolic_spline (Glatte Parabel, Knick in 2. Ableitung) -> O(1/k^3) Abfall
  - C^2 : c2_smooth_wave (Zweimal stetig diffbar, Knick in 3. Ableitung) -> O(1/k^4) Abfall
  - C^inf: smooth_bell (Analytische Glocke, unendlich oft diffbar) -> Exponentieller / spektraler Abfall
"""
import numpy as np


def hat_function(t: np.ndarray | float, h: float = np.pi / 2) -> np.ndarray:
    """
    Zentrierte Hutfunktion (periodisch fortgesetzt auf [-pi, pi]) (Dreieckssignal).
    
    Glattheitsklasse:
    -----------------
    C^0 (Stetig, aber Knickstellen in der 1. Ableitung bei t = 0, +/- h).
    
    Definition:
    -----------
        f(t) = max(0, 1 - |t| / h)  für t in [-pi, pi)
        
    Fourier-Koeffizienten Zerfall:
    ------------------------------
        |a_k| = O(1 / k^2)
        
    Parameter:
    ----------
    t : float oder np.ndarray
        Zeitpunkte oder Phasenwinkel im Bogenmaß.
    h : float, optional
        Halbwertsbreite des Hutes (Standardwert: pi / 2).
        
    Rückgabe:
    ---------
    np.ndarray
        Signalwerte im Bereich [0, 1].
    """
    return np.maximum(0.0, 1.0 - np.abs(t) / h)


def square_wave(t: np.ndarray | float, width: float = np.pi / 2, bipolar: bool = False) -> np.ndarray:
    """
    Zentrierte symmetrische Rechteckwelle (Klasse C^-1).
    
    Glattheitsklasse:
    -----------------
    C^-1 (Stückweise stetig, harte Sprungdiskontinuitäten bei t = +/- width).
    
    Definition:
    -----------
        bipolar = False:  f(t) = 1  für |t| <= width,  sonst 0
        bipolar = True:   f(t) = 1  für |t| <= width,  sonst -1
        
    Fourier-Koeffizienten Zerfall:
    ------------------------------
        |a_k| = O(1 / k)
        Zeigt an den Sprungstellen das klassische Gibbs-Phänomen mit ca. 8.95% Überschwingen.
        
    Parameter:
    ----------
    t : float oder np.ndarray
        Zeitpunkte oder Phasenwinkel im Bogenmaß auf [-pi, pi].
    width : float, optional
        Halbe Pulsbreite (Standardwert: pi / 2).
    bipolar : bool, optional
        Falls True, schwingt das Signal zwischen -1 und +1. Standard: False ([0, 1]).
        
    Rückgabe:
    ---------
    np.ndarray
        Signalwerte der Rechteckwelle.
    """
    t_arr = np.asarray(t, dtype=float)
    low_val = -1.0 if bipolar else 0.0
    return np.where(np.abs(t_arr) <= width, 1.0, low_val)


def c1_parabolic_spline(t: np.ndarray | float) -> np.ndarray:
    """
    Glatte stückweise Parabel (Quadratischer B-Spline, Klasse C^1).
    
    Glattheitsklasse:
    -----------------
    C^1 (Einmal stetig differenzierbar; f' ist stetig, aber f'' besitzt Sprünge bei t = +/- pi/2).
    
    Definition auf [-pi, pi]:
    -------------------------
        s = |t| / pi
        f(t) = 1 - 2*s^2       für s <= 0.5  (|t| <= pi/2)
        f(t) = 2*(1 - s)^2     für s >  0.5  (pi/2 < |t| <= pi)
        
    Eigenschaften:
    --------------
        - f(0) = 1, f(+/- pi) = 0
        - f'(0) = 0, f'(+/- pi) = 0  (periodische Randbedingung glatt erfüllt!)
        - f''(t) springt bei t = +/- pi/2 von -4/pi^2 auf +4/pi^2.
        
    Fourier-Koeffizienten Zerfall:
    ------------------------------
        |a_k| = O(1 / k^3)   (analytisch: a_k = 16/(pi^3 * k^3) * sin(k*pi/2))
        
    Parameter:
    ----------
    t : float oder np.ndarray
        Zeitpunkte oder Phasenwinkel im Bogenmaß.
        
    Rückgabe:
    ---------
    np.ndarray
        Signalwerte im Bereich [0, 1].
    """
    t_arr = np.asarray(t, dtype=float)
    s = np.abs(t_arr) / np.pi
    return np.where(s <= 0.5, 1.0 - 2.0 * s**2, 2.0 * (1.0 - s)**2)


def c2_smooth_wave(t: np.ndarray | float) -> np.ndarray:
    """
    Zweimal stetig differenzierbares Testsignal (Klasse C^2).
    
    Glattheitsklasse:
    -----------------
    C^2 (Zweimal stetig differenzierbar auf dem Torus [-pi, pi]; f, f', f'' sind periodisch stetig;
         die 3. Ableitung f''' springt an den Periodenrändern t = +/- pi).
         
    Definition auf [-pi, pi]:
    -------------------------
        f(t) = (1 - (t / pi)^2)^2
        
    Eigenschaften:
    --------------
        - f(0) = 1, f(+/- pi) = 0
        - f'(+/- pi) = 0
        - f''(-pi) = f''(pi) = 8 / pi^2  (krümmungsstetig am Periodenübergang!)
        - f'''(-pi+) = -24/pi, f'''(pi-) = +24/pi  (Knick in 3. Ableitung)
        
    Fourier-Koeffizienten Zerfall:
    ------------------------------
        |a_k| = O(1 / k^4)   (analytisch: a_k = 48 * (-1)^(k+1) / (pi^4 * k^4))
        
    Parameter:
    ----------
    t : float oder np.ndarray
        Zeitpunkte oder Phasenwinkel im Bogenmaß auf [-pi, pi].
        
    Rückgabe:
    ---------
    np.ndarray
        Signalwerte im Bereich [0, 1].
    """
    t_arr = np.asarray(t, dtype=float)
    return (1.0 - (t_arr / np.pi)**2)**2


def smooth_bell(t: np.ndarray | float) -> np.ndarray:
    """
    Analytische, unendlich oft differenzierbare Glockenkurve (Klasse C^inf).
    
    Glattheitsklasse:
    -----------------
    C^inf (Reell-analytisch auf ganz R, 2*pi-periodisch fortgesetzt).
    
    Definition:
    -----------
        f(t) = exp(cos(t) - 1)
        
    Eigenschaften:
    --------------
        - f(0) = 1, f(+/- pi) = exp(-2) approx 0.1353
        - Absolut glatt, keine Sprünge oder Knicke in irgendeiner Ableitung.
        
    Fourier-Koeffizienten Zerfall:
    ------------------------------
        Exponentieller (spektraler) Zerfall: |a_k| = O(e^(-alpha * k))
        Fällt im Log-Log-Plot extrem steil gekrümmt nach unten ab und erreicht
        bereits ab ca. k = 10..15 Maschinengenauigkeit (1e-15).
        
    Parameter:
    ----------
    t : float oder np.ndarray
        Zeitpunkte oder Phasenwinkel im Bogenmaß.
        
    Rückgabe:
    ---------
    np.ndarray
        Signalwerte im Bereich [exp(-2), 1].
    """
    t_arr = np.asarray(t, dtype=float)
    return np.exp(np.cos(t_arr) - 1.0)
