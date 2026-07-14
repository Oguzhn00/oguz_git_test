# -*- coding: utf-8 -*-
"""
radar_core.py
=============
A121 Öğrenme Laboratuvarı'nın ŞEFFAF çekirdeği.

Buradaki her fonksiyon açıktır: Acconeer'in kapalı `algo` modüllerinin
(sparse_iq.Processor, distance.Detector) yaptığı işi kendimiz, satır satır
görebileceğimiz biçimde yazıyoruz. Amaç "iyi bir algoritma" değil,
"her parametrenin sinyale nasıl etki ettiğini görmek".

İçerik:
  1) Sabitler ve PRF tablosu (Acconeer datasheet değerleri)
  2) Profil (darbe) özellikleri
  3) SensorParams veri sınıfı  (radarın tüm ayarları)
  4) Menzil ekseni yardımcıları (nokta -> metre)
  5) Sparse IQ işleme:  genlik (coherent / fft-max), faz, menzil-hız haritası
  6) Distance işleme:   eşik (CFAR / sabit) ve tepe (peak) bulma
  7) Türetilmiş büyüklükler: menzil çözünürlüğü, hız çözünürlüğü, SNR

Ham veri formatı (hem simülasyon hem donanım aynı formatı üretir):
    frame : np.ndarray, shape = (sweeps_per_frame, num_points), dtype=complex
    Yani bir "frame" içinde birden çok "sweep" var; her sweep bir mesafe
    ekseni boyunca kompleks (IQ) örnekler taşır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1) FİZİKSEL SABİTLER
# ---------------------------------------------------------------------------
C = 299_792_458.0            # ışık hızı [m/s]
F0 = 60.5e9                  # A121 merkez frekansı ~60.5 GHz
LAMBDA = C / F0              # dalga boyu ~4.955 mm  -> faz her lambda/2'de tam tur döner
POINT_M = 2.5e-3            # bir "nokta" ~2.5 mm (A121 taban örnekleme aralığı)

# A121 donanımı step_length'i yalnızca 24'ün BÖLENİ ya da KATI olarak kabul eder
# (SPARSE_IQ_PPC = 24). Diğer değerler "Step length must be a divisor or multiple
# of 24" hatası verir. Bu yüzden 5, 7, 9, 10... geçersizdir.
VALID_STEP_LENGTHS = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 96, 120, 144]


def snap_step_length(step: int) -> int:
    """Verilen step'i geçerli en yakın (onu geçmeyen) A121 step_length'ine yuvarlar."""
    valid = [s for s in VALID_STEP_LENGTHS if s <= max(1, int(step))]
    return valid[-1] if valid else VALID_STEP_LENGTHS[0]


def is_valid_step_length(step: int) -> bool:
    return int(step) in VALID_STEP_LENGTHS


# ---------------------------------------------------------------------------
# 2) PRF TABLOSU  (Acconeer A121 API'sindeki gerçek değerler)
#    key -> (frekans [Hz], MMD [m], MUR [m])
#    MMD = Maksimum Ölçülebilir Mesafe (range end bunu geçemez)
#    MUR = Maksimum Belirsizsiz Menzil (bundan uzaktaki hedef katlanır/fold)
#    Not: 19.5 MHz en yüksek PRF -> en kısa MMD/MUR (3.1 / 7.7 m) ve Profil 2
#    ile birlikte KULLANILAMAZ (donanım kısıtı, bkz. prf_profile_ok).
# ---------------------------------------------------------------------------
PRF_TABLE = {
    "19.5 MHz": (19_500_000, 3.1, 7.7),
    "15.6 MHz": (15_600_000, 5.1, 9.6),
    "13.0 MHz": (13_000_000, 7.0, 11.5),
    "8.7 MHz": (8_700_000, 12.7, 17.3),
    "6.5 MHz": (6_500_000, 18.5, 23.1),
    "5.2 MHz": (5_200_000, 24.3, 28.8),
}


def prf_profile_ok(prf_key: str, profile: int) -> bool:
    """A121 donanım kısıtı: 19.5 MHz PRF, Profil 2 ile birlikte kullanılamaz."""
    if prf_key == "19.5 MHz" and profile == 2:
        return False
    return True


def prf_mur(prf_key: str) -> float:
    """Seçili PRF için Maksimum Belirsizsiz Menzil (m)."""
    return PRF_TABLE[prf_key][2]


def prf_mmd(prf_key: str) -> float:
    """Seçili PRF için Maksimum Ölçülebilir Mesafe (m)."""
    return PRF_TABLE[prf_key][1]


# ---------------------------------------------------------------------------
# 3) PROFİL (DARBE) ÖZELLİKLERİ
#    Yüksek profil = daha uzun darbe = daha çok enerji (yüksek SNR, uzun menzil)
#                    AMA daha kötü mesafe çözünürlüğü + daha güçlü yakın-kaçak.
#    FWHM değerleri Acconeer'in RESMİ değerleridir (exptool ENVELOPE_FWHM_M):
#    darbe zarfının yarı-yükseklik tam genişliği. Mesafe çözünürlüğü ~ bu genişlik.
# ---------------------------------------------------------------------------
#   profil -> darbe zarfı FWHM [m]  (Acconeer resmi: 0.04/0.07/0.14/0.19/0.32)
PROFILE_FWHM_M = {1: 0.04, 2: 0.07, 3: 0.14, 4: 0.19, 5: 0.32}
#   profil -> göreli genlik kazancı (uzun darbe daha çok enerji toplar)
PROFILE_GAIN = {1: 1.0, 2: 1.35, 3: 1.7, 4: 2.2, 5: 2.8}
#   profil -> yakın-mesafe direkt kaçak (leakage) göreli genliği
PROFILE_LEAKAGE = {1: 0.6, 2: 1.0, 3: 1.6, 4: 2.6, 5: 4.0}

# ---------------------------------------------------------------------------
# 3b) TX GÜCÜ — SAYISAL (dB) TEMSİL
#     A121'de SÜREKLİ bir "TX power" yazmacı YOKTUR. Verilen enerji (radar loop
#     gain, RLG) darbe profiliyle (5 kademe) ayarlanır; HWAAS ise entegrasyonla
#     +3 dB/2-kat kazanç verir. Bu yüzden TX gücünü, profile göre BAĞIL RLG [dB]
#     olarak sayısal ifade ediyoruz: profil 1 = 0 dB referans.
#       RLG_dB(profil) = 20*log10( PROFILE_GAIN[profil] )
#     Kullanıcı bir dB girer -> en yakın profile "snap" edilir (donanım ayrık).
# ---------------------------------------------------------------------------
PROFILE_RLG_DB = {p: round(20.0 * np.log10(PROFILE_GAIN[p]), 1) for p in PROFILE_GAIN}


def profile_tx_db(profile: int) -> float:
    """Profilin bağıl TX gücü [dB] (profil 1 = 0 dB)."""
    return PROFILE_RLG_DB[int(profile)]


def tx_db_to_profile(db: float) -> int:
    """Verilen bağıl TX gücünü [dB] en yakın donanım profiline (1..5) eşle."""
    return int(min(PROFILE_RLG_DB, key=lambda p: abs(PROFILE_RLG_DB[p] - db)))



def range_resolution_m(profile: int) -> float:
    """
    Kabaca ayrılabilecek en küçük iki-hedef mesafesi ~ darbe zarfı genişliği (FWHM).
    Adım uzunluğunu (step_length) ne kadar küçültürsen küçült, iki hedefi
    profilin darbe genişliğinden daha yakın ayıramazsın -> bu satır o dersi verir.
    """
    return PROFILE_FWHM_M[profile]


def sparrow_limit_m(profile: int) -> float:
    """
    İki eşit Gauss zarfı için 'tam ayrılabilir' mesafe (Sparrow limiti).
    Zarflar sigma kadar genişse d_min = 2*sigma; FWHM = 2.3548*sigma olduğundan
    d_min ≈ 0.849 * FWHM. Bu mesafenin ALTINDA iki tepe arasındaki çukur kaybolur
    (tek tepeye dönüşür) -> hedefler AYRILAMAZ.
    """
    return 0.8493218 * PROFILE_FWHM_M[profile]


# İki hedef için TEMİZ zarf (envelope) modeli — koherent girişim YOK.
# Çözünürlük sekmesi bunu kullanır ki "çözünürlük = darbe genişliği" dersi
# net görünsün. (Sparse IQ / Distance sekmeleri tam koherent modeli kullanır.)
REF_AMP_RES = 60.0
THERMAL_STD_RES = 0.9


def two_target_amplitude(r: np.ndarray, d1: float, d2: float, profile: int,
                         hwaas: int, rcs: float = 1.0,
                         rng: "np.random.Generator | None" = None) -> np.ndarray:
    """
    d1 ve d2'deki iki eşit hedefin genlik zarfı (girişimsiz, incoherent toplam):
      - her hedef Gauss zarfı, genişliği profile (FWHM) ile belirlenir
      - genlik ~1/R^2 (radar denklemi), profile göre kazanç
      - gürültü ~1/sqrt(hwaas)  (hwaas artınca taban düşer)
    Bu temiz model Sparrow limitini (d<2σ -> tek tepe) birebir gösterir.
    """
    sigma = PROFILE_FWHM_M[profile] / 2.3548
    pgain = PROFILE_GAIN[profile]
    a1 = REF_AMP_RES * rcs * pgain / max(d1, 1e-3) ** 2
    a2 = REF_AMP_RES * rcs * pgain / max(d2, 1e-3) ** 2
    amp = (a1 * np.exp(-0.5 * ((r - d1) / sigma) ** 2) +
           a2 * np.exp(-0.5 * ((r - d2) / sigma) ** 2))
    if rng is not None:
        std = THERMAL_STD_RES / np.sqrt(max(hwaas, 1))
        # Rician benzeri: karmaşık gürültünün büyüklüğü genliğe eklenir
        nz = np.abs(rng.standard_normal(r.size) + 1j * rng.standard_normal(r.size)) * (std / np.sqrt(2))
        amp = amp + nz
    return amp


def two_target_resolved(amp: np.ndarray, r: np.ndarray,
                        d1: float, d2: float, dip_ratio: float = 0.81):
    """
    Genlik eğrisinde d1 ve d2 civarındaki iki hedefin AYRIŞIP ayrışmadığını
    veriden ölçer. Pencere iki hedefin ortasından ikiye bölünür; her yarıdaki
    EN YÜKSEK tepe (biri sol hedef, biri sağ hedef) bulunur ve aralarındaki en
    düşük nokta (çukur) incelenir: çukur, küçük tepenin dip_ratio katından
    düşükse hedefler AYRIŞMIŞ demektir. Zarflar birleşip tek tepe olursa
    (geniş profil) iki yarının tepesi de ortada buluşur, çukur kalmaz -> birleşik.
    Bu yöntem gürültünün tek tepeyi ikiye bölmesine ve tek-hump durumuna dayanıklı.
    Dönüş: (resolved: bool, dip_ratio_measured: float, n_peaks: int)
    """
    if r.size < 5:
        return False, 1.0, 0
    lo, hi = min(d1, d2), max(d1, d2)
    i1 = int(np.argmin(np.abs(r - lo)))
    i2 = int(np.argmin(np.abs(r - hi)))
    if i2 - i1 < 2:
        return False, 1.0, 1
    a = max(0, i1 - 3)
    b = min(r.size - 1, i2 + 3)
    mid = (i1 + i2) // 2
    left = amp[a:mid + 1]
    right = amp[mid:b + 1]
    k1 = a + int(np.argmax(left))          # sol hedefin tepesi
    k2 = mid + int(np.argmax(right))       # sağ hedefin tepesi
    if k2 <= k1:
        return False, 1.0, 1
    p1 = float(amp[k1]); p2 = float(amp[k2])
    valley = float(np.min(amp[k1:k2 + 1]))
    small = max(min(p1, p2), 1e-9)
    ratio = valley / small
    # gerçek çukur: tepeler ayrı yerlerde VE aralarında belirgin düşüş
    resolved = (ratio < dip_ratio) and (k2 - k1 >= 2)
    return resolved, ratio, (2 if resolved else 1)


# ---------------------------------------------------------------------------
# 4) RADAR AYARLARI (tek subsweep)  — kenar çubuğundaki her kontrol burada
# ---------------------------------------------------------------------------
@dataclass
class SensorParams:
    # --- ölçüm aralığı ---
    start_point: int = 80          # başlangıç noktası (~start_point*2.5 mm)
    num_points: int = 160          # ölçülen nokta sayısı
    step_length: int = 1           # noktalar arası adım (adım*2.5 mm)
    # --- darbe & kazanç ---
    profile: int = 3               # 1..5
    hwaas: int = 8                 # donanım ortalama sayısı (gürültüyü ~1/sqrt(hwaas) düşürür)
    receiver_gain: int = 16        # 0..23 analog alıcı kazancı
    prf_key: str = "15.6 MHz"      # PRF anahtarı (PRF_TABLE)
    enable_tx: bool = True         # False -> verici kapalı (gürültü kalibrasyonu)
    # --- zamanlama ---
    sweeps_per_frame: int = 16     # frame başına sweep (hız/Doppler çözünürlüğü)
    sweep_rate: float = 2000.0     # sweep hızı [Hz] (Doppler eksenini belirler)
    # --- işleme (donanıma gitmez, bizim tarafımızda) ---
    amplitude_method: str = "coherent"   # "coherent" | "fftmax"
    doppler_method: str = "FFT"           # menzil-hız haritası: "FFT" | "Capon"
    window_type: str = "none"             # sweep FFT penceresi: "none" | "hamming" | "hann"
    threshold_method: str = "Sabit"      # "Sabit" (gürültü tabanı) | "CFAR"
    sensitivity: float = 0.5             # 0..1 (yüksek = daha çok tespit)
    fixed_threshold: float = 0.0         # "Sabit" eşik seviyesi (0 -> otomatik)
    cfar_guard: int = 16                 # CFAR koruma hücresi (tek yön) - darbe eteğini aş
    cfar_window: int = 10                # CFAR referans penceresi (tek yön)
    detection_floor: float = 0.0         # mutlak tespit tabanı: altındaki tepeler sayılmaz
                                         # (0 -> kapalı). Uzak/zayıf yanlış tespitleri temizler.
    min_snr_db: float = 15.0             # tespit için tepe, gürültü tabanının bu kadar dB
                                         # üstünde olmalı. Saf gürültüde yanlış tespiti önler.

    # A121 donanımının kare (frame) tamponu sabit: en fazla 4095 kompleks örnek.
    # Tek subsweep için gereken tampon = num_points * sweeps_per_frame.
    # Bu sınır aşılırsa donanım "Required buffer size is too large" hatası verir.
    BUFFER_MAX: ClassVar[int] = 4095

    def buffer_usage(self) -> int:
        """Bu ayarların donanımda kaplayacağı kare tamponu (kompleks örnek)."""
        return int(self.num_points) * int(self.sweeps_per_frame)

    def max_sweeps_per_frame(self) -> int:
        """num_points sabitken tampona sığan en büyük sweeps_per_frame (>=1)."""
        return max(1, self.BUFFER_MAX // max(1, int(self.num_points)))

    def buffer_ok(self) -> bool:
        return self.buffer_usage() <= self.BUFFER_MAX

    def start_m(self) -> float:
        return self.start_point * POINT_M

    def step_m(self) -> float:
        return self.step_length * POINT_M

    def end_m(self) -> float:
        return (self.start_point + (self.num_points - 1) * self.step_length) * POINT_M

    def range_axis_m(self) -> np.ndarray:
        """Her nokta için mesafe [m]. distance = (start + i*step) * 2.5mm."""
        idx = np.arange(self.num_points)
        return (self.start_point + idx * self.step_length) * POINT_M

    def velocity_axis_m_s(self) -> np.ndarray:
        """
        Menzil-hız haritasının hız ekseni.
        Sweep'ler arası FFT -> Doppler frekansı f_d ; hız v = f_d * lambda/2.
        FFT sweep_rate ile örneklendiği için f_d ekseni fftfreq(spf, 1/sweep_rate).
        """
        f_d = np.fft.fftshift(np.fft.fftfreq(self.sweeps_per_frame, d=1.0 / self.sweep_rate))
        # Simülasyonda +hız = yaklaşan (mesafe azalır) -> faz s ile artar ->
        # pozitif Doppler frekansı -> bu eksende pozitif hız. İşaret uyumlu.
        return f_d * LAMBDA / 2.0

    def velocity_resolution_m_s(self) -> float:
        """Hız çözünürlüğü = (lambda/2) * sweep_rate / SPF. SPF artınca iyileşir."""
        return (LAMBDA / 2.0) * self.sweep_rate / max(self.sweeps_per_frame, 1)

    def max_velocity_m_s(self) -> float:
        """Belirsizsiz maksimum hız = (lambda/2) * sweep_rate / 2."""
        return (LAMBDA / 2.0) * self.sweep_rate / 2.0


# ---------------------------------------------------------------------------
# 5) SPARSE IQ İŞLEME
#    Ham frame (spf, num_points) kompleks -> genlik / faz / menzil-hız haritası
# ---------------------------------------------------------------------------
def coherent_mean(frame: np.ndarray) -> np.ndarray:
    """
    Sweep'ler üzerinde KOMPLEKS (coherent) ortalama.
    Sabit hedefin fazı sweep'ler arası aynı olduğundan koherent toplanır;
    gürültünün fazı rastgele olduğundan kısmen sönümlenir -> SNR artar.
    """
    return frame.mean(axis=0)


def amplitude_coherent(frame: np.ndarray) -> np.ndarray:
    """Coherent yöntem: |koherent ortalama|."""
    return np.abs(coherent_mean(frame))


def amplitude_fftmax(frame: np.ndarray) -> np.ndarray:
    """
    FFT-max yöntem: sweep ekseninde FFT al, her mesafe için en güçlü Doppler
    bileşenini genlik say. Hareketli (Doppler'lı) hedefi koherent ortalamanın
    aksine sönümlemez -> hareketli hedefte daha iyi.
    """
    spf = frame.shape[0]
    spectrum = np.fft.fft(frame, axis=0) / spf
    return np.abs(spectrum).max(axis=0)


def phase_coherent(frame: np.ndarray) -> np.ndarray:
    """Koherent ortalamanın fazı [rad]. Küçük yer değiştirmeler burada okunur."""
    return np.angle(coherent_mean(frame))


def range_doppler_map(frame: np.ndarray, window: str = "none") -> np.ndarray:
    """
    Menzil-hız (range-Doppler) haritası: sweep (yavaş-zaman) ekseninde FFT +
    fftshift ile büyüklük spektrumu. Satırlar hız, sütunlar mesafe.

    window: FFT ÖNCESİ sweep eksenine uygulanacak pencere.
        "none"    -> pencere yok (dikdörtgen); en dar ana-lob, en yüksek yan-lob.
        "hamming" -> Hamming penceresi; yan-lobları bastırır (~-43 dB), ana-lob
                     bir miktar genişler. Birbirine yakın hızları temiz ayırmada
                     ve güçlü bir hedefin zayıf komşusunu maskelememesinde işe yarar.
        "hann"    -> Hann penceresi; yan-lob bastırma Hamming'e benzer, etekler
                     daha hızlı düşer.
    Pencere yalnızca istendiğinde uygulanır (varsayılan "none").
    """
    spf = frame.shape[0]
    if spf >= 4 and window and window != "none":
        if window == "hamming":
            w = np.hamming(spf)
        elif window == "hann":
            w = np.hanning(spf)
        else:
            w = np.ones(spf)
        frame = frame * w[:, None]
    spectrum = np.fft.fftshift(np.fft.fft(frame, axis=0), axes=0) / spf
    return np.abs(spectrum)


# ---------------------------------------------------------------------------
# 5b) CAPON / MVDR — HIZ (DOPPLER) SÜPER ÇÖZÜNÜRLÜĞÜ
#     A121 tek antenli olduğundan MVDR açı/yön kestirimi için DEĞİL; sweep
#     boyutunda çalışır. FFT'nin sabit çözünürlüğü (~1/M) yerine, MVDR uyarlanır
#     bir "hüzme" kurarak birbirine yakın hızları ayırabilir.
#
#     Yöntem (uzamsal spektral kestirimle aynı mantık, anten yerine sweep):
#       1) Her mesafe hücresinde M-uzunluklu yavaş-zaman vektörü x alınır.
#       2) Uzamsal yumuşatma (spatial smoothing): x, K-uzunluklu L=M-K+1 alt-vektöre
#          bölünüp kovaryans R (KxK) = (1/L) Σ y y^H kurulur (tek-frame'de R kestirimi).
#       3) Köşegen yükleme (diagonal loading) ile R kararlı hale getirilir.
#       4) Doppler yönlendirme vektörü a(f) = [1, e^{j2πf}, ..., e^{j2πf(K-1)}].
#       5) MVDR spektrumu: P(f) = 1 / (a(f)^H R^{-1} a(f)).
#     f normalize Doppler frekansı (çevrim/sweep) ∈ [-0.5, 0.5); hız v = f·sweep_rate·λ/2.
# ---------------------------------------------------------------------------
def capon_doppler_map(frame: np.ndarray, n_vel: int = 0,
                      subarray: int = 0, loading: float = 1e-2) -> np.ndarray:
    """
    Capon (MVDR) menzil-hız haritası. Dönüş: (n_vel, num_points) güç matrisi;
    satırlar hız (negatif->pozitif), sütunlar mesafe. FFT'ye göre daha keskin.
    Bağımsız da kullanılabilir: rc.capon_doppler_map(frame).
    """
    X = np.asarray(frame)
    if X.ndim == 1:
        X = X[None, :]
    M, N = X.shape
    K = subarray if subarray > 0 else max(2, M // 2)   # alt-dizi uzunluğu
    K = min(K, M)
    L = M - K + 1                                       # snapshot sayısı
    if n_vel <= 0:
        n_vel = max(64, 4 * K)                          # ince hız ızgarası
    f = np.linspace(-0.5, 0.5, n_vel, endpoint=False)   # normalize Doppler ekseni
    kk = np.arange(K)[:, None]
    A = np.exp(1j * 2 * np.pi * kk * f[None, :])         # (K, n_vel) yönlendirme
    P = np.zeros((n_vel, N))
    eyeK = np.eye(K)
    Jex = np.fliplr(np.eye(K))                           # değişim (exchange) matrisi
    # M çok küçükse (Capon anlamsız) FFT büyüklüğüne düş
    if M < 4 or L < 1:
        rd = range_doppler_map(X)
        idx = np.linspace(0, rd.shape[0] - 1, n_vel).astype(int)
        return rd[idx, :]
    for n in range(N):
        x = X[:, n]
        # uzamsal yumuşatma ile snapshot matrisi Y (K, L)
        Y = np.lib.stride_tricks.sliding_window_view(x, K).T   # (K, L)
        R = (Y @ Y.conj().T) / L
        # İLERİ-GERİ yumuşatma (forward-backward): aynı mesafedeki KORELASYONLU
        # (koherent) hedefleri ayrıştırabilmek için şart. R = (R + J R* J)/2.
        R = 0.5 * (R + Jex @ R.conj() @ Jex)
        R = R + loading * (np.trace(R).real / K) * eyeK        # köşegen yükleme
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            Rinv = np.linalg.pinv(R)
        RiA = Rinv @ A                                         # (K, n_vel)
        denom = np.real(np.einsum("kn,kn->n", A.conj(), RiA))  # a^H Rinv a
        P[:, n] = 1.0 / np.maximum(denom, 1e-12)
    return P


def capon_velocity_axis(sweep_rate: float, n_vel: int) -> np.ndarray:
    """Capon haritasının hız ekseni [m/s]. f∈[-0.5,0.5) -> v = f·sweep_rate·λ/2."""
    f = np.linspace(-0.5, 0.5, n_vel, endpoint=False)
    return f * sweep_rate * LAMBDA / 2.0


# ---------------------------------------------------------------------------
# 5c) ARKA PLAN (STATİK GÜRÜLTÜ/CLUTTER) KAYDET & ÇIKAR
#     Fikir: sahne boşken (ya da yalnızca durağan ortam varken) birkaç kare
#     kaydet, her mesafe hücresi için KOMPLEKS ortalama al -> "statik arka plan".
#     Sonra gelen her sweep'ten bunu çıkar: durağan yansımalar (duvar, kaçak,
#     sabit clutter) silinir; yalnızca değişen/hareketli bileşen kalır.
#     Kompleks çıkarma yapılır (faz dahil) çünkü A121 koherenttir.
# ---------------------------------------------------------------------------
def record_background(frames: list) -> np.ndarray:
    """
    Kaydedilen karelerden statik arka planı hesapla.
    frames: her biri (spf, num_points) kompleks kare listesi.
    Dönüş: (num_points,) kompleks — tüm sweep'ler üzerinden per-bin ortalama.
    """
    stacked = np.concatenate([np.asarray(f) for f in frames], axis=0)
    return stacked.mean(axis=0)


def subtract_background(frame: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Statik arka planı (per-bin kompleks) karenin HER sweep'inden çıkar."""
    if background is None:
        return frame
    return frame - background[None, :]


def save_background(path: str, background: np.ndarray, p: SensorParams) -> None:
    """
    Arka planı, hangi ölçüm geometrisinde alındığıyla birlikte .npz'e yaz.
    Geometri (start_point/num_points/step_length) daha sonra doğrulama içindir:
    yalnızca aynı aralıkta alınan arka plan geçerli biçimde çıkarılabilir.
    """
    np.savez(path, background=background,
             start_point=p.start_point, num_points=p.num_points,
             step_length=p.step_length, prf_key=p.prf_key)


def load_background(path: str):
    """
    .npz arka planı yükle. Dönüş: (background[complex], meta[dict]).
    meta: start_point/num_points/step_length/prf_key.
    """
    d = np.load(path, allow_pickle=True)
    meta = {
        "start_point": int(d["start_point"]),
        "num_points": int(d["num_points"]),
        "step_length": int(d["step_length"]),
        "prf_key": str(d["prf_key"]),
    }
    return d["background"].astype(np.complex128), meta


# ---------------------------------------------------------------------------
# 7) MERKEZİ İŞLEME  (ayrı süreçte çağrılır; GUI yalnızca sonucu çizer)
#     Aktif sekmeye göre yalnızca gerekli işi yapar -> hız.
# ---------------------------------------------------------------------------
def process_frame(frame: np.ndarray, p: SensorParams, tab: int) -> dict:
    """
    Tek karenin tüm işlenmiş çıktısını döndür (çizilmeye hazır).
    tab=0 -> Ham Veri (genlik/faz/menzil-hız), tab=1 -> Mesafe (eşik/tespit).
    """
    r = p.range_axis_m()
    out = {"r": r}
    if tab == 0:
        amp = amplitude_fftmax(frame) if p.amplitude_method == "fftmax" \
            else amplitude_coherent(frame)
        out["amp"] = amp
        out["phase"] = phase_coherent(frame)
        if p.doppler_method == "Capon":
            rd = capon_doppler_map(frame)
            vel = capon_velocity_axis(p.sweep_rate, rd.shape[0])
        else:
            rd = range_doppler_map(frame, window=p.window_type)
            vel = p.velocity_axis_m_s()
        out["rd"] = rd
        out["vel"] = vel
        out["snr"] = sweep_snr_db(amp)
    else:
        amp = amplitude_coherent(frame)
        nf = noise_floor_amp(frame)
        thr = compute_threshold(amp, p, noise_floor=nf)
        d, st = find_peaks(amp, thr, r)
        out["amp"] = amp
        out["thr"] = thr
        out["d"] = d
        out["st"] = st
    return out



def sweep_snr_db(amplitude: np.ndarray) -> float:
    """
    Kaba SNR göstergesi: en güçlü tepe / gürültü tabanı (medyan) [dB].
    Parametre değiştikçe (hwaas, profile, gain...) bu sayının nasıl
    hareket ettiğini izlemek öğrenmenin özüdür.
    """
    if amplitude.size == 0:
        return float("nan")
    peak = float(amplitude.max())
    # Gürültü tabanı: geniş bir hedef menzilin çoğunu doldurabildiğinden
    # medyan yerine alt %25'lik dilimi kullanmak daha dayanıklıdır.
    noise = float(np.percentile(amplitude, 25)) + 1e-12
    return 20.0 * np.log10(peak / noise)


# ---------------------------------------------------------------------------
# 6) DISTANCE İŞLEME  (eşik + tepe bulma)
# ---------------------------------------------------------------------------
def cfar_threshold(amplitude: np.ndarray, guard: int, window: int,
                   margin: float) -> np.ndarray:
    """
    ŞEFFAF OS-CFAR (Order-Statistic CFAR) eşiği.
    Her mesafe hücresi için: yakın 'guard' hücreleri atla, dışındaki 'window'
    referans hücrelerinin MEDYANINI yerel gürültü kabul et, eşiği medyan*margin
    yap. Medyan (ortalama değil) sayesinde referansa taşan komşu hedef/etek
    birkaç yüksek hücre yerel gürültü tahminini bozmaz -> maskeleme olmaz.
    'margin' doğrudan lineer çarpandır (çağıran, dB'den hesaplar).
    """
    n = amplitude.size
    thr = np.full(n, np.nan)
    for i in range(n):
        lo1 = max(0, i - guard - window)
        lo2 = max(0, i - guard)
        hi1 = min(n, i + guard + 1)
        hi2 = min(n, i + guard + window + 1)
        ref = np.concatenate([amplitude[lo1:lo2], amplitude[hi1:hi2]])
        if ref.size == 0:
            continue
        thr[i] = np.median(ref) * margin
    return thr


def fixed_threshold(amplitude: np.ndarray, level: float,
                    sensitivity: float, noise_floor: float = None) -> np.ndarray:
    """
    Sabit eşik. level>0 ise doğrudan o seviye; level<=0 ise gürültü tabanından
    (sinyalden bağımsız noise_floor verilirse ondan, yoksa alt %25) duyarlılığa
    bağlı otomatik sabit seviye üretir.
    """
    n = amplitude.size
    if level > 0:
        return np.full(n, level)
    noise = noise_floor if noise_floor is not None else \
        (float(np.percentile(amplitude, 25)) + 1e-12)
    margin = 1.0 + (0.3 + 3.0 * (1.0 - np.clip(sensitivity, 0.0, 1.0)))
    return np.full(n, noise * margin)


def noise_floor_amp(frame: np.ndarray) -> float:
    """
    SİNYALDEN BAĞIMSIZ gürültü tabanı kestirimi (koherent genlik ölçeğinde).

    Fikir: statik bir hedefin sinyali sweep'ten sweep'e SABİTTİR; gürültü ise
    değişir. Bu yüzden her mesafe hücresinde sweep-ler-arası dalgalanma
    (frame - sweep_ortalaması) yalnızca gürültüdür — hedef ne kadar güçlü ya da
    geniş olursa olsun. Böylece darbe tüm pencereyi doldursa bile gürültü tabanı
    doğru kestirilir (uzamsal yüzdelik burada çöker).

    Dönüş: koherent ortalama genlik üzerindeki yaklaşık gürültü tabanı.
    """
    f = np.atleast_2d(np.asarray(frame))
    spf = f.shape[0]
    if spf < 2:
        # Tek sweep: zaman bilgisi yok -> uzamsal alt %25'e düş (yaklaşık).
        return float(np.percentile(np.abs(f[0]), 25)) + 1e-12
    resid = f - f.mean(axis=0, keepdims=True)                 # statik bileşeni çıkar
    nstd = np.sqrt(np.mean(np.abs(resid) ** 2, axis=0))       # per-bin gürültü std
    noise_coherent = float(np.median(nstd)) / np.sqrt(spf)    # koherent ort. gürültüsü
    return noise_coherent * 1.2533 + 1e-12                    # Rayleigh ortalama faktörü


def compute_threshold(amplitude: np.ndarray, p: SensorParams,
                      noise_floor: float = None) -> np.ndarray:
    nf = noise_floor if noise_floor is not None else \
        (float(np.percentile(amplitude, 25)) + 1e-12)
    gate = nf * (10.0 ** (p.min_snr_db / 20.0))          

    if p.threshold_method == "CFAR":
        step_m = max(p.step_m(), 1e-6)
        
        # 1. KORUMA KALKANI (Guard): Hedefin darbe genişliğine (Profile) göre otomatik hesapla
        sigma_bins = (PROFILE_FWHM_M[p.profile] / 2.3548) / step_m
        eff_guard = max(1, int(np.ceil(2.5 * sigma_bins)))
        
        # 2. REFERANS PENCERESİ (Window): Hangi step_length olursa olsun sabit 40 cm olarak ayarla
        eff_window = max(2, int(np.ceil(0.40 / step_m)))
        
        cap = max(1, amplitude.size // 2 - eff_window - 2)
        eff_guard = min(eff_guard, cap)
        
        margin_db = 9.0 + (0.5 - np.clip(p.sensitivity, 0.0, 1.0)) * 24.0
        margin = 10.0 ** (margin_db / 20.0)
        thr = cfar_threshold(amplitude, eff_guard, eff_window, margin)
        thr = np.fmax(thr, gate)                          
    else:
        thr = np.full(amplitude.size, gate)               

    if p.detection_floor > 0:
        thr = np.fmax(thr, p.detection_floor)
    return thr

def find_peaks(amplitude: np.ndarray, threshold: np.ndarray,
               range_m: np.ndarray, min_dist_m: float = 0.30):
    """
    Eşiği aşan yerel maksimumları bul. Alt-hücre (sub-bin) doğruluk için
    parabolik interpolasyon uygula. Ardından NMS (Non-Maximum Suppression)
    ile birbirine çok yakın (yan lob) sahte tepeleri temizle.
    Dönüş: (mesafeler[m], güçler).
    """
    raw_dists = []
    raw_strengths = []
    n = amplitude.size
    
    # --- 1. AŞAMA: Tüm yerel maksimumları (tepeleri) bul ---
    for i in range(1, n - 1):
        a0, a1, a2 = amplitude[i - 1], amplitude[i], amplitude[i + 1]
        thr = threshold[i]
        if np.isnan(thr):
            continue
        # yerel tepe VE eşiğin üstünde mi?
        if a1 > thr and a1 >= a0 and a1 >= a2:
            denom = (a0 - 2 * a1 + a2)
            delta = 0.5 * (a0 - a2) / denom if denom != 0 else 0.0
            delta = float(np.clip(delta, -0.5, 0.5))
            if i + 1 < n:
                step = range_m[i + 1] - range_m[i]
            else:
                step = range_m[i] - range_m[i - 1]
            raw_dists.append(float(range_m[i] + delta * step))
            raw_strengths.append(float(a1))
            
    # Eğer hiç hedef yoksa veya sadece 1 hedef varsa doğrudan döndür
    if len(raw_dists) < 2:
        return np.array(raw_dists), np.array(raw_strengths)

    # --- 2. AŞAMA: NMS (Maksimum Olmayanları Bastırma) ---
    dists = np.array(raw_dists)
    strengths = np.array(raw_strengths)
    
    # Güçlerine göre büyükten küçüğe sırala
    sort_idx = np.argsort(strengths)[::-1]
    dists_sorted = dists[sort_idx]
    strengths_sorted = strengths[sort_idx]
    
    keep_dists = []
    keep_strengths = []
    suppressed = np.zeros(len(dists_sorted), dtype=bool)
    
    for i in range(len(dists_sorted)):
        if suppressed[i]:
            continue
        
        # Bu, henüz bastırılmamış en güçlü tepelerden biri, listeye ekle
        keep_dists.append(float(dists_sorted[i]))
        keep_strengths.append(float(strengths_sorted[i]))
        
        # Kendinden zayıf olan ve 'min_dist_m' (örn: 20 cm) yakınına düşenleri sil
        for j in range(i + 1, len(dists_sorted)):
            if not suppressed[j] and abs(dists_sorted[i] - dists_sorted[j]) < min_dist_m:
                suppressed[j] = True
                
    # Mesafeye göre baştan sona (yakından uzağa) tekrar sırala
    final_dists = np.array(keep_dists)
    final_strengths = np.array(keep_strengths)
    sort_by_dist = np.argsort(final_dists)
    
    return final_dists[sort_by_dist], final_strengths[sort_by_dist]