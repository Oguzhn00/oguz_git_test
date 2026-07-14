# -*- coding: utf-8 -*-
"""
sources.py
==========
İki veri kaynağı, aynı arayüz:

    get_frame(params) -> np.ndarray  (spf, num_points) kompleks + meta bilgi

1) SimSource : AÇIK fizik modeli. Donanım GEREKMEZ. Her parametrenin
   sinyale etkisini deterministik biçimde görürsün. "Neden böyle oluyor"
   sorusunun cevabı tamamen bu dosyada.

2) HwSource  : Gerçek A121 (acconeer.exptool). Ham `result.frame`'i alır,
   işlemeyi yine BİZİM radar_core fonksiyonlarımız yapar (Acconeer'in kapalı
   algo'su değil). acconeer kütüphanesi yalnızca donanım seçilince import edilir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from radar_core import (
    LAMBDA,
    PROFILE_FWHM_M,
    PROFILE_GAIN,
    PROFILE_LEAKAGE,
    SensorParams,
    prf_mur,
)


# ---------------------------------------------------------------------------
# Simülasyon sahnesindeki bir hedef
# ---------------------------------------------------------------------------
@dataclass
class Target:
    distance_m: float = 0.5     # gerçek mesafe [m]
    rcs: float = 1.0            # göreli yansıtıcılık (radar kesit alanı benzeri)
    velocity_m_s: float = 0.0   # radyal hız [m/s] (+: YAKLAŞIYOR, mesafe azalır)
    enabled: bool = True


@dataclass
class FrameResult:
    frame: np.ndarray            # (spf, num_points) kompleks ham IQ
    saturated: bool = False      # alıcı doygunluğa ulaştı mı?
    source: str = "sim"


# ---------------------------------------------------------------------------
# Simülasyon çekirdeği için ayar sabitleri (eğitim amaçlı seçilmiş ölçekler)
# ---------------------------------------------------------------------------
REF_AMP = 60.0          # 1 m'deki, rcs=1 hedefin referans genliği
THERMAL_STD = 0.9       # alıcı termal gürültü std'si (hwaas ile ~1/sqrt bölünür)
ADC_FLOOR_STD = 0.04    # kazançtan SONRA eklenen sabit taban (ADC/kuantalama)
SAT_LEVEL = 2200.0      # |örnek| bunu geçerse doygunluk (clip)
LEAK_BASE = 90.0        # yakın-mesafe direkt kaçağın taban genliği (1/R^2'ye tabi değil)


def _gain_linear(receiver_gain: int) -> float:
    """
    Alıcı kazancını (0..23) lineer voltaj çarpanına çevir (TEMSİLİ).
    Varsayılan 16 -> 1.0. Her kademe ~1.5 dB. Yüksek kazanç sinyali büyütür
    ama ADC'yi doygunluğa iter; düşük kazanç sinyali sabit tabana yaklaştırır.
    """
    gain_db = (receiver_gain - 16) * 1.5
    return float(10.0 ** (gain_db / 20.0))


class SimSource:
    """AÇIK fizik simülasyonu. Donanım gerektirmez."""

    def __init__(self, targets: Optional[List[Target]] = None,
                 noise_scale: float = 1.0, leakage: bool = True, seed: Optional[int] = None):
        self.targets: List[Target] = targets if targets is not None else [
            Target(0.40, 1.0, 0.0),
            Target(0.90, 0.6, 0.0),
        ]
        self.noise_scale = noise_scale     # gürültü tabanını el ile ölçekle
        self.leakage = leakage             # yakın-mesafe direkt kaçak ekle
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def get_frame(self, p: SensorParams) -> FrameResult:
        spf = int(p.sweeps_per_frame)
        r = p.range_axis_m()                     # (num_points,)
        n = r.size
        dt = 1.0 / p.sweep_rate
        k = 2.0 * np.pi / LAMBDA                 # dalga sayısı
        mur = prf_mur(p.prf_key)                 # katlanma (fold) menzili

        frame = np.zeros((spf, n), dtype=np.complex128)

        # --- 1) Sinyal: verici açıksa hedefleri (ve kaçağı) ekle ---
        if p.enable_tx:
            sigma = PROFILE_FWHM_M[p.profile] / 2.3548   # FWHM -> Gauss sigma
            pgain = PROFILE_GAIN[p.profile]

            # (a) Gerçek hedefler: genlik radar denklemine göre ~1/R^2
            for t in self.targets:
                if not t.enabled:
                    continue
                Rt = max(t.distance_m, 1e-3)
                base = REF_AMP * t.rcs * pgain / (Rt ** 2)
                for s in range(spf):
                    # +velocity YAKLAŞIYOR demek -> mesafe azalır
                    Rs = t.distance_m - t.velocity_m_s * s * dt
                    r_app = Rs % mur                                  # MUR katlanması
                    env = np.exp(-0.5 * ((r - r_app) / sigma) ** 2)   # darbe zarfı (mesafe)
                    carrier = np.exp(-1j * 2.0 * k * Rs)              # hızlı taşıyıcı faz
                    frame[s] += base * env * carrier

            # (b) Yakın-mesafe direkt kaçak (leakage): 0 m'de SABİT büyük genlikli
            # blob. 1/R^2 kuralına tabi DEĞİL (uzak hedef değil, doğrudan kuplaj).
            # Profil büyüdükçe kaçak büyür ve genişler; yüksek kazançta doygunluğa
            # itebilir. start_point küçükse ekranda görünür, büyükse zar zor.
            if self.leakage:
                leak_amp = LEAK_BASE * PROFILE_LEAKAGE[p.profile]
                env = np.exp(-0.5 * (r / sigma) ** 2)
                carrier = np.exp(-1j * 2.0 * k * r)  # kaçak da faz taşır
                frame += (leak_amp * env * carrier)[None, :]

        # --- 2) Termal gürültü: hwaas ortalaması ~1/sqrt(hwaas) düşürür ---
        std = THERMAL_STD * self.noise_scale / np.sqrt(max(p.hwaas, 1))
        noise = (self._rng.standard_normal((spf, n)) +
                 1j * self._rng.standard_normal((spf, n))) * (std / np.sqrt(2.0))
        frame += noise

        # --- 3) Alıcı kazancı (sinyal+gürültüyü birlikte büyütür) ---
        g = _gain_linear(p.receiver_gain)
        frame *= g

        # --- 4) Kazançtan sonra sabit taban (ADC/kuantalama gürültüsü) ---
        adc = (self._rng.standard_normal((spf, n)) +
               1j * self._rng.standard_normal((spf, n))) * (ADC_FLOOR_STD / np.sqrt(2.0))
        frame += adc

        # --- 5) Doygunluk (clip): |örnek| SAT_LEVEL'i geçerse kırp ---
        mag = np.abs(frame)
        saturated = bool(np.any(mag > SAT_LEVEL))
        if saturated:
            over = mag > SAT_LEVEL
            frame[over] = frame[over] / mag[over] * SAT_LEVEL

        return FrameResult(frame=frame, saturated=saturated, source="sim")


class HwSource:
    """
    Gerçek A121 donanımı. acconeer.exptool üzerinden ham frame okur.

    BAĞLANTI MANTIĞI, çalıştığı doğrulanan a121_modes_gui ile BİREBİR aynıdır:
      - client_kwargs: USB -> {"usb_device": True}, Seri -> {"serial_port": ...},
        Socket -> {"ip_address": ...}, Mock -> {"mock": True}
      - a121.Client.open(**client_kwargs)
      - a121.SessionConfig([{sensor_id: sensor_config}], extended=True)
      - setup_session -> start_session -> döngüde get_next
    İşlemeyi yine BİZİM radar_core fonksiyonlarımız yapar (ham frame alınır).
    """

    def __init__(self):
        self._client = None
        self._a121 = None
        self._IdleState = None
        self._Profile = None
        self._configured_key = None
        self._sensor_id = 1

    def _lazy_import(self):
        if self._a121 is None:
            from acconeer.exptool import a121  # yalnızca donanım seçilince
            # idle-state enum'u (çalışan dosyadaki ile aynı kaynak)
            try:
                from acconeer.exptool.a121._core.entities.configs.config_enums import (
                    IdleState, Profile,
                )
            except Exception:  # sürüm farkı: üst seviyeden dene
                IdleState = a121.IdleState
                Profile = a121.Profile
            self._a121 = a121
            self._IdleState = IdleState
            self._Profile = Profile
        return self._a121

    def open(self, client_kwargs: Optional[dict] = None):
        """
        client_kwargs, GUI'de çalışan dosyayla aynı biçimde kurulur:
        {"usb_device": True} / {"serial_port": "COM4"} / {"ip_address": "..."} / {"mock": True}
        """
        a121 = self._lazy_import()
        if self._client is None:
            self._client = a121.Client.open(**(client_kwargs or {"usb_device": True}))

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                self._configured_key = None

    def _build_sensor_config(self, p: SensorParams):
        """
        SensorConfig kurulumu çalışan dosyadaki ile aynı üslupta: alanlar doğrudan
        SensorConfig üzerinde ayarlanır (tek örtük subsweep). Ek olarak bu araçtaki
        kenar-çubuğu kontrolleri de uygulanır (receiver_gain, prf, enable_tx).
        """
        a121 = self._a121
        IdleState = self._IdleState
        Profile = self._Profile
        prf_map = {
            "15.6 MHz": a121.PRF.PRF_15_6_MHz,
            "13.0 MHz": a121.PRF.PRF_13_0_MHz,
            "8.7 MHz": a121.PRF.PRF_8_7_MHz,
            "6.5 MHz": a121.PRF.PRF_6_5_MHz,
            "5.2 MHz": a121.PRF.PRF_5_2_MHz,
        }
        cfg = a121.SensorConfig(
            sweeps_per_frame=p.sweeps_per_frame,
            inter_frame_idle_state=IdleState.READY,
            inter_sweep_idle_state=IdleState.READY,
            continuous_sweep_mode=False,
            double_buffering=False,
        )
        # tek subsweep alanları (SensorConfig bunları örtük subsweep'e yansıtır)
        cfg.start_point = p.start_point
        cfg.num_points = p.num_points
        cfg.step_length = p.step_length
        cfg.profile = Profile(p.profile)
        cfg.hwaas = p.hwaas
        cfg.receiver_gain = p.receiver_gain
        cfg.enable_tx = p.enable_tx
        cfg.prf = prf_map[p.prf_key]
        return cfg

    def reconfigure(self, p: SensorParams, sensor_id: int = 1):
        """Yeni parametreleri donanıma yükle (oturumu yeniden kur)."""
        a121 = self._lazy_import()
        if self._client is None:
            raise RuntimeError("Önce open() ile bağlanın.")
        # Donanıma göndermeden önce kare tamponunu doğrula (net Türkçe mesaj).
        if not p.buffer_ok():
            raise ValueError(
                f"Kare tamponu aşıldı: {p.buffer_usage()}/{p.BUFFER_MAX} örnek. "
                f"num_points={p.num_points} iken sweeps_per_frame en fazla "
                f"{p.max_sweeps_per_frame()} olabilir. sweeps_per_frame'i düşürün "
                f"ya da num_points'i azaltın.")
        # step_length yalnızca 24'ün böleni/katı olabilir (donanım kuralı).
        from radar_core import is_valid_step_length, VALID_STEP_LENGTHS, prf_profile_ok
        if not is_valid_step_length(p.step_length):
            raise ValueError(
                f"Geçersiz step_length={p.step_length}. A121 yalnızca 24'ün "
                f"böleni/katını kabul eder: {VALID_STEP_LENGTHS}.")
        # 19.5 MHz PRF, Profil 2 (TX gücü 2) ile birlikte kullanılamaz.
        if not prf_profile_ok(p.prf_key, p.profile):
            raise ValueError(
                "19.5 MHz PRF, TX gücü 2 (Profil 2) ile birlikte kullanılamaz "
                "(A121 donanım kısıtı). Başka bir TX gücü ya da PRF seçin.")
        self._sensor_id = sensor_id
        try:
            self._client.stop_session()
        except Exception:
            pass
        sensor_config = self._build_sensor_config(p)
        # ÇALIŞAN DOSYAYLA AYNI: extended SessionConfig ile sar
        session_config = a121.SessionConfig(
            [{sensor_id: sensor_config}], extended=True
        )
        self._client.setup_session(session_config)
        self._client.start_session()
        self._configured_key = repr(p)

    def get_frame(self, p: SensorParams, sensor_id: int = 1) -> FrameResult:
        if self._client is None:
            raise RuntimeError("Donanım bağlı değil.")
        if self._configured_key != repr(p):
            self.reconfigure(p, sensor_id)
        sid = self._sensor_id
        results = self._client.get_next()
        # extended=True -> results: [ {sensor_id: Result}, ... ]
        res = results[0][sid]
        frame = np.asarray(res.frame)
        if frame.ndim == 1:
            frame = frame[None, :]
        saturated = bool(getattr(res, "data_saturated", False))
        return FrameResult(frame=frame.astype(np.complex128), saturated=saturated, source="hw")
