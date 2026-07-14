# -*- coding: utf-8 -*-
# =============================================================================
#  app.py — A121 Radar Panel  (PySide6 + pyqtgraph + multiprocessing)
# -----------------------------------------------------------------------------
#  HIZ MİMARİSİ (çok-süreçli / multiprocessing):
#    * AYRI BİR SÜREÇ (acquisition_worker) donanımdan/simülasyondan kare alır,
#      TÜM ağır işlemeyi (genlik, menzil-hız, Capon, eşik, tespit) yapar ve
#      yalnızca EN SON sonucu bir kuyruğa (queue) koyar (eskiler düşürülür).
#    * GUI süreci kuyruktan en son sonucu çekip yalnızca ÇİZER. Böylece donanımın
#      bloklaması ya da Capon'un ağırlığı arayüzü kilitlemez -> gecikme minimum,
#      "anında" yansır.
#
#    radar_core.py  ->  saf DSP (test edilebilir, donanımsız)
#    sources.py     ->  SimSource / HwSource
#    app.py (bu)    ->  çok-süreçli akış + arayüz
# =============================================================================

from __future__ import annotations

import os
import sys
import time
import queue
import multiprocessing as mp

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

import radar_core as rc
from radar_core import SensorParams
from sources import SimSource, HwSource, Target


# =============================================================================
#  AYRI SÜREÇ: veri alma + işleme  (GUI'den bağımsız çalışır)
# =============================================================================
def acquisition_worker(cmd_q, out_q, evt_q, stop_ev):
    """
    Sonsuz döngü: komutları (parametre/sahne/bağlantı/arka plan/kayıt) uygula,
    kare al, işle, EN SON sonucu out_q'ya koy (eskiler düşer). Bağlantı/kayıt gibi
    KAYBOLMAMASI gereken durum mesajları ayrı evt_q'ya gider (düşürülmez).
    GUI yalnızca çizer -> düşük gecikme.
    """
    import radar_core as rc
    from radar_core import SensorParams
    from sources import SimSource, HwSource, Target

    try:
        out_q.cancel_join_thread()   # kapanışta kuyruğu flush etmeyi bekleme
    except Exception:
        pass
    sim = SimSource(targets=[Target(0.30, 1.0, 0.0), Target(0.50, 0.7, 0.0)],
                    noise_scale=1.0, leakage=True, seed=0)
    hw = HwSource()
    st = {"use_hw": False, "params": SensorParams(), "tab": 1,
          "bg": None, "bg_meta": None, "subtract": False, "rec_env": None,
          "adaptive": False, "alpha": 0.05, "record_left": 0, "record_accum": [],
          "rec_on": False, "rec_path": None, "rec_frames": [], "rec_ts": []}

    def cfg_meta(p):
        return {"start_point": p.start_point, "num_points": p.num_points,
                "step_length": p.step_length, "prf_key": p.prf_key}

    def cfg_ok(p):
        m = st["bg_meta"]
        return (st["bg"] is not None and m is not None and
                m["start_point"] == p.start_point and m["num_points"] == p.num_points and
                m["step_length"] == p.step_length)

    def put_latest(item):
        # kuyruğu boşalt, yalnızca en son sonucu tut -> her zaman en güncel kare
        try:
            while True:
                out_q.get_nowait()
        except queue.Empty:
            pass
        try:
            out_q.put_nowait(item)
        except queue.Full:
            pass

    def handle(cmd):
        t = cmd.get("type")
        if t == "params":
            st["params"] = cmd["params"]
            if cmd.get("noise_scale") is not None:
                sim.noise_scale = cmd["noise_scale"]
            if cmd.get("leakage") is not None:
                sim.leakage = cmd["leakage"]
            if cmd.get("targets") is not None:
                sim.targets = [Target(*x) for x in cmd["targets"]]
            for k in ("tab", "subtract", "adaptive", "alpha"):
                if cmd.get(k) is not None:
                    st[k] = cmd[k]
        elif t == "source":
            st["use_hw"] = cmd["use_hw"]
        elif t == "connect":
            try:
                hw.close(); hw.open(cmd["kwargs"]); hw.reconfigure(st["params"])
                evt_q.put({"status": "Donanım bağlı ve yapılandırıldı.", "ok": True, "kind": "conn"})
            except Exception as e:
                evt_q.put({"status": f"Bağlantı hatası: {e}", "ok": False, "kind": "conn"})
        elif t == "record":
            st["record_left"] = int(cmd["n"]); st["record_accum"] = []
        elif t == "set_bg":
            st["bg"] = cmd["bg"]; st["bg_meta"] = cmd["meta"]; st["rec_env"] = cmd.get("env")
        elif t == "clear_bg":
            st["bg"] = None; st["bg_meta"] = None; st["rec_env"] = None
        elif t == "rec_start":
            # ham veri kaydını başlat: her kareyi zaman damgasıyla biriktir
            st["rec_on"] = True; st["rec_path"] = cmd["path"]
            st["rec_frames"] = []; st["rec_ts"] = []
        elif t == "rec_stop":
            # kaydı bitir: TÜM matrisleri + zaman damgalarını .npz'ye yaz
            st["rec_on"] = False
            if st["rec_frames"]:
                try:
                    import numpy as _np
                    p = st["params"]
                    _np.savez_compressed(
                        st["rec_path"],
                        frames=_np.stack(st["rec_frames"]),      # (kare, SPF, N) kompleks
                        timestamps=_np.array(st["rec_ts"]),       # (kare,) unix zaman
                        start_point=p.start_point, num_points=p.num_points,
                        step_length=p.step_length, prf_key=p.prf_key,
                        sweeps_per_frame=p.sweeps_per_frame, sweep_rate=p.sweep_rate)
                    evt_q.put({"status": f"Kaydedildi: {len(st['rec_frames'])} kare → "
                                          f"{os.path.basename(st['rec_path'])}", "ok": True, "kind": "rec"})
                except Exception as e:
                    evt_q.put({"status": f"Kayıt hatası: {e}", "ok": False, "kind": "rec"})
            st["rec_frames"] = []; st["rec_ts"] = []

    while not stop_ev.is_set():
        t_loop = time.perf_counter()
        try:
            while True:
                handle(cmd_q.get_nowait())
        except queue.Empty:
            pass

        p = st["params"]
        try:
            fr = hw.get_frame(p) if st["use_hw"] else sim.get_frame(p)
        except Exception as e:
            put_latest({"error": str(e)}); time.sleep(0.05); continue
        raw = fr.frame

        # --- ham veri kaydı: TÜM matrisleri zaman damgasıyla biriktir ---
        if st["rec_on"]:
            st["rec_frames"].append(raw.copy()); st["rec_ts"].append(time.time())
        # --- arka plan: N-kare kaydı ---
        bg_ready = False
        if st["record_left"] > 0:
            st["record_accum"].append(raw.copy()); st["record_left"] -= 1
            if st["record_left"] == 0:
                st["bg"] = rc.record_background(st["record_accum"])
                st["rec_env"] = rc.record_background_env(st["record_accum"])
                st["bg_meta"] = cfg_meta(p); st["record_accum"] = []; bg_ready = True
                evt_q.put({"bg_ready": True, "kind": "bg"})
        # --- arka plan: sürekli/adaptif (EMA), kare-kare güncellenir ---
        if st["adaptive"]:
            fm = raw.mean(axis=0)
            st["bg"] = fm if st["bg"] is None else (1 - st["alpha"]) * st["bg"] + st["alpha"] * fm
            st["bg_meta"] = cfg_meta(p)

        # --- ana matristen çıkar ---
        frame = raw
        if st["subtract"] and cfg_ok(p):
            frame = rc.subtract_background(raw, st["bg"])

        # --- işle (aktif sekmeye göre) ---
        try:
            res = rc.process_frame(frame, p, st["tab"], rec_env=st["rec_env"])
        except Exception as e:
            put_latest({"error": str(e)}); continue
        res.update({"raw": raw, "proc": frame, "saturated": fr.saturated,
                    "tab": st["tab"], "bg_ready": bg_ready,
                    "record_left": st["record_left"],
                    "rec_on": st["rec_on"], "rec_count": len(st["rec_frames"]),
                    "bg": st["bg"], "bg_meta": st["bg_meta"], "cfg_ok": cfg_ok(p),
                    "rec_env": st["rec_env"]})
        put_latest(res)

        # SİM TEMPOSU: gerçek radar kare periyodu (spf/sweep_rate) kadar bekle.
        # Hem gerçekçi zamanlama sağlar hem CPU'yu boşuna %100 kilitlemez.
        # Donanımda get_frame zaten temposu belirlediği için bekleme yok.
        if not st["use_hw"]:
            period = p.sweeps_per_frame / max(p.sweep_rate, 1.0)
            period = min(max(period, 0.005), 0.1)          # 10..200 FPS arası sınırla
            elapsed = time.perf_counter() - t_loop
            if elapsed < period:
                time.sleep(period - elapsed)

    try:
        hw.close()
    except Exception:
        pass


# =============================================================================
#  TEMA
# =============================================================================
BG = "#0e1420"; PANEL = "#161f2e"; PANEL2 = "#1e2a3d"; GRID = "#243247"
TEXT = "#e6edf6"; MUTED = "#8ba0bd"
ACCENT = "#2fd6c6"; ACCENT2 = "#ff9f43"; ACCENT3 = "#7aa2ff"; DANGER = "#ff5c7a"
GOOD = "#43e08a"

QSS = f"""
QWidget {{ background:{BG}; color:{TEXT}; font-size:12px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background:{BG}; }}
QGroupBox {{ background:{PANEL}; border:1px solid {GRID}; border-radius:10px;
    margin-top:14px; padding:10px 10px 12px 10px; font-weight:600; }}
QGroupBox::title {{ subcontrol-origin:margin; left:12px; padding:2px 8px; color:{ACCENT}; }}
QLabel#hint {{ color:{MUTED}; font-size:11px; }}
QLabel#value {{ color:{ACCENT}; font-weight:600; }}
QTabWidget::pane {{ border:1px solid {GRID}; border-radius:10px; top:-1px; background:{PANEL}; }}
QTabBar::tab {{ background:{PANEL2}; color:{MUTED}; padding:9px 22px; margin-right:4px;
    border-top-left-radius:8px; border-top-right-radius:8px; font-weight:600; }}
QTabBar::tab:selected {{ background:{ACCENT}; color:#06121a; }}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{ background:{PANEL2}; border:1px solid {GRID};
    border-radius:6px; padding:4px 6px; selection-background-color:{ACCENT}; }}
QComboBox QAbstractItemView {{ background:{PANEL2}; selection-background-color:{ACCENT}; }}
QPushButton {{ background:{PANEL2}; border:1px solid {GRID}; border-radius:7px; padding:7px 12px; font-weight:600; }}
QPushButton:hover {{ border:1px solid {ACCENT}; }}
QPushButton#primary {{ background:{ACCENT}; color:#06121a; border:none; }}
QCheckBox {{ spacing:8px; }}
QSlider::groove:horizontal {{ height:5px; background:{GRID}; border-radius:3px; }}
QSlider::handle:horizontal {{ background:{ACCENT}; width:15px; margin:-6px 0; border-radius:8px; }}
QSlider::sub-page:horizontal {{ background:{ACCENT}; border-radius:3px; }}
QTableWidget {{ background:{PANEL2}; gridline-color:{GRID}; border:1px solid {GRID}; border-radius:6px; }}
QHeaderView::section {{ background:{PANEL}; color:{MUTED}; border:none; padding:4px; }}
QPlainTextEdit {{ background:#0b1018; border:1px solid {GRID}; border-radius:8px; color:{ACCENT}; }}
QSplitter::handle {{ background:{GRID}; }}
QScrollBar:vertical {{ background:{BG}; width:12px; margin:2px; border-radius:6px; }}
QScrollBar::handle:vertical {{ background:{GRID}; min-height:32px; border-radius:5px; }}
QScrollBar::handle:vertical:hover {{ background:{ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
"""


def make_lut(name="inferno"):
    stops = [(0.0, (0, 0, 4)), (0.25, (60, 12, 90)), (0.5, (150, 44, 90)),
             (0.72, (229, 92, 48)), (0.88, (250, 160, 60)), (1.0, (252, 255, 164))]
    xs = np.array([s[0] for s in stops]); cols = np.array([s[1] for s in stops], float)
    g = np.linspace(0, 1, 256); lut = np.zeros((256, 3))
    for c in range(3):
        lut[:, c] = np.interp(g, xs, cols[:, c])
    return lut.astype(np.uint8)


# =============================================================================
#  Kontrol yapıcıları
# =============================================================================
def group(title):
    g = QtWidgets.QGroupBox(title); lay = QtWidgets.QVBoxLayout(g); lay.setSpacing(8)
    return g, lay


def hint(text):
    lb = QtWidgets.QLabel(text); lb.setObjectName("hint")
    lb.setWordWrap(True); lb.setTextFormat(QtCore.Qt.RichText)
    return lb


def row(label, widget, value_label=None):
    w = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0)
    lb = QtWidgets.QLabel(label); lb.setMinimumWidth(96)
    h.addWidget(lb); h.addWidget(widget, 1)
    if value_label is not None:
        h.addWidget(value_label)
    return w


class Slider(QtWidgets.QWidget):
    changed = QtCore.Signal(int)

    def __init__(self, label, lo, hi, val, fmt="{}"):
        super().__init__()
        self.fmt = fmt
        h = QtWidgets.QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0)
        lb = QtWidgets.QLabel(label); lb.setMinimumWidth(96)
        self.s = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.s.setRange(lo, hi); self.s.setValue(val)
        self.val = QtWidgets.QLabel(fmt.format(val)); self.val.setObjectName("value")
        self.val.setMinimumWidth(58); self.val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        h.addWidget(lb); h.addWidget(self.s, 1); h.addWidget(self.val)
        self.s.valueChanged.connect(self._on)

    def _on(self, v):
        self.val.setText(self.fmt.format(v)); self.changed.emit(v)

    def value(self):
        return self.s.value()

    def set_value(self, v):
        self.s.blockSignals(True); self.s.setValue(int(v)); self.s.blockSignals(False)
        self.val.setText(self.fmt.format(int(v)))


# =============================================================================
#  PARK SENSÖRÜ / ENGEL GÖRSELİ  (üstten bakış: menzil halkaları + tespitler)
# =============================================================================
class ParkingWidget(QtWidgets.QWidget):
    """
    Araç park sensörü mantığı: sensör altta, menzil yukarı doğru artar.
    Tespit edilen engeller mesafelerinde yatay yay/çubuk olarak çizilir;
    en yakın engel yeşil(uzak)→sarı→kırmızı(yakın) renkle vurgulanır.
    """
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(320)
        self.rmin, self.rmax = 0.2, 0.6
        self.dets = []          # [(mesafe_m, güç), ...]

    def set_data(self, rmin, rmax, dets):
        self.rmin, self.rmax = rmin, max(rmax, rmin + 1e-3)
        self.dets = dets
        self.update()

    def _color_for(self, frac):
        # frac: 0 (en yakın) .. 1 (en uzak)
        if frac < 0.33:
            return QtGui.QColor(DANGER)
        if frac < 0.66:
            return QtGui.QColor(ACCENT2)
        return QtGui.QColor(GOOD)

    def paintEvent(self, ev):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        W, H = self.width(), self.height()
        qp.fillRect(0, 0, W, H, QtGui.QColor("#0b1018"))
        cx, cy = W / 2, H - 26          # sensör konumu (alt-orta)
        max_r = H - 70                   # piksel yarıçapı
        span = 150                       # yay açıklığı (derece)
        start_ang = 90 - span / 2

        # menzil halkaları + etiket
        pen = QtGui.QPen(QtGui.QColor(GRID)); pen.setWidth(1)
        qp.setPen(pen)
        qp.setFont(QtGui.QFont("sans", 8))
        for k in range(1, 5):
            rr = max_r * k / 4
            rect = QtCore.QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr)
            qp.drawArc(rect, int(start_ang * 16), int(span * 16))
            dist = self.rmin + (self.rmax - self.rmin) * k / 4
            qp.setPen(QtGui.QColor(MUTED))
            qp.drawText(QtCore.QPointF(cx + rr * 0.02, cy - rr - 3), f"{dist:.2f} m")
            qp.setPen(pen)

        # sensör üçgeni
        qp.setBrush(QtGui.QColor(ACCENT)); qp.setPen(QtCore.Qt.NoPen)
        qp.drawEllipse(QtCore.QPointF(cx, cy), 6, 6)

        # engeller
        nearest = None
        for dist, strength in self.dets:
            frac = (dist - self.rmin) / (self.rmax - self.rmin)
            frac = min(max(frac, 0.0), 1.0)
            rr = max_r * frac
            col = self._color_for(frac)
            pen = QtGui.QPen(col); pen.setWidth(6); pen.setCapStyle(QtCore.Qt.RoundCap)
            qp.setPen(pen); qp.setBrush(QtCore.Qt.NoBrush)
            rect = QtCore.QRectF(cx - rr, cy - rr, 2 * rr, 2 * rr)
            # engelin merkezinde ~40° genişlikte parlak yay
            qp.drawArc(rect, int((90 - 20) * 16), int(40 * 16))
            if nearest is None or dist < nearest:
                nearest = dist

        # en yakın engel bilgisi (büyük)
        qp.setFont(QtGui.QFont("sans", 15, QtGui.QFont.Bold))
        if nearest is not None:
            frac = min(max((nearest - self.rmin) / (self.rmax - self.rmin), 0.0), 1.0)
            qp.setPen(self._color_for(frac))
            qp.drawText(QtCore.QRectF(0, 8, W, 30), QtCore.Qt.AlignCenter,
                        f"● EN YAKIN ENGEL: {nearest*100:.0f} cm")
        else:
            qp.setPen(QtGui.QColor(MUTED))
            qp.drawText(QtCore.QRectF(0, 8, W, 30), QtCore.Qt.AlignCenter, "engel yok")
        qp.end()


# =============================================================================
#  ANA PENCERE
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cmd_q, out_q, evt_q, stop_ev):
        super().__init__()
        self.setWindowTitle("A121 Radar")
        self.resize(1380, 880)
        self.cmd_q, self.out_q, self.evt_q, self.stop_ev = cmd_q, out_q, evt_q, stop_ev

        self.params = SensorParams()
        self.sim_targets = [(0.30, 1.0, 0.0), (0.50, 0.7, 0.0)]  # sahne (worker'a gönderilir)
        self.running = True
        self.use_hw = False
        self.background = None
        self.bg_meta = None
        self.rec_env = None
        self._t_last = time.time(); self._fps = 0.0

        pg.setConfigOptions(antialias=True, background=PANEL, foreground=TEXT)
        self.lut = make_lut()
        self._build_ui()
        self._send_state()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(16)     # ~60 FPS çizim denemesi; worker kendi hızında üretir

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QtWidgets.QWidget(); self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central); root.setContentsMargins(12, 12, 12, 12); root.setSpacing(10)
        root.addWidget(self._build_header())
        body = QtWidgets.QHBoxLayout(); body.setSpacing(12)
        body.addWidget(self._build_sidebar(), 0)
        body.addWidget(self._build_tabs(), 1)
        root.addLayout(body, 1)

    def _build_header(self):
        bar = QtWidgets.QWidget(); bar.setStyleSheet(f"background:{PANEL}; border-radius:10px;")
        h = QtWidgets.QHBoxLayout(bar); h.setContentsMargins(16, 10, 16, 10)
        title = QtWidgets.QLabel("A121 Radar"); title.setStyleSheet(f"font-size:17px; font-weight:700;")
        h.addWidget(title); h.addStretch(1)
        self.fps_lb = QtWidgets.QLabel("0 FPS"); self.fps_lb.setStyleSheet(f"color:{MUTED};")
        self.pause_btn = QtWidgets.QPushButton("Duraklat"); self.pause_btn.clicked.connect(self._toggle_run)
        self.reset_btn = QtWidgets.QPushButton("Varsayılana Dön"); self.reset_btn.clicked.connect(self._reset_defaults)
        h.addWidget(self.fps_lb); h.addSpacing(8); h.addWidget(self.reset_btn); h.addWidget(self.pause_btn)
        return bar

    def _build_sidebar(self):
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setFixedWidth(372)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(2, 2, 8, 2); v.setSpacing(10)

        # Veri kaynağı
        g, l = group("Veri Kaynağı")
        self.source_combo = QtWidgets.QComboBox(); self.source_combo.addItems(["Simülasyon", "Donanım (A121)"])
        self.source_combo.currentTextChanged.connect(self._on_source); l.addWidget(self.source_combo)
        self.conn_combo = QtWidgets.QComboBox(); self.conn_combo.addItems(["USB", "Seri Port", "Socket / IP", "Mock (test)"])
        self.conn_combo.currentTextChanged.connect(self._on_conn_changed)
        self.conn_row = row("bağlantı", self.conn_combo); l.addWidget(self.conn_row)
        self.hw_port = QtWidgets.QLineEdit(); self.hw_port.setPlaceholderText("COM4 / /dev/ttyUSB0 veya IP")
        self.hw_connect = QtWidgets.QPushButton("Bağlan ve Uygula"); self.hw_connect.setObjectName("primary")
        self.hw_connect.clicked.connect(self._hw_connect)
        self.hw_status = hint("Simülasyon aktif — donanım gerekmez.")
        l.addWidget(self.hw_port); l.addWidget(self.hw_connect); l.addWidget(self.hw_status)
        self.conn_row.setVisible(False); self.hw_port.setVisible(False); self.hw_connect.setVisible(False)
        v.addWidget(g)

        # Ölçüm aralığı
        g, l = group("Ölçüm Aralığı")
        self.sp_start = self._spin(0, 4000, self.params.start_point)
        self.sp_num = self._spin(1, 1000, self.params.num_points)
        self.step_combo = QtWidgets.QComboBox(); self.step_combo.addItems([str(s) for s in rc.VALID_STEP_LENGTHS])
        self.step_combo.setCurrentText(str(self.params.step_length)); self.step_combo.currentTextChanged.connect(self._apply)
        l.addWidget(row("start_point", self.sp_start)); l.addWidget(row("num_points", self.sp_num))
        l.addWidget(row("step_length", self.step_combo))
        self.range_lb = hint(""); l.addWidget(self.range_lb)
        v.addWidget(g)

        # TX gücü & kazanç  (SAYISAL dB)
        g, l = group("TX Gücü & Kazanç")
        self.tx_db = QtWidgets.QDoubleSpinBox(); self.tx_db.setRange(0.0, 9.0); self.tx_db.setSingleStep(0.1)
        self.tx_db.setDecimals(1); self.tx_db.setValue(rc.profile_tx_db(self.params.profile))
        self.tx_db.setSuffix(" dB"); self.tx_db.valueChanged.connect(self._apply)
        l.addWidget(row("TX gücü (RLG)", self.tx_db))
        self.tx_map_lb = hint("")
        l.addWidget(self.tx_map_lb)
        self.sl_hwaas = Slider("hwaas", 1, 511, self.params.hwaas); self.sl_hwaas.changed.connect(self._apply); l.addWidget(self.sl_hwaas)
        self.sl_gain = Slider("receiver_gain", 0, 23, self.params.receiver_gain); self.sl_gain.changed.connect(self._apply); l.addWidget(self.sl_gain)
        self.prf_combo = QtWidgets.QComboBox(); self.prf_combo.addItems(list(rc.PRF_TABLE.keys()))
        self.prf_combo.setCurrentText(self.params.prf_key); self.prf_combo.currentTextChanged.connect(self._apply)
        l.addWidget(row("prf", self.prf_combo))
        self.tx_check = QtWidgets.QCheckBox("enable_tx (verici açık)"); self.tx_check.setChecked(True); self.tx_check.stateChanged.connect(self._apply)
        l.addWidget(self.tx_check)
        l.addWidget(hint("TX gücü <b>bağıl radar loop gain [dB]</b> olarak girilir "
                         "(A121'de sürekli güç yazmacı yok; darbe profili 5 kademe). "
                         "Girilen dB en yakın profile eşlenir. HWAAS her 2-kat ~+3 dB "
                         "entegrasyon kazancı ekler."))
        v.addWidget(g)

        # Zamanlama
        g, l = group("Zamanlama (hız / Doppler)")
        self.sp_spf = self._spin(1, 512, self.params.sweeps_per_frame)
        self.sp_rate = self._dspin(50, 20000, self.params.sweep_rate, 50)
        l.addWidget(row("sweeps_per_frame", self.sp_spf)); l.addWidget(row("sweep_rate", self.sp_rate))
        self.vel_lb = hint(""); self.buf_lb = hint("")
        l.addWidget(self.vel_lb); l.addWidget(self.buf_lb)
        v.addWidget(g)

        # İşleme
        g, l = group("İşleme")
        self.amp_combo = QtWidgets.QComboBox(); self.amp_combo.addItems(["coherent", "fftmax"]); self.amp_combo.currentTextChanged.connect(self._apply)
        l.addWidget(row("genlik yöntemi", self.amp_combo))
        self.dopp_combo = QtWidgets.QComboBox(); self.dopp_combo.addItems(["FFT", "Capon"]); self.dopp_combo.currentTextChanged.connect(self._apply)
        l.addWidget(row("menzil–hız", self.dopp_combo))
        # Doppler penceresi — İSTENDİĞİNDE uygulanır (sürekli değil), varsayılan yok.
        self.win_combo = QtWidgets.QComboBox(); self.win_combo.addItems(["Pencere yok", "Hamming", "Hann"])
        self.win_combo.currentTextChanged.connect(self._apply)
        l.addWidget(row("Doppler penceresi", self.win_combo))
        l.addWidget(hint("Pencere yalnızca seçildiğinde uygulanır. Hamming/Hann, "
                         "Doppler yan-loblarını bastırır (yakın hızları ayırmada iyi); "
                         "‘Pencere yok’ en dar ana-lobu verir."))
        self.thr_combo = QtWidgets.QComboBox(); self.thr_combo.addItems(["Gürültü tabanı", "CFAR", "Kayıtlı eşik"]); self.thr_combo.currentTextChanged.connect(self._apply)
        l.addWidget(row("eşik yöntemi", self.thr_combo))
        l.addWidget(hint("<b>Kayıtlı eşik</b>: sabit yansımaları (montaj, kablo, duvar, "
                         "kaçak) dışlamak için önce Arka Plan panelinden boş sahneyi "
                         "kaydet; eşik o kayıtlı seviyenin üstüne oturur — yalnızca yeni/gerçek "
                         "hedefler görünür. (Acconeer distance detector kalibrasyonu.)"))
        self.sl_sens = Slider("duyarlılık", 0, 100, int(self.params.sensitivity * 100), fmt="{}%"); self.sl_sens.changed.connect(self._apply); l.addWidget(self.sl_sens)
        self.sl_snr = Slider("min. tespit SNR", 0, 40, int(self.params.min_snr_db), fmt="{} dB"); self.sl_snr.changed.connect(self._apply); l.addWidget(self.sl_snr)
        self.sl_floor = Slider("min. tespit tabanı", 0, 400, int(self.params.detection_floor)); self.sl_floor.changed.connect(self._apply); l.addWidget(self.sl_floor)
        v.addWidget(g)

        # Ham veri kaydı (zaman damgalı)
        g, l = group("Ham Veri Kaydı (zaman damgalı)")
        self.rec_btn = QtWidgets.QPushButton("● Kayda Başla"); self.rec_btn.setObjectName("primary")
        self.rec_btn.clicked.connect(self._toggle_record)
        l.addWidget(self.rec_btn)
        self.rec_status = hint("Kayda başladığında GELEN TÜM ham matrisler, her birinin "
                               "zaman damgasıyla birlikte tek bir .npz dosyasına yazılır "
                               "(frames: kare×SPF×N kompleks, timestamps: unix saniye). "
                               "İstediğin an başlat/durdur.")
        l.addWidget(self.rec_status)
        self.recording = False
        v.addWidget(g)

        # Arka plan (statik gürültü) çıkarma
        g, l = group("Arka Plan (Statik Gürültü) Çıkarma")
        rb = QtWidgets.QHBoxLayout()
        self.bg_frames = self._spin(1, 200, 20, connect=False)
        rec_b = QtWidgets.QPushButton("Kaydet (N kare)"); rec_b.clicked.connect(self._bg_record)
        rb.addWidget(QtWidgets.QLabel("N:")); rb.addWidget(self.bg_frames); rb.addWidget(rec_b, 1); l.addLayout(rb)
        self.bg_subtract = QtWidgets.QCheckBox("gelen veriden çıkar"); self.bg_subtract.stateChanged.connect(self._apply); l.addWidget(self.bg_subtract)
        self.bg_adaptive = QtWidgets.QCheckBox("sürekli/adaptif (kare-kare EMA)"); self.bg_adaptive.stateChanged.connect(self._apply); l.addWidget(self.bg_adaptive)
        rio = QtWidgets.QHBoxLayout()
        save_b = QtWidgets.QPushButton("Dosyaya kaydet"); load_b = QtWidgets.QPushButton("Dosyadan yükle")
        save_b.clicked.connect(self._bg_save); load_b.clicked.connect(self._bg_load)
        rio.addWidget(save_b); rio.addWidget(load_b); l.addLayout(rio)
        self.bg_status = hint("Sahne boşken ‘Kaydet’ ile statik yansımayı al; ‘gelen "
                              "veriden çıkar’ ile durağan clutter’ı sil. ‘Sürekli/adaptif’, "
                              "arka planı kare-kare EMA ile günceller (yavaş değişen ortam).")
        l.addWidget(self.bg_status)
        v.addWidget(g)

        # Simülasyon sahnesi
        self.scene_group, l = group("Simülasyon Sahnesi")
        self.tbl = QtWidgets.QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Mesafe(m)", "RCS", "Hız(m/s)", "Aktif"])
        self.tbl.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False); self.tbl.setFixedHeight(150)
        self.tbl.cellChanged.connect(self._on_scene_edit); l.addWidget(self.tbl)
        bt = QtWidgets.QHBoxLayout()
        ab = QtWidgets.QPushButton("+ Hedef"); db = QtWidgets.QPushButton("− Sil")
        ab.clicked.connect(self._add_target); db.clicked.connect(self._del_target)
        bt.addWidget(ab); bt.addWidget(db); l.addLayout(bt)
        self.sl_noise = Slider("gürültü ×", 0, 400, 100, fmt="{}%"); self.sl_noise.changed.connect(self._apply); l.addWidget(self.sl_noise)
        self.leak_check = QtWidgets.QCheckBox("yakın-mesafe kaçak (leakage)"); self.leak_check.setChecked(True); self.leak_check.stateChanged.connect(self._apply); l.addWidget(self.leak_check)
        v.addWidget(self.scene_group)
        self._refresh_table()

        v.addStretch(1); scroll.setWidget(panel)
        return scroll

    def _spin(self, lo, hi, val, connect=True):
        s = QtWidgets.QSpinBox(); s.setRange(lo, hi); s.setValue(val)
        if connect:
            s.valueChanged.connect(self._apply)
        return s

    def _dspin(self, lo, hi, val, step):
        s = QtWidgets.QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(0); s.setSingleStep(step); s.setValue(val)
        s.valueChanged.connect(self._apply)
        return s

    # --------------------------------------------------------------- TABS
    def _build_tabs(self):
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_raw_tab(), "Ham Veri")
        self.tabs.addTab(self._build_distance_tab(), "Mesafe")
        self.tabs.currentChanged.connect(lambda *_: self._apply())
        return self.tabs

    def _build_raw_tab(self):
        # Üst üste binmeyi önlemek için DİKEY SPLITTER: kullanıcı bölmeleri sürükler.
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        self.raw_text = QtWidgets.QPlainTextEdit(); self.raw_text.setReadOnly(True)
        self.raw_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.raw_text.setFont(QtGui.QFont("Menlo, Consolas, monospace", 9))
        wrap_txt = QtWidgets.QWidget(); lt = QtWidgets.QVBoxLayout(wrap_txt); lt.setContentsMargins(0,0,0,0)
        lt.addWidget(QtWidgets.QLabel("Gelen ham matris (SPF × N kompleks IQ):"))
        lt.addWidget(self.raw_text)
        split.addWidget(wrap_txt)

        self.amp_plot = pg.PlotWidget(); self.amp_plot.setLabel("bottom", "Mesafe", units="m")
        self.amp_plot.setLabel("left", "Genlik"); self.amp_plot.showGrid(x=True, y=True, alpha=0.25)
        self.amp_curve = self.amp_plot.plot(pen=pg.mkPen(ACCENT, width=2))
        self.amp_peak = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(DANGER), pen=None); self.amp_plot.addItem(self.amp_peak)
        split.addWidget(self.amp_plot)

        self.phase_plot = pg.PlotWidget(); self.phase_plot.setLabel("bottom", "Mesafe", units="m")
        self.phase_plot.setLabel("left", "Faz [rad]"); self.phase_plot.showGrid(x=True, y=True, alpha=0.25)
        self.phase_curve = self.phase_plot.plot(pen=pg.mkPen(ACCENT3, width=1.5))
        split.addWidget(self.phase_plot)

        rdw = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(rdw); rl.setContentsMargins(0,0,0,0)
        self.rd_plot = pg.PlotWidget(); self.rd_plot.setLabel("bottom", "Mesafe", units="m"); self.rd_plot.setLabel("left", "Hız", units="m/s")
        self.rd_img = pg.ImageItem(); self.rd_img.setLookupTable(self.lut); self.rd_plot.addItem(self.rd_img)
        self.rd_bar = pg.ColorBarItem(values=(0, 1), colorMap=pg.ColorMap(np.linspace(0, 1, 256), self.lut))
        self.rd_bar.setImageItem(self.rd_img, insert_in=self.rd_plot.getPlotItem())
        rl.addWidget(self.rd_plot); self.iq_info = hint(""); rl.addWidget(self.iq_info)
        split.addWidget(rdw)
        split.setSizes([170, 260, 150, 300])

        # PANEL GÖRÜNÜRLÜK: her akışı (matris / genlik / faz / menzil-hız)
        # istediğin gibi ekle-çıkar. İşaret kaldırınca o bölme gizlenir.
        self.raw_panels = {"Matris": wrap_txt, "Genlik": self.amp_plot,
                           "Faz": self.phase_plot, "Menzil–Hız": rdw}
        toolbar = QtWidgets.QWidget(); tb = QtWidgets.QHBoxLayout(toolbar); tb.setContentsMargins(2, 2, 2, 2)
        tb.addWidget(QtWidgets.QLabel("Göster:"))
        self.raw_checks = {}
        for name, widget in self.raw_panels.items():
            cb = QtWidgets.QCheckBox(name); cb.setChecked(True)
            cb.stateChanged.connect(lambda _s, wv=widget, c=None: wv.setVisible(_s != 0))
            tb.addWidget(cb); self.raw_checks[name] = cb
        tb.addStretch(1)

        w = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(w); lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(toolbar); lay.addWidget(split, 1)
        return w

    def _build_distance_tab(self):
        w = QtWidgets.QWidget(); v = QtWidgets.QVBoxLayout(w); v.setSpacing(8)

        # Görünürlük araç çubuğu: eşik çizgisi + tespitler + park görseli aç/kapa
        toolbar = QtWidgets.QWidget(); tb = QtWidgets.QHBoxLayout(toolbar); tb.setContentsMargins(2, 2, 2, 2)
        tb.addWidget(QtWidgets.QLabel("Göster:"))
        self.chk_thr = QtWidgets.QCheckBox("Eşik çizgisi"); self.chk_thr.setChecked(True)
        self.chk_det = QtWidgets.QCheckBox("Tespitler"); self.chk_det.setChecked(True)
        self.chk_park = QtWidgets.QCheckBox("Park/engel görseli"); self.chk_park.setChecked(True)
        self.chk_thr.stateChanged.connect(lambda s: self.thr_curve.setVisible(s != 0))
        self.chk_det.stateChanged.connect(lambda s: self.peak_scatter.setVisible(s != 0))
        self.chk_park.stateChanged.connect(lambda s: self.parking.setVisible(s != 0))
        for c in (self.chk_thr, self.chk_det, self.chk_park):
            tb.addWidget(c)
        tb.addStretch(1)
        v.addWidget(toolbar)

        self.sweep_plot = pg.PlotWidget(); self.sweep_plot.setLabel("bottom", "Mesafe", units="m"); self.sweep_plot.setLabel("left", "Genlik")
        self.sweep_plot.showGrid(x=True, y=True, alpha=0.25); self.sweep_plot.addLegend(offset=(-10, 10))
        self.sweep_curve = self.sweep_plot.plot(pen=pg.mkPen(ACCENT, width=2), name="Sweep")
        self.thr_curve = self.sweep_plot.plot(pen=pg.mkPen(ACCENT2, width=2, style=QtCore.Qt.DashLine), name="Eşik")
        self.peak_scatter = pg.ScatterPlotItem(size=13, brush=pg.mkBrush(DANGER), pen=pg.mkPen("w", width=1), name="Tespit")
        self.sweep_plot.addItem(self.peak_scatter)
        v.addWidget(self.sweep_plot, 2)
        # Park sensörü / engel görseli (tahmini-mesafe grafiği yerine)
        self.parking = ParkingWidget(); v.addWidget(self.parking, 3)
        self.dist_info = hint(""); v.addWidget(self.dist_info)
        return w

    # ------------------------------------------------------------- EVENTS
    def _toggle_run(self):
        self.running = not self.running
        self.pause_btn.setText("Başlat" if not self.running else "Duraklat")

    def _on_source(self, text):
        self.use_hw = (text == "Donanım (A121)")
        for wgt in (self.conn_row, self.hw_connect):
            wgt.setVisible(self.use_hw)
        self.scene_group.setVisible(not self.use_hw)
        self.hw_port.setVisible(self.use_hw and self.conn_combo.currentText() in ("Seri Port", "Socket / IP"))
        self.cmd_q.put({"type": "source", "use_hw": self.use_hw})
        if not self.use_hw:
            self.hw_status.setText("Simülasyon aktif — donanım gerekmez."); self.hw_status.setStyleSheet(f"color:{MUTED};")

    def _on_conn_changed(self, text):
        self.hw_port.setVisible(self.use_hw and text in ("Seri Port", "Socket / IP"))
        self.hw_port.setPlaceholderText("COM4 veya /dev/ttyUSB0" if text == "Seri Port" else "192.168.0.1")

    def _build_client_kwargs(self):
        conn = self.conn_combo.currentText(); addr = self.hw_port.text().strip()
        if conn == "Mock (test)":
            return {"mock": True}
        if conn == "USB":
            return {"usb_device": True}
        if conn == "Seri Port":
            if not addr:
                raise ValueError("Seri port boş (ör. COM4).")
            return {"serial_port": addr}
        if conn == "Socket / IP":
            if not addr:
                raise ValueError("IP boş (ör. 192.168.0.1).")
            return {"ip_address": addr}
        raise ValueError("Bilinmeyen bağlantı tipi.")

    def _hw_connect(self):
        try:
            kwargs = self._build_client_kwargs()
            self._apply()   # önce güncel parametreleri worker'a gönder
            self.cmd_q.put({"type": "connect", "kwargs": kwargs})
            self.hw_status.setText("Bağlanılıyor…"); self.hw_status.setStyleSheet(f"color:{ACCENT2};")
        except Exception as e:
            self.hw_status.setText(f"Hata: {e}"); self.hw_status.setStyleSheet(f"color:{DANGER};")

    # ---------------------------------------------------- PARAMETRE -> WORKER
    def _apply(self, *_):
        p = self.params
        p.start_point = self.sp_start.value(); p.num_points = self.sp_num.value()
        p.step_length = int(self.step_combo.currentText())
        p.profile = rc.tx_db_to_profile(self.tx_db.value())     # dB -> profil
        p.hwaas = self.sl_hwaas.value(); p.receiver_gain = self.sl_gain.value()
        p.prf_key = self.prf_combo.currentText(); p.enable_tx = self.tx_check.isChecked()
        p.sweeps_per_frame = self.sp_spf.value()
        max_spf = p.max_sweeps_per_frame()
        if self.sp_spf.maximum() != max_spf:
            self.sp_spf.blockSignals(True); self.sp_spf.setMaximum(max_spf); self.sp_spf.blockSignals(False)
        if p.sweeps_per_frame > max_spf:
            p.sweeps_per_frame = max_spf
            self.sp_spf.blockSignals(True); self.sp_spf.setValue(max_spf); self.sp_spf.blockSignals(False)
        p.sweep_rate = float(self.sp_rate.value())
        p.amplitude_method = self.amp_combo.currentText()
        p.doppler_method = self.dopp_combo.currentText()
        p.window_type = {"Pencere yok": "none", "Hamming": "hamming",
                         "Hann": "hann"}[self.win_combo.currentText()]
        p.threshold_method = {"CFAR": "CFAR", "Kayıtlı eşik": "Kayıtlı"}.get(
            self.thr_combo.currentText(), "Sabit")
        p.sensitivity = self.sl_sens.value() / 100.0
        p.min_snr_db = float(self.sl_snr.value())
        p.detection_floor = float(self.sl_floor.value())
        self._update_derived_labels()
        self._send_state()

    def _send_state(self):
        self.cmd_q.put({
            "type": "params", "params": self.params,
            "noise_scale": self.sl_noise.value() / 100.0,
            "leakage": self.leak_check.isChecked(),
            "targets": self.sim_targets,
            "tab": self.tabs.currentIndex() if hasattr(self, "tabs") else 1,
            "subtract": self.bg_subtract.isChecked(),
            "adaptive": self.bg_adaptive.isChecked(),
            "alpha": 0.05,
        })

    def _update_derived_labels(self):
        p = self.params
        prof = p.profile
        warn = "" if rc.prf_profile_ok(p.prf_key, prof) else \
            f"  <span style='color:{DANGER};'>⚠ 19.5 MHz + profil 2 geçersiz</span>"
        self.tx_map_lb.setText(f"→ {self.tx_db.value():.1f} dB ≈ <b>profil {prof}</b> "
                               f"(RLG {rc.profile_tx_db(prof):.1f} dB){warn}")
        self.range_lb.setText(
            f"→ {p.start_m()*100:.1f} … {p.end_m()*100:.1f} cm  (adım {p.step_m()*1000:.1f} mm, "
            f"MMD {rc.prf_mmd(p.prf_key):.1f} m, MUR {rc.prf_mur(p.prf_key):.1f} m)")
        self.vel_lb.setText(f"→ hız çöz. {p.velocity_resolution_m_s():.3f} m/s, maks |v| {p.max_velocity_m_s():.2f} m/s")
        used, mx = p.buffer_usage(), p.BUFFER_MAX
        col = ACCENT if used <= mx else DANGER
        self.buf_lb.setText(f"<span style='color:{col}'>→ tampon {used}/{mx} (SPF ≤ {p.max_sweeps_per_frame()})</span>")

    # ------------------------------------------------------- SAHNE
    def _refresh_table(self):
        self.tbl.blockSignals(True); self.tbl.setRowCount(len(self.sim_targets))
        for i, t in enumerate(self.sim_targets):
            for c, val in enumerate((f"{t[0]:.2f}", f"{t[1]:.2f}", f"{t[2]:.2f}")):
                self.tbl.setItem(i, c, QtWidgets.QTableWidgetItem(val))
            chk = QtWidgets.QTableWidgetItem(); chk.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            chk.setCheckState(QtCore.Qt.Checked); self.tbl.setItem(i, 3, chk)
        self.tbl.blockSignals(False)

    def _on_scene_edit(self, r, c):
        try:
            t = list(self.sim_targets[r])
            if c in (0, 1, 2):
                t[c] = float(self.tbl.item(r, c).text())
            self.sim_targets[r] = tuple(t)
            self._send_state()
        except (ValueError, AttributeError):
            pass

    def _add_target(self):
        self.sim_targets.append((0.6, 0.5, 0.0)); self._refresh_table(); self._send_state()

    def _del_target(self):
        r = self.tbl.currentRow()
        if 0 <= r < len(self.sim_targets):
            self.sim_targets.pop(r); self._refresh_table(); self._send_state()

    # ------------------------------------------------------- ARKA PLAN
    def _bg_record(self):
        self.cmd_q.put({"type": "record", "n": self.bg_frames.value()})
        self.bg_status.setText(f"Kaydediliyor… ({self.bg_frames.value()} kare)"); self.bg_status.setStyleSheet(f"color:{ACCENT2};")

    def _bg_save(self):
        if self.background is None:
            self.bg_status.setText("Önce bir arka plan kaydet."); self.bg_status.setStyleSheet(f"color:{DANGER};"); return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Kaydet", "arkaplan.npz", "NumPy (*.npz)")
        if path:
            rc.save_background(path, self.background, self.params, env=self.rec_env)
            self.bg_status.setText(f"Kaydedildi: {os.path.basename(path)}"); self.bg_status.setStyleSheet(f"color:{ACCENT};")

    def _bg_load(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Yükle", "", "NumPy (*.npz)")
        if not path:
            return
        try:
            self.background, self.rec_env, self.bg_meta = rc.load_background(path)
            self.cmd_q.put({"type": "set_bg", "bg": self.background, "meta": self.bg_meta, "env": self.rec_env})
            self.bg_status.setText(f"Yüklendi: {os.path.basename(path)}"); self.bg_status.setStyleSheet(f"color:{ACCENT};")
        except Exception as e:
            self.bg_status.setText(f"Hata: {e}"); self.bg_status.setStyleSheet(f"color:{DANGER};")

    # ------------------------------------------------------- HAM VERİ KAYDI
    def _toggle_record(self):
        if not self.recording:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Ham veri kaydı", "ham_kayit.npz", "NumPy (*.npz)")
            if not path:
                return
            self.cmd_q.put({"type": "rec_start", "path": path})
            self.recording = True
            self.rec_btn.setText("■ Kaydı Durdur")
            self.rec_status.setText("● Kayıt sürüyor…"); self.rec_status.setStyleSheet(f"color:{DANGER};")
        else:
            self.cmd_q.put({"type": "rec_stop"})
            self.recording = False
            self.rec_btn.setText("● Kayda Başla")
            self.rec_status.setText("Kayıt durduruluyor / dosyaya yazılıyor…"); self.rec_status.setStyleSheet(f"color:{ACCENT2};")

    # ---------------------------------------------------------------- LOOP
    def _tick(self):
        # (a) OLAYLAR: kaybolmaması gereken durum mesajları (bağlantı/kayıt/arka plan)
        try:
            while True:
                ev = self.evt_q.get_nowait()
                kind = ev.get("kind")
                if kind == "conn":
                    self.hw_status.setText(ev["status"]); self.hw_status.setStyleSheet(f"color:{ACCENT if ev.get('ok') else DANGER};")
                elif kind == "rec":
                    self.rec_status.setText(ev["status"]); self.rec_status.setStyleSheet(f"color:{ACCENT if ev.get('ok') else DANGER};")
                elif kind == "bg":
                    self.bg_status.setText("Arka plan hazır. ‘gelen veriden çıkar’ ile aç."); self.bg_status.setStyleSheet(f"color:{ACCENT};")
        except queue.Empty:
            pass

        # (b) KARE: kuyruktaki en son sonucu al (varsa) -> daima en güncel kare
        res = None
        try:
            while True:
                res = self.out_q.get_nowait()
        except queue.Empty:
            pass
        if res is None:
            return
        if "error" in res:
            self.iq_info.setText(f"hata: {res['error']}"); return
        # canlı kayıt sayacı
        if res.get("rec_on") and self.recording:
            self.rec_status.setText(f"● Kayıt sürüyor… {res.get('rec_count', 0)} kare"); self.rec_status.setStyleSheet(f"color:{DANGER};")
        # arka planı GUI'de sakla (kaydet için)
        if res.get("bg") is not None:
            self.background = res["bg"]; self.bg_meta = res.get("bg_meta")
        self.rec_env = res.get("rec_env")

        if not self.running:
            return
        # FPS
        now = time.time(); dt = now - self._t_last; self._t_last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self.fps_lb.setText(f"{self._fps:4.0f} FPS")

        if res.get("tab") == 0:
            self._draw_raw(res)
        else:
            self._draw_distance(res)

    def _draw_raw(self, res):
        raw = res["raw"]; r = res["r"]
        # (1) HAM MATRİS: ne geliyorsa onu bas (numpy matris gösterimi)
        with np.printoptions(precision=1, suppress=True, threshold=240,
                             edgeitems=3, linewidth=200):
            mat = np.array2string(raw, separator=" ")
        spf, n = raw.shape
        header = (f"shape=(SPF,N)=({spf},{n})  dtype={raw.dtype}  (I+jQ kompleks)\n"
                  f"çıkarma={'AÇIK' if res.get('cfg_ok') and self.bg_subtract.isChecked() else 'kapalı'}\n")
        self.raw_text.setPlainText(header + mat)
        # (2) genlik / faz / menzil-hız
        amp = res["amp"]; self.amp_curve.setData(r, amp)
        pk = int(np.argmax(amp)); self.amp_peak.setData([r[pk]], [amp[pk]])
        self.phase_curve.setData(r, res["phase"])
        rd = res["rd"]; vel = res["vel"]; img = rd.T
        mx = max(img.max(), 1e-6)
        self.rd_img.setImage(img, autoLevels=False, levels=(0, mx))
        self.rd_img.setRect(QtCore.QRectF(r[0], vel[0], r[-1] - r[0], vel[-1] - vel[0]))
        self.rd_bar.setLevels((0, mx))
        self.iq_info.setText(f"Tepe {r[pk]*100:.1f} cm • SNR ≈ {res['snr']:.1f} dB • "
                             f"menzil–hız: {self.dopp_combo.currentText()}")

    def _draw_distance(self, res):
        r = res["r"]; amp = res["amp"]; thr = res["thr"]; d = res["d"]; st = res["st"]
        self.sweep_curve.setData(r, amp)
        valid = ~np.isnan(thr); self.thr_curve.setData(r[valid], thr[valid])
        self.peak_scatter.setData(d, st) if len(d) else self.peak_scatter.setData([], [])
        # park sensörü görseli
        dets = list(zip(list(d), list(st)))
        self.parking.set_data(float(r[0]), float(r[-1]), dets)
        txt = "yok" if not len(d) else ", ".join(f"{x*100:.1f}cm" for x in d[:6])
        self.dist_info.setText(f"Tespit ({len(d)}): {txt} • eşik: {self.thr_combo.currentText()} • "
                               f"min SNR: {self.params.min_snr_db:.0f} dB")

    # ----------------------------------------------------- RESET / KAPANIŞ
    def _reset_defaults(self, *_):
        self.sim_targets = [(0.30, 1.0, 0.0), (0.50, 0.7, 0.0)]
        self._refresh_table()
        p = SensorParams()
        for wgt in (self.sp_start, self.sp_num, self.step_combo, self.sp_spf, self.sp_rate,
                    self.tx_db, self.prf_combo, self.tx_check, self.amp_combo, self.dopp_combo,
                    self.win_combo, self.thr_combo, self.leak_check, self.bg_subtract, self.bg_adaptive):
            wgt.blockSignals(True)
        self.sp_spf.setMaximum(512)
        self.sp_start.setValue(p.start_point); self.sp_num.setValue(p.num_points)
        self.step_combo.setCurrentText(str(p.step_length)); self.sp_spf.setValue(p.sweeps_per_frame)
        self.sp_rate.setValue(p.sweep_rate); self.tx_db.setValue(rc.profile_tx_db(p.profile))
        self.prf_combo.setCurrentText(p.prf_key); self.tx_check.setChecked(p.enable_tx)
        self.amp_combo.setCurrentText(p.amplitude_method); self.dopp_combo.setCurrentText(p.doppler_method)
        self.win_combo.setCurrentText("Pencere yok")
        self.thr_combo.setCurrentText("Gürültü tabanı"); self.leak_check.setChecked(True)
        self.bg_subtract.setChecked(False); self.bg_adaptive.setChecked(False)
        for wgt in (self.sp_start, self.sp_num, self.step_combo, self.sp_spf, self.sp_rate,
                    self.tx_db, self.prf_combo, self.tx_check, self.amp_combo, self.dopp_combo,
                    self.win_combo, self.thr_combo, self.leak_check, self.bg_subtract, self.bg_adaptive):
            wgt.blockSignals(False)
        self.sl_hwaas.set_value(p.hwaas); self.sl_gain.set_value(p.receiver_gain)
        self.sl_sens.set_value(int(p.sensitivity * 100)); self.sl_snr.set_value(int(p.min_snr_db))
        self.sl_floor.set_value(int(p.detection_floor)); self.sl_noise.set_value(100)
        self.cmd_q.put({"type": "clear_bg"}); self.background = None; self.rec_env = None
        self._apply()

    def closeEvent(self, ev):
        self.stop_ev.set()
        super().closeEvent(ev)


def main():
    # Worker QApplication'dan ÖNCE başlatıldığından Linux'ta fork güvenlidir.
    # Windows/macOS'ta fork yoksa spawn'a düşülür (child, __main__ guard sayesinde
    # yalnızca acquisition_worker'ı yükler, GUI'yi tekrar başlatmaz).
    try:
        mp.set_start_method("fork")
    except (RuntimeError, ValueError):
        pass
    cmd_q = mp.Queue(); out_q = mp.Queue(maxsize=2); evt_q = mp.Queue(); stop_ev = mp.Event()
    worker = mp.Process(target=acquisition_worker, args=(cmd_q, out_q, evt_q, stop_ev), daemon=True)
    worker.start()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(QSS)
    win = MainWindow(cmd_q, out_q, evt_q, stop_ev)
    win.show()
    ret = app.exec()
    stop_ev.set()
    worker.join(timeout=1.0)
    if worker.is_alive():
        worker.terminate()
    sys.exit(ret)


if __name__ == "__main__":
    main()
