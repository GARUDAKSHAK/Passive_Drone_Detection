"""
Garudakshak RF Dataset Capture
===============================
Passive drone detection — dataset collection script

Band    : 2400–2500 MHz (full ISM band)
Sweep   : 10 dwell positions × 10 MHz IBW = 100 MHz complete coverage
Classes : drone | wifi | bluetooth | environment
Storage : HDF5, complex64 IQ, both RX channels

Chain   : LPDA (14 dBi) → LaNA LNA → Taoglas BPF.24 → PlutoSDR 2RX

Usage
-----
    python capture_dataset.py

Select class from the menu, set up your RF source, press ENTER to capture.
Each class captures 10 dwells × 100 bursts = 1000 bursts total per class.
Re-run the menu to capture additional classes or top up existing ones.
"""

import adi
import numpy as np
import h5py
import time
import os
import sys


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  review every parameter before each session
# ══════════════════════════════════════════════════════════════════════════════

SDR_IP           = "ip:192.168.3.1"  # PlutoSDR default USB-network IP

SAMPLE_RATE      = int(10e6)         # 10 MSPS — reliable USB 2.0 ceiling
RX_BANDWIDTH     = int(10e6)         # 10 MHz IBW — exactly matches dwell spacing
BUFFER_SIZE      = 65536             # samples/burst -> 6.5536 ms per burst
                                     # long enough to capture multiple RC packets

# Gain: external chain already has 14 dBi antenna + ~16 dB LNA.
# Start at 30 dB. Reduce if CLIPPING warning appears. Increase for distant targets.
RX_GAIN_DB       = 30

PLL_SETTLE_S     = 0.15              # wait after LO retune for PLL lock
FLUSH_BURSTS     = 3                 # discard stale buffers after retune

# 10 dwells x 100 bursts = 1000 bursts per class (minimum viable: 50/dwell)
BURSTS_PER_DWELL = 100

# 10 center frequencies covering 2400-2500 MHz with no gaps (each +/-5 MHz)
DWELL_FREQS_HZ = [
    int(2.405e9),   # slice  1 :  2400-2410 MHz
    int(2.415e9),   # slice  2 :  2410-2420 MHz
    int(2.425e9),   # slice  3 :  2420-2430 MHz
    int(2.435e9),   # slice  4 :  2430-2440 MHz
    int(2.445e9),   # slice  5 :  2440-2450 MHz
    int(2.455e9),   # slice  6 :  2450-2460 MHz
    int(2.465e9),   # slice  7 :  2460-2470 MHz
    int(2.475e9),   # slice  8 :  2470-2480 MHz
    int(2.485e9),   # slice  9 :  2480-2490 MHz
    int(2.495e9),   # slice 10 :  2490-2500 MHz
]

CLASSES    = ["drone", "wifi", "bluetooth", "environment"]
OUTPUT_DIR = "dataset"
HDF5_PATH  = os.path.join(OUTPUT_DIR, "garudakshak_rf_dataset.h5")

# PlutoSDR 12-bit ADC: signed, half-scale = 2^11 = 2048 counts
ADC_FULL_SCALE = 2048.0

# Warn about ADC clipping if power gets within 2 dB of 0 dBFS
CLIP_WARN_DBFS = -2.0


# ══════════════════════════════════════════════════════════════════════════════
#  ANSI TERMINAL COLOURS
# ══════════════════════════════════════════════════════════════════════════════

USE_COLOUR = sys.stdout.isatty()


def _c(code, text):
    if not USE_COLOUR:
        return text
    return "\033[" + code + "m" + text + "\033[0m"


def green(t):   return _c("32;1", t)
def red(t):     return _c("31;1", t)
def yellow(t):  return _c("33;1", t)
def cyan(t):    return _c("36;1", t)
def bold(t):    return _c("1",    t)
def dim(t):     return _c("2",    t)


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE DISPLAY
#
#  The display box is exactly BOX_LINES tall.
#  After the first draw, each update moves the cursor up BOX_LINES lines and
#  redraws completely in-place — no scroll, no flicker.
# ══════════════════════════════════════════════════════════════════════════════

BOX_LINES = 12   # must equal the exact number of print() calls inside _draw_box


def _power_dbfs(samples):
    """Compute mean power of a complex IQ burst in dBFS (re: PlutoSDR 12-bit ADC)."""
    pwr_linear = float(np.mean(np.abs(samples) ** 2))
    return 10.0 * np.log10(max(pwr_linear, 1e-12)) - 20.0 * np.log10(ADC_FULL_SCALE)


def _power_bar(dbfs, width=22):
    """ASCII bar: maps -80 dBFS (silent) -> empty, 0 dBFS (full scale) -> full."""
    frac   = max(0.0, min(1.0, (dbfs + 80.0) / 80.0))
    filled = int(round(frac * width))
    empty  = width - filled

    if dbfs >= CLIP_WARN_DBFS:
        colour = red
    elif dbfs >= -20.0:
        colour = green
    elif dbfs >= -45.0:
        colour = yellow
    else:
        colour = dim

    return colour("*" * filled) + dim("." * empty)


def _channel_line(ch_idx, freq_hz, samples, is_active):
    """Build one formatted channel status line for the live display box."""
    label    = "RX" + str(ch_idx)
    freq_str = str(int(freq_hz // 1e6)) + " MHz"

    if not is_active:
        return ("  " + bold(label) + "  " + red("x NOT CONNECTED") +
                "  " + dim("uFL cable required for RX1"))

    dbfs     = _power_dbfs(samples)
    bar      = _power_bar(dbfs)
    dbfs_str = "{:+6.1f} dBFS".format(dbfs)

    if dbfs >= CLIP_WARN_DBFS:
        note = red("  WARNING: CLIPPING  reduce RX_GAIN_DB")
    elif dbfs < -70.0:
        note = dim("  (no signal detected)")
    else:
        note = ""

    return ("  " + bold(label) + "  " + green("ACTIVE") +
            "  " + freq_str.ljust(10) +
            "  " + cyan(dbfs_str) +
            "  [" + bar + "]" + note)


def _progress_bar(fraction, width=26, colour_fn=None):
    """ASCII progress bar for a given fraction 0.0-1.0."""
    filled = int(round(fraction * width))
    empty  = width - filled
    fill   = colour_fn("=" * filled) if colour_fn else ("=" * filled)
    return fill + dim("." * empty)


def _draw_box(class_name, dwell_i, freq_hz, burst_i,
              total_bursts_so_far, total_saved,
              ch0_samples, ch1_samples, dual_channel, elapsed_s):
    """
    Print the live status box.  Exactly BOX_LINES print() calls — keep accurate.
    """
    lo_mhz   = freq_hz / 1e6
    lo_low   = lo_mhz - RX_BANDWIDTH / 2e6
    lo_hi    = lo_mhz + RX_BANDWIDTH / 2e6
    total_target = len(DWELL_FREQS_HZ) * BURSTS_PER_DWELL

    frac_dwell   = burst_i / BURSTS_PER_DWELL
    frac_session = min(total_bursts_so_far / total_target, 1.0)

    eta_str = "---"
    if total_bursts_so_far > 0 and elapsed_s > 0:
        rate      = total_bursts_so_far / elapsed_s
        remaining = total_target - total_bursts_so_far
        eta_s     = remaining / max(rate, 0.001)
        eta_str   = "{:.0f}s".format(eta_s) if eta_s < 3600 else "{:.1f}min".format(eta_s / 60)

    W = 60

    # Line 1 — top border
    print("  +" + "=" * W + "+")
    # Line 2 — title
    title = " GARUDAKSHAK  RF  DATASET  CAPTURE "
    print("  |" + bold(title.center(W)) + "|")
    # Line 3 — class and dwell info
    info = "  CLASS: {}   DWELL {:2d}/10   {} MHz  [{:.0f}-{:.0f} MHz]".format(
        bold(class_name.upper()), dwell_i + 1,
        bold("{:.0f}".format(lo_mhz)), lo_low, lo_hi
    )
    print("  |" + info + " " * max(0, W - len(class_name) - 42) + "|")
    # Line 4 — separator
    print("  +" + "-" * W + "+")
    # Line 5 — RX0 channel
    print("  |" + _channel_line(0, freq_hz, ch0_samples, True) + " " * 2 + "|")
    # Line 6 — RX1 channel
    print("  |" + _channel_line(1, freq_hz, ch1_samples, dual_channel) + " " * 2 + "|")
    # Line 7 — separator
    print("  +" + "-" * W + "+")
    # Line 8 — dwell burst progress bar
    dwell_pct = int(frac_dwell * 100)
    dwell_bar = _progress_bar(frac_dwell, colour_fn=green)
    print("  |  Dwell burst  {:>3}/{:<3}  [{}]  {:>3}%  ETA: {}  |".format(
        burst_i, BURSTS_PER_DWELL, dwell_bar, dwell_pct, eta_str
    ))
    # Line 9 — session total progress bar
    sess_pct = int(frac_session * 100)
    sess_bar = _progress_bar(frac_session, colour_fn=cyan)
    print("  |  Session      {:>3}/{:<3}  [{}]  {:>3}%               |".format(
        total_bursts_so_far, total_target, sess_bar, sess_pct
    ))
    # Line 10 — saved count
    print("  |  Saved to HDF5: {:>5} bursts total                        |".format(
        total_saved
    ))
    # Line 11 — chain reminder
    chain_note = "  Chain: LPDA->LaNA->BPF.24->PlutoSDR   Gain: {} dB".format(RX_GAIN_DB)
    print("  |" + dim(chain_note) + " " * max(0, W - len(chain_note) + 2) + "|")
    # Line 12 — bottom border
    print("  +" + "=" * W + "+")


def _cursor_up(n):
    """Move terminal cursor up n lines (only if stdout is a TTY)."""
    if USE_COLOUR:
        sys.stdout.write("\033[" + str(n) + "A")
        sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  SDR INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_sdr():
    """
    Connect to PlutoSDR, configure for dual-channel RX.
    Falls back gracefully to single-channel if the 2RX uFL mod is absent.
    Returns (sdr_object, dual_channel_bool).
    """
    print("\n[SDR] Connecting to PlutoSDR at " + SDR_IP + " ...")
    try:
        sdr = adi.ad9361(SDR_IP)
    except Exception as e:
        print("[ERROR] Cannot reach PlutoSDR: " + str(e))
        print("        Check USB cable and confirm device is at 192.168.2.1")
        sys.exit(1)

    sdr.sample_rate     = SAMPLE_RATE
    sdr.rx_rf_bandwidth = RX_BANDWIDTH
    sdr.rx_buffer_size  = BUFFER_SIZE
    sdr.rx_lo           = DWELL_FREQS_HZ[0]

    dual_channel = False
    try:
        sdr.rx_enabled_channels     = [0, 1]
        sdr.gain_control_mode_chan0 = "manual"
        sdr.gain_control_mode_chan1 = "manual"
        sdr.rx_hardwaregain_chan0   = RX_GAIN_DB
        sdr.rx_hardwaregain_chan1   = RX_GAIN_DB
        test = sdr.rx()
        if isinstance(test, list) and len(test) == 2:
            dual_channel = True
            print(green("[SDR] Dual-channel RX confirmed (RX0 + RX1 via uFL mod)"))
        else:
            raise ValueError("single channel returned")
    except Exception:
        sdr.rx_enabled_channels     = [0]
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0   = RX_GAIN_DB
        print(yellow("[SDR] WARNING: Dual-channel unavailable — running RX0 only"))
        print(yellow("              Attach uFL cable to RX1 for dual-channel mode"))

    n_ch = 2 if dual_channel else 1
    print("[SDR] Sample rate   : " + str(SAMPLE_RATE // 1000000) + " MSPS")
    print("[SDR] IBW per dwell : " + str(RX_BANDWIDTH // 1000000) + " MHz")
    print("[SDR] Buffer size   : " + str(BUFFER_SIZE) +
          " samples  ({:.2f} ms)".format(BUFFER_SIZE / SAMPLE_RATE * 1e3))
    print("[SDR] Active chans  : " + str(n_ch))
    print("[SDR] Gain (manual) : " + str(RX_GAIN_DB) + " dB")

    return sdr, dual_channel


def retune_and_flush(sdr, freq_hz):
    """Retune LO, wait for PLL lock, discard FLUSH_BURSTS stale buffers."""
    sdr.rx_lo = int(freq_hz)
    time.sleep(PLL_SETTLE_S)
    for _ in range(FLUSH_BURSTS):
        sdr.rx()


# ══════════════════════════════════════════════════════════════════════════════
#  BURST CAPTURE
# ══════════════════════════════════════════════════════════════════════════════

def capture_one_burst(sdr, dual_channel):
    """
    Capture a single burst in one sdr.rx() call to preserve hardware
    time-alignment between both channels.

    Returns ndarray shape (2, BUFFER_SIZE) complex64  — dual channel
                          (1, BUFFER_SIZE) complex64  — single channel
    """
    raw = sdr.rx()
    if dual_channel:
        ch0 = np.asarray(raw[0], dtype=np.complex64)
        ch1 = np.asarray(raw[1], dtype=np.complex64)
        return np.stack([ch0, ch1], axis=0)
    else:
        ch0 = np.asarray(raw, dtype=np.complex64)
        return ch0[np.newaxis, :]


def capture_dwell_batch(sdr, dual_channel, class_name, dwell_i,
                        freq_hz, total_saved, session_start_t,
                        total_bursts_so_far):
    """
    Capture BURSTS_PER_DWELL bursts at the current LO frequency.
    Redraws the live display box in-place after every burst.

    Returns ndarray shape (BURSTS_PER_DWELL, n_channels, BUFFER_SIZE) complex64
    """
    n_ch   = 2 if dual_channel else 1
    batch  = np.empty((BURSTS_PER_DWELL, n_ch, BUFFER_SIZE), dtype=np.complex64)
    zeros  = np.zeros(BUFFER_SIZE, dtype=np.complex64)
    first  = True

    for i in range(BURSTS_PER_DWELL):
        burst     = capture_one_burst(sdr, dual_channel)
        batch[i]  = burst

        ch0_samp  = burst[0]
        ch1_samp  = burst[1] if dual_channel else zeros
        elapsed   = time.time() - session_start_t

        if first:
            _draw_box(class_name, dwell_i, freq_hz,
                      i + 1, total_bursts_so_far + i + 1, total_saved,
                      ch0_samp, ch1_samp, dual_channel, elapsed)
            first = False
        else:
            _cursor_up(BOX_LINES)
            _draw_box(class_name, dwell_i, freq_hz,
                      i + 1, total_bursts_so_far + i + 1, total_saved,
                      ch0_samp, ch1_samp, dual_channel, elapsed)

    return batch


# ══════════════════════════════════════════════════════════════════════════════
#  HDF5 STORAGE
# ══════════════════════════════════════════════════════════════════════════════

def open_or_create_hdf5(filepath, n_channels):
    """
    Open the HDF5 file in append mode.  Create per-class groups if absent.

    Layout:
      /<class>/
          iq_data          (N, n_channels, BUFFER_SIZE)  complex64
          center_freq_hz   (N,)  float64
          dwell_idx        (N,)  int32
          timestamp_unix   (N,)  float64
          attrs: sample_rate_hz, rx_bandwidth_hz, buffer_size, rx_gain_db, n_channels
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    f = h5py.File(filepath, "a")

    for cls in CLASSES:
        if cls not in f:
            grp = f.create_group(cls)
            grp.create_dataset(
                "iq_data",
                shape=(0, n_channels, BUFFER_SIZE),
                maxshape=(None, n_channels, BUFFER_SIZE),
                dtype=np.complex64,
                chunks=(1, n_channels, BUFFER_SIZE),
                compression="gzip",
                compression_opts=4,
            )
            grp.create_dataset("center_freq_hz",
                               shape=(0,), maxshape=(None,), dtype=np.float64)
            grp.create_dataset("dwell_idx",
                               shape=(0,), maxshape=(None,), dtype=np.int32)
            grp.create_dataset("timestamp_unix",
                               shape=(0,), maxshape=(None,), dtype=np.float64)

            grp.attrs["sample_rate_hz"]  = SAMPLE_RATE
            grp.attrs["rx_bandwidth_hz"] = RX_BANDWIDTH
            grp.attrs["buffer_size"]     = BUFFER_SIZE
            grp.attrs["rx_gain_db"]      = RX_GAIN_DB
            grp.attrs["n_channels"]      = n_channels

            print("[HDF5] Created group  /" + cls)
        else:
            n = f[cls]["iq_data"].shape[0]
            print("[HDF5] Opened group   /" + cls + "  (" + str(n) + " bursts stored)")

    return f


def write_dwell_batch(hdf5_file, class_name, batch, center_freq_hz, dwell_idx):
    """Append one dwell batch to the class group and flush to disk."""
    grp   = hdf5_file[class_name]
    n_old = grp["iq_data"].shape[0]
    n_new = batch.shape[0]
    n_end = n_old + n_new
    t_now = time.time()

    grp["iq_data"].resize(n_end, axis=0)
    grp["iq_data"][n_old:n_end] = batch

    grp["center_freq_hz"].resize(n_end, axis=0)
    grp["center_freq_hz"][n_old:n_end] = np.full(n_new, center_freq_hz, dtype=np.float64)

    grp["dwell_idx"].resize(n_end, axis=0)
    grp["dwell_idx"][n_old:n_end] = np.full(n_new, dwell_idx, dtype=np.int32)

    grp["timestamp_unix"].resize(n_end, axis=0)
    grp["timestamp_unix"][n_old:n_end] = np.full(n_new, t_now, dtype=np.float64)

    hdf5_file.flush()


# ══════════════════════════════════════════════════════════════════════════════
#  CLASS CAPTURE SESSION
# ══════════════════════════════════════════════════════════════════════════════

def capture_class_session(sdr, dual_channel, hdf5_file, class_name):
    """Sweep all 10 dwell positions for one class with live display."""
    t_capture  = BURSTS_PER_DWELL * (BUFFER_SIZE / SAMPLE_RATE)
    t_settle   = PLL_SETTLE_S + FLUSH_BURSTS * (BUFFER_SIZE / SAMPLE_RATE)
    t_est      = len(DWELL_FREQS_HZ) * (t_capture + t_settle)
    total_tgt  = len(DWELL_FREQS_HZ) * BURSTS_PER_DWELL

    print("\n  Class     : " + bold(class_name.upper()))
    print("  Target    : " + str(len(DWELL_FREQS_HZ)) + " dwells x " +
          str(BURSTS_PER_DWELL) + " bursts = " + str(total_tgt) + " bursts")
    print("  Per burst : {:.2f} ms  @ {} MSPS".format(
        BUFFER_SIZE / SAMPLE_RATE * 1e3, SAMPLE_RATE // 1000000))
    print("  Est. time : ~{:.0f}s  ({:.1f} min)\n".format(t_est, t_est / 60))

    session_start = time.time()
    bursts_done   = 0

    for dwell_i, freq_hz in enumerate(DWELL_FREQS_HZ):

        retune_and_flush(sdr, freq_hz)

        total_saved_before = hdf5_file[class_name]["iq_data"].shape[0]

        batch = capture_dwell_batch(
            sdr, dual_channel,
            class_name, dwell_i, freq_hz,
            total_saved_before,
            session_start,
            bursts_done,
        )

        write_dwell_batch(hdf5_file, class_name, batch, float(freq_hz), dwell_i)
        bursts_done += batch.shape[0]

        print("\n  " + green("SAVED") + "  dwell {}/10  {} MHz  {} bursts  "
              "({}/{} session)\n".format(
                  dwell_i + 1, int(freq_hz // 1e6),
                  batch.shape[0], bursts_done, total_tgt))

    total_saved = hdf5_file[class_name]["iq_data"].shape[0]
    elapsed     = time.time() - session_start
    print("\n  " + green("SESSION COMPLETE") + "  " + class_name.upper())
    print("  Bursts this session : " + str(bursts_done))
    print("  Total in file       : " + str(total_saved))
    print("  Time                : {:.0f}s\n".format(elapsed))


# ══════════════════════════════════════════════════════════════════════════════
#  MENU UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("\n" + "=" * 64)
    print("  " + bold("GARUDAKSHAK  --  RF Dataset Capture"))
    print("  Band   :  2400-2500 MHz  (10 dwell x 10 MHz, zero gaps)")
    print("  Target :  " + str(len(DWELL_FREQS_HZ) * BURSTS_PER_DWELL) +
          " bursts/class  x  " + str(len(CLASSES)) + " classes")
    print("  File   :  " + HDF5_PATH)
    print("=" * 64)


def print_status(hdf5_file):
    target = len(DWELL_FREQS_HZ) * BURSTS_PER_DWELL
    print("\n  -- Dataset status ------------------------------------------")
    for cls in CLASSES:
        if cls in hdf5_file:
            n      = hdf5_file[cls]["iq_data"].shape[0]
            frac   = min(n, target) / target
            w      = 20
            filled = int(round(frac * w))
            bar    = green("=" * filled) + dim("." * (w - filled))
            status = green("COMPLETE") if n >= target else yellow(str(n) + "/" + str(target))
            print("  {:<15}  [{}]  {}".format(cls, bar, status))
        else:
            print("  {:<15}  {}".format(cls, dim("not yet captured")))

    if os.path.exists(HDF5_PATH):
        size_mb = os.path.getsize(HDF5_PATH) / 1e6
        print("\n  File size : {:.1f} MB".format(size_mb))
    print("  ------------------------------------------------------------")


def print_field_notes():
    print("""
  FIELD SETUP REMINDERS
  ---------------------------------------------------------------
  drone        Power on drone + controller.  Fly or hover in open area.
               Capture at multiple distances: 5 m, 20 m, 50 m.
               Use at least 2 drone models if available.

  wifi         Ensure a WiFi AP is actively transmitting.
               Run a continuous file transfer to keep traffic up.
               Capture on channels 1, 6, and 11 separately if possible.

  bluetooth    Use a streaming BT device (headphones playing audio).
               Also capture a BLE device (e.g. phone broadcasting).
               Keep device within 5-10 m of the antenna.

  environment  Remove all intentional RF sources from the area.
               Capture ambient ISM background only.
               Do this both indoors and outdoors.
  ---------------------------------------------------------------""")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_banner()

    sdr, dual_channel = init_sdr()
    n_channels        = 2 if dual_channel else 1

    hdf5_file = open_or_create_hdf5(HDF5_PATH, n_channels)

    try:
        while True:
            print_status(hdf5_file)
            print()
            print("  SELECT ACTION")
            print("  ------------------------------------------------------------")
            target = len(DWELL_FREQS_HZ) * BURSTS_PER_DWELL
            for i, cls in enumerate(CLASSES):
                n    = hdf5_file[cls]["iq_data"].shape[0] if cls in hdf5_file else 0
                tick = green("[OK]") if n >= target else "    "
                print("  [{}] {}  Capture  {:<15}  ({} bursts)".format(
                    i + 1, tick, cls, n))
            print("  [5]      Show field setup reminders")
            print("  [6]      Exit")
            print("  ------------------------------------------------------------")

            choice = input("  Choice: ").strip()

            if choice in {"1", "2", "3", "4"}:
                selected = CLASSES[int(choice) - 1]
                print_field_notes()
                input("\n  Press ENTER when ready to capture " +
                      bold(selected.upper()) + " ...")
                capture_class_session(sdr, dual_channel, hdf5_file, selected)

            elif choice == "5":
                print_field_notes()

            elif choice == "6":
                break

            else:
                print("  Invalid choice.")

    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted -- partial data has been flushed to disk.")

    finally:
        hdf5_file.flush()
        hdf5_file.close()
        print("[INFO] HDF5 closed: " + HDF5_PATH)
        print("[INFO] All captured data saved.")


if __name__ == "__main__":
    main()