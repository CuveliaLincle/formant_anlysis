import PySimpleGUI as sg
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 音声データを蓄積するためのバッファ
audio_data_accumulated = np.array([], dtype='float64')
fs = 44100  # サンプリングレートをグローバル変数として定義


# 利用可能なオーディオデバイスをリストアップ
def list_devices():
    devices = sd.query_devices()
    input_devices = [(i, device['name']) for i, device in enumerate(devices) if device['max_input_channels'] > 0]
    output_devices = [(i, device['name']) for i, device in enumerate(devices) if device['max_output_channels'] > 0]
    return input_devices, output_devices


input_devices, output_devices = list_devices()

# GUIレイアウト
layout = [
    [sg.Text("マイク選択:"), sg.Combo([name for i, name in input_devices], size=(30, 1), key='-MIC-')],
    [sg.Text("スピーカー選択:"), sg.Combo([name for i, name in output_devices], size=(30, 1), key='-SPEAKER-')],
    [sg.Text("ピッチ:"), sg.Text(size=(15, 1), key='-PITCH-')],
    [sg.Text("フォルマント:"), sg.Text(size=(30, 1), key='-FORMANTS-')],
    [sg.Text("倍音数:"), sg.Text(size=(30, 1), key='-HARMONICS-')],
    [sg.Button('開始'), sg.Button('停止'), sg.Button('Exit')],
    [sg.Canvas(key='-CANVAS-')],
]

# ウィンドウの作成
window = sg.Window("Voice Spectrum Analyzer", layout, finalize=True)

# matplotlib図の描画設定
fig, ax = plt.subplots()
fig_agg = FigureCanvasTkAgg(fig, window['-CANVAS-'].TKCanvas)
fig_agg.draw()
fig_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

stream = None


def start_recording(input_device_id, output_device_id):
    global stream
    if stream:
        stream.stop()
        stream.close()
    stream = sd.Stream(device=(input_device_id, output_device_id), samplerate=fs, channels=1, callback=audio_callback)
    stream.start()


def stop_recording():
    global stream
    if stream:
        stream.stop()
        stream.close()


def audio_callback(indata, outdata, frames, time, status):
    global audio_data_accumulated
    audio_data_accumulated = np.concatenate((audio_data_accumulated, indata[:, 0]))
    # outdata[:] = indata


# スペクトラムを線で描画するための関数
def plot_spectrum_line(ax, data, fs, pitch, formants, signal, harmonics_count):
    ax.clear()
    frequencies, times, Sxx = scipy.signal.spectrogram(data, fs=fs, nperseg=4000)
    Sxx_mean = np.mean(10 * np.log10(Sxx), axis=1)
    ax.plot(frequencies, Sxx_mean)
    # FFTを計算
    n_fft = len(signal) * 4
    fft_spectrum = np.fft.fft(signal, n=n_fft)
    frequencies = np.fft.fftfreq(n_fft, d=1 / fs)
    magnitude = np.abs(fft_spectrum)
    magnitude_db = 20 * np.log10(magnitude + 1e-6)  # dBスケールに変換
    # 倍音の表示（修正）
    if pitch > 0:
        for i in range(1, harmonics_count + 1):
            harmonic_freq = pitch * i
            idx = (np.abs(frequencies - harmonic_freq)).argmin()
            if idx < n_fft // 2:
                # yの値を修正して、グラフの下から正しい強度まで線を描画
                ax.vlines(frequencies[idx], -150, 0, color='r', linestyles='dashed')

    # フォルマントの表示（修正）
    for f in formants:
        if f < fs / 2:
            idx = (np.abs(frequencies - f)).argmin()
            # 同様に修正
            ax.vlines(frequencies[idx], -150, 0, color='g', linestyles='dotted')

    ax.set_xlim(0, 5000)
    ax.set_ylim(-110, -30)  # スペクトラムのスケールを固定
    ax.set_ylabel('Intensity [dB]')
    ax.set_xlabel('Frequency [Hz]')
    fig_agg.draw()


def detect_pitch_acf(signal, fs):
    """
    自己相関関数を用いてピッチを検出する改善版
    """
    # 自己相関関数を計算
    corr = scipy.signal.correlate(signal, signal, mode='full')
    corr = corr[len(corr) // 2:]

    # 最初のピークを見つけるために、低い周波数の範囲を無視
    min_index = fs // 500  # 例: 500Hzを最大ピッチとして最小インデックスを設定
    max_index = fs // 50  # 例: 50Hzを最小ピッチとして最大インデックスを設定
    corr[:min_index] = 0  # 最小ピッチに対応するインデックスまでの相関をゼロにする
    corr[max_index:] = 0  # 最大ピッチを超える相関をゼロにする

    # 最大ピーク（基本周波数に対応）を見つける
    peak_index = np.argmax(corr)
    if corr[peak_index] > 0:  # ピークが見つかった場合
        pitch = fs / peak_index
    else:  # ピークが見つからない場合
        pitch = 0
    return pitch


def estimate_formants_lpc(signal, fs):
    """
    librosaを用いてフォルマント周波数を推定する。第3から第5フォルマントを返す。
    """
    lpc_order = int(2 + fs / 1000)
    # LPC係数を計算
    a = librosa.lpc(y=signal, order=lpc_order)
    # LPC係数からルーツ（根）を計算
    rts = np.roots(a)
    # 実部が正のルーツのみを対象とする
    rts = [r for r in rts if np.imag(r) >= 0]
    # フォルマント周波数を計算
    angz = np.arctan2(np.imag(rts), np.real(rts))
    formants = sorted(angz * (fs / (2 * np.pi)))
    # フォルマントの周波数をロギング
    print(f"Formants: {formants}")
    # 第3から第5フォルマントを返す
    return formants[2:5]  # インデックスは0始まりなので、2:4で第3から第5を抽出


# ピッチとフォルマントを検出する関数（先に提供した関数を使用）
def detect_pitch_and_formants(signal, fs):
    pitch = detect_pitch_acf(signal, fs)
    formants = estimate_formants_lpc(signal, fs)
    return pitch, formants[:3]  # 最初の2つのフォルマントを返す


def count_harmonics(signal, fs, pitch, n_harmonics=10):
    if pitch <= 0:
        return 0

    # FFTの解析精度を向上
    n_fft = len(signal) * 4  # 元の信号の長さの4倍をFFTの点数とする
    fft_spectrum = np.fft.fft(signal, n=n_fft)
    frequencies = np.fft.fftfreq(n_fft, d=1 / fs)
    magnitude = np.abs(fft_spectrum)

    # 閾値を動的に設定
    threshold = np.median(magnitude) * 5  # 中央値の5倍以上を閾値とする

    harmonics_count = 0
    for i in range(2, n_harmonics + 2):  # 第1倍音（基本周波数）からn_harmonics倍音まで検出
        harmonic_freq = pitch * i
        idx = (np.abs(frequencies - harmonic_freq)).argmin()
        if magnitude[idx] > threshold:  # 閾値より大きい場合、倍音としてカウント
            harmonics_count += 1

    return harmonics_count


while True:
    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED or event == 'Exit':
        break
    elif event == '開始':
        input_device_name = values['-MIC-']
        output_device_name = values['-SPEAKER-']
        input_device_id = [device_id for device_id, name in input_devices if name == input_device_name][0]
        output_device_id = [device_id for device_id, name in output_devices if name == output_device_name][0]
        start_recording(input_device_id, output_device_id)
    elif event == '停止':
        stop_recording()

    # データ長が0.1秒分に達したらスペクトラムを描画
    if len(audio_data_accumulated) >= fs * 0.1:
        pitch, formants = detect_pitch_and_formants(audio_data_accumulated, fs)
        harmonics_count = count_harmonics(audio_data_accumulated, fs, pitch)
        formants_str = ', '.join(f"{f:.2f} Hz" for f in formants)
        window['-PITCH-'].update(f"{pitch:.2f} Hz")
        window['-FORMANTS-'].update(formants_str)
        window['-HARMONICS-'].update(f"{harmonics_count}")
        plot_spectrum_line(ax, audio_data_accumulated, fs, pitch, formants, audio_data_accumulated, harmonics_count)
        audio_data_accumulated = np.array([], dtype='float64')  # リセット

stop_recording()
window.close()
