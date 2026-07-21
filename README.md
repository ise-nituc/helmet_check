# Helmet Check

Web カメラ映像をリアルタイムで解析し、人物が安全ヘルメットを着用しているかを表示する Python アプリケーションです。人物検出には YOLOv8n、ヘルメット検出には同梱の学習済みモデルを使用します。

## 1. 主な機能

- Web カメラ映像から人物とヘルメットをリアルタイム検出
- ヘルメットが人物の頭部付近にあるかを判定
- 人物ごとに `HELMET ON` または `NO HELMET` を表示
- 検出した人物数とヘルメット着用者数を画面上に表示
- ONNX 形式の人物検出モデルを同梱

## 2. 必要環境

- Windows、macOS、または Linux
- Python 3.10 以降
- Web カメラ
- 仮想環境の利用を推奨

GPU は必須ではありません。CPU でも動作しますが、カメラ解像度や PC の性能によって処理速度は変わります。

## 3. セットアップ

### 3.1. リポジトリへ移動

PowerShell の例:

```powershell
cd C:\path\to\helmet_check
```

### 3.2. 依存パッケージをインストール

`requirements.txt` には作成元環境固有の `file:///...` パスが含まれるため、別の PC では次の最小構成をインストールしてください。

```powershell
python -m pip install ultralytics onnxruntime opencv-python numpy
```

PyTorch モデルの実行環境を明示的に準備する必要がある場合は、PyTorch の公式サイトで環境に合うインストールコマンドを確認してください。

### 3.3. モデルを確認

以下のモデルは `src` に同梱されています。通常は追加のダウンロードや変換は不要です。

- `yolov8n.onnx`: 人物検出用の YOLOv8n モデル
- `helmet.pt`: ヘルメット検出用の学習済みモデル

`yolov8n.pt` も同梱されていますが、標準設定では ONNX 版を使用します。

## 4. 実行方法

モデルが相対パスで指定されているため、`src` ディレクトリから実行します。

```powershell
cd src
python main.py
```

起動すると、カメラの解像度がターミナルに表示され、映像ウィンドウが開きます。

- 緑: ヘルメットを着用している人物（`HELMET ON`）
- 赤: ヘルメット未着用と判定された人物（`NO HELMET`）
- 黄: 検出されたヘルメット

映像ウィンドウで `Esc` キーを押すと終了し、カメラが解放されます。

## 5. 判定方法

各フレームで人物とヘルメットをそれぞれ検出します。ヘルメットの中心が人物のバウンディングボックス内かつ上部 45% の領域にあり、ヘルメット領域の 15% 以上が人物領域と重なる場合、その人物はヘルメットを着用していると判定されます。

## 6. 設定の調整

設定値は [`src/main.py`](src/main.py) の `SafetyDetector.__init__()` にあります。

```python
self.target_resolution = (1920, 1080)
self.target_fps = 30
self.model_input_size = 416
self.person_confidence_threshold = 0.25
self.helmet_confidence_threshold = 0.75
self.min_helmet_inside_person = 0.15
self.helmet_person_height_ratio = 0.45
self.window_size = (800, 600)
```

- `person_confidence_threshold` / `helmet_confidence_threshold`: 検出として採用する最低信頼度です。誤検出が多い場合は上げ、見逃しが多い場合は下げます。
- `model_input_size`: 推論画像のサイズです。小さくすると高速になり、大きくすると小さな対象を検出しやすくなります。
- `helmet_person_height_ratio`: 人物領域のうちヘルメットを探索する上部の割合です。
- `target_resolution` / `target_fps`: カメラに要求する解像度とフレームレートです。処理が重い場合は下げてください。

複数のカメラを使用する場合は、`initialize_camera()` 内の `cv2.VideoCapture(0)` の番号を `1` などに変更します。

## 7. 参考

- 人物検出モデル: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- ヘルメット検出モデル: [Vansh2693/Helmet_Detection_OpenCV](https://github.com/Vansh2693/Helmet_Detection_OpenCV)

## 8. ディレクトリ構成

```text
helmet_check/
├── README.md
├── README_example.md
├── requirements.txt
└── src/
    ├── main.py
    ├── helmet.pt
    ├── yolov8n.onnx
    └── yolov8n.pt
```
