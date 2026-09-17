# Helmet Check

Web カメラ映像をリアルタイムで解析し、人物が安全ヘルメットを着用しているかを推定する Python アプリケーションです。同梱の `ppe_yolov8n.pt` を使用し、1 回の推論で人物（Person）とヘルメット（Hardhat）を検出します。

## 1. 主な機能

- Web カメラ映像から人物とヘルメットをリアルタイム検出
- ヘルメットが人物の頭部付近にあるかを判定
- 人物ごとに `HELMET ON` または `NO HELMET` を表示
- 検出した人物数とヘルメット着用者数を画面上に表示
- 人物とヘルメットを検出する統合モデルを同梱

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

アプリの実行に必要なパッケージをインストールします。

```powershell
python -m pip install ultralytics opencv-python numpy
```

リポジトリに記録されたバージョンを指定する場合は、`python -m pip install -r requirements.txt` を使用します。

PyTorch モデルの実行環境を明示的に準備する必要がある場合は、PyTorch の公式サイトで環境に合うインストールコマンドを確認してください。

### 3.3. モデルを確認

`src/ppe_yolov8n.pt` が標準の統合モデルです。通常は追加のダウンロードや変換は不要です。

別のモデルを使う場合は `SafetyDetector(model_path=...)` で指定できます。モデルには `Person` と `Hardhat` クラスが必要です（大文字・小文字は区別しません）。いずれかがない場合は `ValueError` が発生します。

## 4. 実行方法

リポジトリのルートから実行します。標準モデルのパスは `main.py` の場所を基準に解決されます。

```powershell
python src/main.py
```

起動すると、カメラの解像度がターミナルに表示され、映像ウィンドウが開きます。映像を左右反転し、416 × 416 ピクセルにリサイズして検出・表示します。

- 緑: ヘルメットを着用している人物（`HELMET ON`）
- 赤: ヘルメット未着用と判定された人物（`NO HELMET`）
- 黄: 検出されたヘルメット

映像ウィンドウで `Esc` キーを押すと終了し、カメラが解放されます。

## 5. 判定方法

人物のバウンディングボックスの上部 25%（横幅は人物ボックスと同じ）を頭部領域と推定し、ヘルメット領域との IoU（Intersection over Union）を計算します。

```text
IoU = intersection_area / (helmet_area + head_region_area - intersection_area)
```

IoU が 0.15 以上のヘルメットが 1 つでもあれば、その人物を `HELMET ON`、なければ `NO HELMET` と判定します。ヘルメット・人物・推定頭部領域の面積が 0 の場合は一致と判定しません。

頭部領域は実際の頭部検出ではなく推定であり、IoU は正しい装着状態を保証するものではありません。また、人物ごとに独立して判定するため、同じヘルメットが複数の人物に一致する場合があります。実際の映像に合わせて `min_helmet_head_iou` と `head_person_height_ratio` を調整してください。

## 6. 設定の調整

設定値は [`src/main.py`](src/main.py) の `SafetyDetector.__init__()` にあります。

```python
self.target_resolution = (1920, 1080)
self.target_fps = 30
self.model_input_size = 416
self.person_confidence_threshold = 0.25
self.helmet_confidence_threshold = 0.25
self.min_helmet_head_iou = 0.15
self.head_person_height_ratio = 0.25
self.window_size = (800, 600)
```

- `person_confidence_threshold` / `helmet_confidence_threshold`: 検出として採用する最低信頼度です。誤検出が多い場合は上げ、見逃しが多い場合は下げます。
- `model_input_size`: 推論・表示画像の一辺のサイズです。入力を正方形にリサイズするため、元の映像の縦横比は保持されません。
- `min_helmet_head_iou`: ヘルメットと推定頭部領域の IoU の最低値です。大きくすると一致条件が厳しくなります。
- `head_person_height_ratio`: 人物ボックスの高さに対する推定頭部領域の割合です。標準値の `0.25` は上部 25% を表します。
- `target_resolution` / `target_fps`: カメラに要求する解像度とフレームレートです。処理が重い場合は下げてください。

複数のカメラを使用する場合は、`initialize_camera()` 内の `cv2.VideoCapture(0)` の番号を `1` などに変更します。

## 7. 参考

- 検出ライブラリ: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- 統合モデルの出典（`main.py` に記載）: [Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection)

## 8. ディレクトリ構成

```text
helmet_check/
├── README.md
├── requirements.txt
└── src/
    ├── main.py
    └── ppe_yolov8n.pt
```
