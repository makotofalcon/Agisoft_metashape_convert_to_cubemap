# Metashape Spherical → Cubemap → PostShot Converter

> **Fork of [MaikiOS/Agisoft_metashape_convert_to_cubemap](https://github.com/MaikiOS/Agisoft_metashape_convert_to_cubemap)**
> by [smert999](https://github.com/MaikiOS)
>
> This fork adds PostShot-ready COLMAP export and a Japanese UI.
> The original work — spherical-to-cubemap projection math, overlap handling, and Metashape integration — is entirely by smert999.

---

## このフォークについて

[smert999](https://github.com/MaikiOS) 氏による球面→キューブマップ変換スクリプトをベースに、[PostShot](https://postshot.app/) で直接読み込める COLMAP 形式出力と日本語 UI を追加したフォークです。

元リポジトリの優れた基盤コード（プロジェクション数学、オーバーラップ処理、Metashape 連携）に深く感謝します。

## What this fork adds / 追加機能

| 機能 | 元リポジトリ | このフォーク |
|---|---|---|
| キューブマップ変換 | Yes | Yes |
| COLMAP バイナリ出力 | 部分的 | 完全対応（PostShot 互換） |
| PostShot ドラッグ＆ドロップ | No | Yes |
| UI 言語 | ロシア語 | 日本語 |
| 不完全キューブマップの除外 | No | Yes（自動） |

## 使い方

### 前提条件

- Agisoft Metashape Professional 1.8+
- OpenCV（Metashape の Python 環境にインストール済みであること）

### ワークフロー

1. Metashape で球面画像をインポート
2. **Align Cameras** を実行（1回のみ）
3. **Tools → Run Script** から `postshot_converter.py` を実行
4. 出力フォルダを指定し、設定を確認して処理を開始
5. 生成されたフォルダを **PostShot にドラッグ＆ドロップ**

### 設定項目

| 項目 | デフォルト | 説明 |
|---|---|---|
| オーバーラップ | 10° | キューブマップ面間のオーバーラップ角度 |
| 面サイズ | 自動 | 各面の解像度（1024 / 2048 / 4096px または自動） |
| ポイント制限 | 50,000 | スパースポイントクラウドの最大点数 |

## 出力構造

```
output_folder/
├── images/           # キューブマップ画像（6面 × カメラ数）
├── sparse/0/         # COLMAP バイナリデータ
│   ├── cameras.bin   # カメラ内部パラメータ（PINHOLE）
│   ├── images.bin    # カメラ位置・姿勢
│   └── points3D.bin  # カラー付きスパースポイントクラウド
└── README.txt        # 使用パラメータの記録
```

## ファイル構成

| ファイル | 説明 |
|---|---|
| `postshot_converter.py` | **メインスクリプト（これを使用）** |
| `unified_fixed_v002.py` | smert999 氏によるベーススクリプト（差分比較用に同梱） |

過去バージョン（v007〜v012）は[元リポジトリ](https://github.com/MaikiOS/Agisoft_metashape_convert_to_cubemap)を参照してください。

## Credits

This project builds on the excellent work by **smert999** ([@MaikiOS](https://github.com/MaikiOS)):

- Spherical-to-cubemap projection with configurable overlap
- Metashape camera parameter extraction and virtual camera creation
- COLMAP binary format export pipeline
- Critical projection math fixes in `unified_fixed_v002.py`

Thank you for open-sourcing this work.

## License

MIT License — see [LICENSE.md](LICENSE.md)

Original work copyright (c) 2025 smert999.
Fork additions copyright (c) 2026 makotofalcon.
