# Duck Pixiv Assistant (Pixiv 投稿アシスタント)

ImageStudio で生成したイラストを Pixiv 向けに最適化・タグ付け・タイトル生成し、確認後にワンクリックで投稿できるスタンドアロンのデスクトップ支援ツールです。

> **注意**: 本アプリケーションは常駐型デーモンではなく、ユーザーが手動で画像を選択した時のみ動作する投稿支援アシスタントです。既存の ImageStudio 本体（フロントエンド、バックエンド、API、ComfyUI連携、モデルロード等）には一切干渉しません。

---

## 主な機能

1. **メタデータ自動解析**:
   - PNG 内包メタデータ (`parameters`, `prompt`) および ImageStudio の出力メタデータ (`outputs/metadata/*.json`) から、キャラクター名、衣装、表情、ポーズ、素材、モデル情報を自動抽出。
2. **制御された日本語タグシステム (`data/tags.json`)**:
   - 英語のプロンプト概念を辞書に基づいて正確な日本語タグへ変換。
   - LLM が勝手に未知のタグを捏造しないよう厳密に管理。
   - 常に付けたいタグ (`data/fixed_tags.json`) と統合し、重複排除・優先度順位付けを行い最大10個に調整。
3. **キャラクター / 作品マスター (`data/characters.json`)**:
   - キャラクター名から公式日本語表記および作品名（例: `Belfast` → `ベルファスト`, `アズールレーン`）を自動マッピング。
4. **自然な日本語タイトル候補の自動生成**:
   - 直訳調ではない、Pixiv らしい魅力的で詩的なタイトルを 3〜5 個提示。
   - 気に入ったタイトルをワンクリックで採用 (`[採用]`) または直接編集可能。
5. **マルチ画像作品対応**:
   - 複数枚の画像を1つの Pixiv 作品として投稿可能。
   - 画像の並び替え（ドラッグまたは上下移動）、個別削除が可能。
6. **完全日本語レビュー画面**:
   - 投稿前にタイトル、キャプション、タグ（タグチップの追加/削除）、年齢制限（全年齢/R-18/R-18G）、AI生成フラグを柔軟に編集可能。
   - 「投稿する」ボタンを押すまで勝手に外部送信されません。
7. **SQLite による投稿履歴管理**:
   - 投稿した作品、タグ、日時、Pixiv ID、エラー履歴をローカルデータベース (`database/duck_pixiv.db`) で管理。

---

## 起動方法

### Windows
`duck_pixiv` フォルダ内の `run.bat` をダブルクリックして実行します。

またはターミナルから:
```powershell
cd d:\AI\imageStudio\duck_pixiv
python app\main.py
```

ブラウザで `http://127.0.0.1:8765` を開きます。

---

## ディレクトリ構成

```
duck_pixiv/
├── app/
│   ├── main.py               # サーバー起動エントリポイント
│   ├── metadata_reader.py    # PNG および ImageStudio メタデータ抽出
│   ├── image_analyzer.py     # 画像フォールバック解析
│   ├── tag_generator.py      # タグ辞書マッピング・重複排除・優先順位
│   ├── title_generator.py    # 日本語タイトル・キャプション生成
│   ├── post_manager.py       # 投稿データ統合管理・保存
│   ├── pixiv_uploader.py     # 独立した Pixiv 投稿インターフェース
│   └── database.py           # SQLite 投稿履歴管理
├── data/
│   ├── tags.json             # 編集可能な日本語タグ辞書
│   ├── fixed_tags.json       # 固定タグ定義 (AIイラスト, 巨乳 等)
│   ├── characters.json       # キャラクター・作品定義
│   └── settings.json         # アプリケーション設定
├── metadata/                 # 準備された投稿メタデータJSON
├── database/                 # SQLite データベース (duck_pixiv.db)
├── ui/                       # 日本語 Web UI (HTML / Tailwind / Vanilla JS)
├── run.bat                   # 起動スクリプト
└── README.md
```
