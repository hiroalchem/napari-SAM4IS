# napari-SAM4IS

[English](README.md) | [日本語](README.ja.md)

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/napari-SAM4IS.svg?color=green)](https://github.com/hiroalchem/napari-SAM4IS/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-SAM4IS.svg?color=green)](https://pypi.org/project/napari-SAM4IS)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-SAM4IS.svg?color=green)](https://python.org)
[![tests](https://github.com/hiroalchem/napari-SAM4IS/workflows/tests/badge.svg)](https://github.com/hiroalchem/napari-SAM4IS/actions)
[![codecov](https://codecov.io/gh/hiroalchem/napari-SAM4IS/branch/main/graph/badge.svg)](https://codecov.io/gh/hiroalchem/napari-SAM4IS)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-SAM4IS)](https://napari-hub.org/plugins/napari-SAM4IS)


### Segment Anything Model (SAM) を用いたインスタンス・セマンティックセグメンテーションアノテーション用 napari プラグイン

これは Python 用の多次元画像ビューア [napari](https://napari.org/) のプラグインで、インスタンスセグメンテーションおよびセマンティックセグメンテーションのアノテーションを行えます。COCO 形式でのアノテーション出力にも対応しています。

----------------------------------

この [napari] プラグインは [Cookiecutter] を使い、[@napari] の [cookiecutter-napari-plugin] テンプレートから生成されました。

## インストール

**必要条件**: Python 3.10-3.13

### Step 1: napari-SAM4IS のインストール

[pip] でインストールできます:

```bash
pip install napari-SAM4IS
```

conda でのインストール:

```bash
conda install -c conda-forge napari-SAM4IS
```

最新の開発版をインストールする場合:

```bash
pip install git+https://github.com/hiroalchem/napari-SAM4IS.git
```

### Step 2: Segment Anything Model のインストール（オプション - ローカルモデル使用時のみ）

**注意**: Segment Anything Model のインストールは、ローカルモデルを使用する場合にのみ必要です。API モードを使用する場合はこのステップをスキップできます。

ローカルモデルを使用するには、SAM をインストールしてください:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

または、リポジトリをクローンしてソースからインストールすることもできます:

```bash
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .
```

詳しい手順は [SAM インストールガイド](https://github.com/facebookresearch/segment-anything#installation) を参照してください。

## 使い方
### 準備
1. napari で画像を開き、プラグインを起動します（プラグイン起動後に画像を開くことも可能です）。
2. プラグインを起動すると、SAM-Box、SAM-Positive、SAM-Negative、SAM-Predict、Accepted の各レイヤーが自動的に作成されます。各レイヤーの使い方は後述します。
3. ローカルモデルまたは API モードを選択します:
   - **ローカルモデルモード**: 使用するモデルを選択し、ロードボタンをクリックします（デフォルトのモデルが推奨です）。
   - **API モード**: 「Use API」チェックボックスをオンにし、API URL と API Key を入力します。モデルのロードは不要です。このモードは [LPIXEL Inc.](https://lpixel.net/) が [IMACEL](https://imacel.net/) を通じて提供する SAM API との連携を想定しています。API の利用に関するお問い合わせは [IMACEL](https://imacel.net/contact) へ直接ご連絡ください。
4. アノテーション対象の画像レイヤーを選択します。
5. インスタンスセグメンテーションとセマンティックセグメンテーションのどちらを行うかを選択します（現バージョンでは、3D 画像にはセマンティックセグメンテーションを選択してください）。
6. 出力レイヤーとして、インスタンスセグメンテーションの場合は「shapes」、セマンティックセグメンテーションの場合は「labels」を選択します（インスタンスセグメンテーションでは「Accepted」レイヤーも使用できます）。

### クラス管理
セグメンテーション結果に割り当てるアノテーションクラスを定義できます。

1. **Class Management** セクションで、クラス名を入力して **Add**（または Enter キー）をクリックすると新しいクラスが追加されます。
2. リスト内のクラスをクリックして選択します。選択したクラスが以降のアノテーションに割り当てられます。
3. クラスを再割り当てするには、出力 Shapes レイヤーで既存のアノテーションを選択し、目的のクラスをクリックします。
4. 使用中のクラスは削除できません。先に関連するアノテーションを削除してください。
5. YAML ファイルからクラス定義を読み込めます（**Load** をクリック）。想定されるフォーマット:
   ```yaml
   names:
     0: cat
     1: dog
     2: bird
   ```
6. クラス定義は COCO JSON 出力と同じディレクトリに `class.yaml` として自動保存されます。

### SAM によるアノテーション
1. SAM-Box レイヤーを選択し、矩形ツールでセグメンテーション対象のオブジェクトを囲みます。
2. 自動セグメンテーションマスクが生成され、SAM-Predict レイヤーに出力されます。
3. ポイントプロンプトを追加して予測を調整できます。SAM-Positive レイヤーをクリックして含めるべきポイントを追加し、SAM-Negative レイヤーをクリックして除外すべきポイントを追加します。
4. さらに調整が必要な場合は、SAM-Predict レイヤーで行います。
5. アノテーションを承認または拒否するには、キーボードで **A** または **R** を押します。
6. アノテーションを承認すると、セマンティックセグメンテーションの場合はラベル 1 として出力され、インスタンスセグメンテーションの場合はポリゴンに変換されて指定のレイヤーに出力されます。現在選択中のクラスがアノテーションに割り当てられます。
7. アノテーションを拒否すると、SAM-Predict レイヤーのセグメンテーションマスクは破棄されます。
8. 承認または拒否の後、SAM-Predict レイヤーは自動的にリセットされ、SAM-Box レイヤーに戻ります。

### 手動アノテーション（SAM なし）
**Manual Mode** を有効にすると、SAM を使わずにアノテーションを行えます。

1. **Manual Mode** チェックボックスをオンにします。SAM 関連のコントロールとレイヤーが非表示になります。
2. SAM-Predict レイヤーがペイントモードに切り替わります。レイヤーコントロールパネルから napari 標準の Labels ツール（ペイントブラシ、消しゴム、塗りつぶし）を使ってアノテーションを描画します。
3. ブラシサイズは napari 標準の Labels コントロールで調整します。
4. SAM モードと同様に、**A** で承認、**R** で拒否します。
5. 承認後、ペイントされたマスクはポリゴンに変換（インスタンスモード）または出力 Labels レイヤーにマージ（セマンティックモード）され、選択したクラスが割り当てられます。

### アノテーション属性
各アノテーションに品質管理ワークフローを支援する追加属性を設定できます。

1. 出力 Shapes レイヤーで1つ以上のアノテーションを選択します。
2. **Annotation Attributes** パネルで以下を設定できます:
   - **Unclear boundary**: オブジェクトの境界が曖昧なアノテーションをマークします。
   - **Uncertain class**: オブジェクトのクラスが不確定なアノテーションをマークします。
3. **Accept Selected** をクリックすると、選択したアノテーションがレビュー済みとしてマークされます（ステータスが "approved" に設定され、タイムスタンプが記録されます）。**Accept All** をクリックすると、すべてのアノテーションを一括でレビュー済みにできます。
4. 属性は COCO JSON 出力の各アノテーションの `"attributes"` フィールドに保存されます。
5. 複数のアノテーションが選択され、属性値が異なる場合、チェックボックスは混合状態のインジケータを表示します。

### アノテーションの保存と読み込み
1. labels レイヤーに出力した場合は、napari 標準の機能でマスクを保存できます。
2. shapes レイヤーに出力した場合は、napari 標準の機能で shapes レイヤーを保存するか、**Save** ボタンをクリックしてフォルダ内の各画像に対応する COCO 形式の JSON ファイルを出力できます（JSON ファイルは画像と同じ名前になります）。クラス定義も同じディレクトリに `class.yaml` として保存されます。
3. 以前保存したアノテーションを読み込むには、**Load** ボタンをクリックして COCO JSON ファイルを選択します。アノテーション、クラス定義、属性が復元されます。
4. 画像コンボボックスで画像を切り替えると、プラグインは以下の動作をします:
   - 未保存のアノテーションがある場合、保存を確認するダイアログを表示（Save / Discard / Cancel）
   - 出力レイヤーを自動的にクリア
   - 対応する JSON ファイルが存在する場合、アノテーションを自動読み込み



## コントリビュート

コントリビュートを歓迎します。テストは [tox] で実行できます。
プルリクエストを提出する前に、カバレッジが少なくとも同等以上であることを確認してください。

## ライセンス

[Apache Software License 2.0] ライセンスの下で配布されています。
「napari-SAM4IS」はフリーかつオープンソースのソフトウェアです。

## Issues

問題が発生した場合は、詳細な説明とともに [Issue を報告](https://github.com/hiroalchem/napari-SAM4IS/issues) してください。

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
