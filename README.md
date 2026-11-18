# qc_rs

Gauss 基底による閉殻 Hartree–Fock（RHF）と Kohn–Sham DFT（RKS）の Rust 実装です。
1点エネルギー・解析的エネルギー勾配・構造最適化ができ、分散補正 DFT-D3（BJ・ゼロ減衰）を加えられます。依存クレートは `rayon`（並列化）だけです。
数式はすべて `docs/theory.typ`（理論仕様書）に従い、コード中のコメントは仕様書の式ラベル（`eq:grad` など）を参照しています。

純 Rust で書かれており、C/Fortran のライブラリやシステムの BLAS を必要としません。
Windows・Linux・macOS のどれでも `cargo build` だけでビルドできることを要件にしています
（開発と検証は Windows で行いました。Linux と macOS での実行はまだ確かめていません）。

## できること・制限

- 閉殻（電子数が偶数、RHF/RKS）のみ。開殻、ECP、振動数計算はありません。
- 元素は H–Ar。基底は STO-3G、6-31G、6-31G\*、6-31G\*\*（殻は d まで、d は既定で球面 5 成分、デカルト 6 成分も可）。
- 2電子積分の扱いは2通りあります（`ScfOptions::integral_strategy`、`Calculation::with_integral_strategy`）。
  - **保存法**（`IntegralStrategy::Stored`）：8重対称性と Schwarz 選別を使って独立な積分をすべてメモリに保持します。
    メモリは基底数 N の4乗に比例し（約 N⁴/8 個の倍精度数。N = 256 で 4.3 GB、N = 425 で 33 GB）、
    小さい分子では最も速い方法です。
  - **直接法**（`IntegralStrategy::Direct`）：積分を保存せず、反復ごとに殻四つ組を計算して J・K に直接足し込みます。
    密度で重みづけた Schwarz 選別と、密度の差分 ΔP による Fock 行列の構築で計算を減らします。メモリは N² 程度です。
  - 既定は**自動**（`IntegralStrategy::Auto`）：保存するテンソルが `ScfOptions::eri_memory_budget`（既定 0.5 GiB、N ≈ 150 に相当）
    に収まれば保存法、超えれば直接法を使います。どちらを使ったかは `ScfResult::integral_strategy` に入ります。
    直接法の選別の閾値は `ScfOptions::direct_threshold`（既定 1e-12。エネルギーの誤差はその 10 倍程度）です。
    計測例は下の「保存法と直接法の速さとメモリ」にあります。
  勾配の微分積分はどちらの場合も保存せず、その場で縮約します。
- 汎関数：HF、Slater、SVWN5、SVWN-RPA、B88、BLYP、B3LYP（VWN-RPA 版 = PySCF の `b3lypg`）、B3LYP5、
  CAM-B3LYP（距離分離混成。局所相関は VWN5 = PySCF の `camb3lyp`）。
  CAM-B3LYP の保存法では誤差関数で減衰した2電子積分 erf(ωr)/r のテンソルをもう1つ持つので、2電子積分のメモリが2倍になります
  （直接法では2つの演算子の積分を同じ四つ組の中で続けて求めます）。
- DFT の格子は Becke 分割＋Gauss–Chebyshev 動径格子＋**直積型の角度格子**（Gauss–Legendre × 等間隔 φ、Lebedev ではない）で、刈り込みはしません。
  勾配では格子重みの核座標微分を無視します。その誤差（合力が 0 からずれる大きさ）は細かい格子で 1e-9〜2e-6（最大は H2CO）、既定の格子で 1e-6 程度です。
- 構造最適化は**デカルト座標の BFGS**（信頼半径つき）です。内部座標・対称性・遷移状態探索はありません。
- 分散補正は **DFT-D3 の2体項**だけで、減衰は **BJ**（D3(BJ)）と**ゼロ減衰**（D3(0)）です（`Calculation::with_dispersion`）。
  エネルギー・解析的勾配・構造最適化のすべてに入ります。既定では入りません。
  - 公表パラメータがあるのは B3LYP（B3LYP5 も同じ値）と CAM-B3LYP。他の手法（HF を含む）では `with_dispersion` はエラーで、
    パラメータを `D3Parameters::new` で自分で与えれば `with_dispersion_parameters` で使えます。
  - 3体項（Axilrod–Teller–Muto）はなく、距離の打ち切りもしません（全原子対を足す）。D2・D4 もありません。元素は H–Ar。
  - 参照値は dftd3 1.7.0（同じ条件：3体項なし・打ち切りなし）。一致度は下の「PySCF との比較」にあります。
  - B3LYP の D3 パラメータが VWN-RPA 版と VWN5 版のどちらに当てはめられたものかは未確認です（`docs/theory.typ` の分散補正の節）。

## 使い方

```rust
use qc_rs::{Calculation, D3Variant, Molecule, OptimizerSettings, Unit};

fn main() -> qc_rs::Result<()> {
    // 座標は Unit::Angstrom または Unit::Bohr
    let mol = Molecule::from_symbols(
        &["O", "H", "H"],
        &[[0.0, 0.0, 0.12], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]],
        Unit::Angstrom,
        0, // 電荷
    )?;
    // 手法は "hf" または汎関数名（"b3lyp", "blyp", "svwn5" など）
    let calc = Calculation::new(&mol, "hf", "6-31g*")?;
    let e = calc.energy()?.energy; // Hartree
    let g = calc.gradient()?.gradient; // dE/dR、原子ごとの [x, y, z]、Hartree/Bohr
    let opt = calc.optimize(&OptimizerSettings::default())?;
    println!("E = {e:.10}, dE/dx(O) = {:.6}", g[0][0]);
    println!("E_opt = {:.10} ({} steps)", opt.energy, opt.steps);

    // 分散補正 B3LYP-D3(BJ)：energy・gradient・optimize のすべてに E_disp とその勾配が入る
    let d3 = Calculation::new(&mol, "b3lyp", "6-31g*")?.with_dispersion(D3Variant::Bj)?;
    let sp = d3.energy()?;
    println!("E = {:.10} (E_disp = {:?})", sp.energy, sp.dispersion_energy);
    Ok(())
}
```

DFT の格子は `.with_grid(GridSettings::fine())`（既定は 75 × 18 × 36 点/原子）、デカルト d は `.with_cartesian_d(true)`、
SCF の設定は `.with_scf_options(...)` で変えられます。SCF の初期値は原子密度の重ね合わせです
（`ScfOptions::initial_guess` でコアハミルトニアンにも変えられますが、大きい分子では収束しないことがあります）。
XYZ ファイルは `Molecule::from_xyz(&text)` で読めます。勾配と最適化では SCF を DIIS 誤差 1e-10 まで収束させます
（勾配の誤差は密度の誤差の1次なので、既定の 1e-7 では 1e-7 程度の誤差が残るため）。
最適化が収束しなければエラー（`QcError::OptimizationNotConverged`）になります。
低水準の関数（`run_rhf`、`run_rks`、`gradient::rhf_gradient`、`gradient::rks_gradient`、`bfgs_minimize` など）も公開しています。
勾配の各項（1電子・2電子・重なり/W・交換相関・核間反発）は `GradientComponents` で個別に得られます。
分散補正を入れたときは、`SinglePoint::energy` と `GradientResult::energy`・`gradient` が SCF 部分と分散部分の和で、
分散部分は `dispersion_energy`・`dispersion_gradient` に別に入ります（`components` は SCF 部分だけ）。
分散補正だけを計算する関数 `dispersion::dispersion_energy`、`dispersion::dispersion_energy_and_gradient` もあります。

水分子の例（RHF/6-31G\*、B3LYP/6-31G、CAM-B3LYP/6-31G の1点計算・勾配・構造最適化）：

```sh
cargo run --release --example h2o
```

### 保存法と直接法の速さとメモリ

RHF の1点計算（SCF 全体、既定の収束条件、`cargo run --release --example scf_benchmark -- <分子> --basis <基底> [--cart] --strategy stored|direct`）。
AMD Ryzen 9 9900X（24 スレッド）、メモリ 31 GB、Windows 11。メモリはプロセスの最大ワーキングセットです。
「省略」は密度で重みづけた Schwarz 選別で省いた殻四つ組の割合（最初の構築／最後の ΔP による構築／最後の全体の構築）。

| 分子・基底 | N | 保存法 | 直接法 | 反復 | メモリ（保存／直接） | 省略（最初／最後の ΔP／最後の全体） |
|---|---|---|---|---|---|---|
| H2CO / 6-31G\* | 32 | 0.02 s | 0.04 s | 13 | 4 MB／8 MB | 37 %／2 %／0 % |
| ベンゼン / 6-31G\* | 96 | 0.71 s | 0.83 s | 10 | 224 MB／26 MB | 74 %／24 %／5 % |
| PhNHCOCH3 / 6-31G | 108 | 2.2 s | 2.1 s | 17 | 292 MB／23 MB | 82 %／62 %／31 % |
| PhNHCOCH3 / 6-31G\*（6d） | 168 | 10.0 s | 5.4 s | 17 | 1.4 GB／49 MB | 82 %／62 %／32 % |
| BarPh / 6-31G | 160 | 8.7 s | 6.4 s | 18 | 978 MB／40 MB | 90 %／77 %／51 % |
| BarPh / 6-31G\*（6d） | 256 | 57.3 s | 30.0 s | 18 | 4.8 GB／94 MB | 90 %／78 %／53 % |
| BarPh + PhNHCOCH3 二量体 / 6-31G\*（6d） | 424 | （32.5 GB 必要） | 74 s | 21 | —／229 MB | 95 %／90 %／69 % |

最後の行は `examples/molecules/barph_acetanilide_dimer.xyz`（27 重原子、テスト分子2つを 3.4 Å 以上離して並べた非結合二量体）で、
J/K の構築は1回あたり 2.2 s、残りは主に逐次の固有値分解（1反復あたり約 1.2 s）です。
同じ分子の勾配（SCF を DIIS 誤差 1e-10 まで収束）は 27 反復・117 s でした。
既定の自動選択では、上の表で N ≤ 168 の行は保存法、N = 256 以上は直接法になります（d 関数を含む N ≈ 150 以上では直接法のほうが速いこともあります）。

## ビルドとテスト

```sh
cargo build --release
cargo test          # 全テスト（数十秒、テストプロファイルは opt-level 3）
```

## 開発方針

変更は目的別のブランチで管理します：結果を変えない高速化は `Imp/`、式や実装の誤りの修正は `Bug/`、
計算理論の追加は `Feat/`。それぞれで守ることは [CONTRIBUTING.md](CONTRIBUTING.md) にあります。

## 理論仕様書

`docs/theory.typ`（Typst、日本語）。PDF は次で作ります：

```sh
typst compile docs/theory.typ
```

## PySCF との比較

参照値 `tests/reference/pyscf_reference.json` は PySCF 2.15.1（構造最適化は geomeTRIC）で作りました。
PySCF はブラックボックスとして数値だけを使っています。基底は `data/basis/*.gbs` を読んで PySCF に渡すので、両者は同じ基底データを使います。
再生成するには venv に PySCF と geomeTRIC を入れてから実行します：

```sh
python -m venv .venv
# Windows
.venv/Scripts/pip install pyscf==2.15.1 geometric
.venv/Scripts/python tests/reference/gen_pyscf_reference.py
# Linux / macOS
.venv/bin/pip install pyscf==2.15.1 geometric
.venv/bin/python tests/reference/gen_pyscf_reference.py
```

参照値の JSON はリポジトリに入っているので、`cargo test` に Python や PySCF は要りません。

分散補正 D3 の参照値 `tests/reference/d3_reference.json` は、dftd3 1.7.0（simple-dftd3 の Python API、LGPL。参照値を作るためだけに使い、同梱しない）で作りました。
減衰パラメータは汎関数名で読み込まず数値で渡し、3 体項は切り（s9 = 0）、カットオフは全原子間距離より大きくしています：

```sh
.venv/Scripts/pip install dftd3==1.7.0          # Linux / macOS は .venv/bin/
.venv/Scripts/python tests/reference/gen_d3_reference.py
```

D3 の一致度（`tests/dispersion_d3.rs`、B3LYP・CAM-B3LYP の BJ・ゼロ減衰、H–Ar の全元素とテスト分子）：エネルギーの相対差は
BJ で 3.8e-8 以下、ゼロ減衰で 6.9e-7 以下、勾配の差は BJ で 2.2e-11、ゼロ減衰で 6.4e-10 Eh/Bohr 以下です
（差は取り込んだ数表が単精度の値であることによる）。解析的勾配と中心差分の差は 6e-12 Eh/Bohr 以下です。

### テスト用分子の追加

`tests/molecules/` に XYZ ファイル（座標は Å）を置くと、テスト `tests/molecules_pyscf.rs` が自動で拾い、
RHF/6-31G と B3LYP/6-31G のエネルギーと勾配を PySCF の参照値と比べます。コードの変更は要りません。

1. `tests/molecules/<名前>.xyz` を置く。電荷があるときは 2 行目（コメント行）に `charge=-1` のように書く（なければ 0）。
   閉殻（電子数が偶数）、元素は H–Ar に限ります。
2. 参照値を作り直す（変わっていない分子は再計算しません）：
   `.venv/Scripts/python tests/reference/gen_molecule_reference.py`
3. `cargo test --test molecules_pyscf -- --nocapture` で確認する。
   一部だけ回すには環境変数 `QC_TEST_MOLECULES=名前1,名前2` を指定します。

参照値のない分子や、参照値を作ったあとで座標を変えた分子があると、テストは飛ばさずに失敗します。

達成した一致度（最大絶対偏差、原子単位）：

| 量 | ケース | 偏差 |
|---|---|---|
| RHF 全エネルギー | 11 分子/基底 | 1.2e-12 Eh |
| RHF 軌道エネルギー・双極子 | 同上（既定の SCF 収束） | 8.9e-8・1.1e-7 |
| RKS 全エネルギー | 13（細かい格子、PySCF は 150 × 1202 Lebedev） | 4.0e-8 Eh |
| RKS 軌道エネルギー・双極子 | 同上 | 4.8e-8・2.6e-8 |
| RHF 勾配 | 11（デカルト d、HCl を含む） | 6.9e-10 Eh/Bohr |
| RKS 勾配 | 12（細かい格子） | 1.6e-6 Eh/Bohr（H2CO。他は 7e-8 以下） |
| 最適化エネルギー | H2O・NH3 の RHF/6-31G\* | 4e-13 Eh |
| 最適化エネルギー | H2O・NH3 の B3LYP/6-31G（PySCF は 99 × 590 格子） | 8.2e-9 Eh |
| 最適化後の原子間距離 | 上の4件 | 4.2e-6 Bohr |
| CAM-B3LYP 全エネルギー | 6（細かい格子） | 3.1e-8 Eh |
| CAM-B3LYP 軌道エネルギー・双極子 | 同上 | 4.1e-8・9.5e-9 |
| CAM-B3LYP 勾配 | 同上 | 1.5e-6 Eh/Bohr（H2CO。他は 4e-8 以下） |
| CAM-B3LYP 最適化 | H2O（6-31G、PySCF は 99 × 590 格子） | エネルギー 8.4e-9 Eh、距離 9.2e-7 Bohr |

PySCF に依存しない検査として、エネルギーの中心差分（h = 1e-3 Bohr）との差は RHF/6-31G\* で 1.5e-7
（Richardson 外挿で 3e-10）、B3LYP/6-31G の細かい格子で 1.5e-7（同 2.5e-9）、CAM-B3LYP/6-31G の細かい格子で 1.5e-7（同 1.6e-9）、既定の格子で 1.3e-6
（格子重みの微分を無視しているため）です。RHF 勾配の並進・回転不変性（合力・トルク）は 1e-11 以下です。

## 支えているプロジェクト

qc_rs 自体が実行時に使うのは Rust の標準ライブラリと rayon だけですが、開発と検証は次のプロジェクトに支えられています。

| プロジェクト | 何に使っているか |
|---|---|
| [PySCF](https://pyscf.org/) | 参照値の生成（エネルギー・軌道エネルギー・勾配）。ブラックボックスとして実行するだけで、コードは読んでいません。テストの正しさの基準です |
| [Libxc](https://libxc.gitlab.io/) | PySCF が内部で使う汎関数ライブラリ。`b3lypg`、`vwn_rpa` などの名前はこのライブラリの定義に対応します（qc_rs からは使いません） |
| [geomeTRIC](https://github.com/leeping/geomeTRIC) | PySCF 側の構造最適化（参照となる最適化構造とエネルギー） |
| [Basis Set Exchange](https://www.basissetexchange.org/) | 基底関数データ（`data/basis/*.gbs`）の入手元 |
| [tad-dftd3](https://github.com/dftd3/tad-dftd3) / [tad-mctc](https://github.com/tad-mctc/tad-mctc) | DFT-D3 の数表（`data/dispersion/d3_h_ar.dat`、Apache-2.0）の入手元 |
| [simple-dftd3](https://github.com/dftd3/simple-dftd3)（PyPI の `dftd3` 1.7.0） | D3 の参照値（エネルギー・勾配）の生成。ブラックボックスとして実行するだけで、コードは読んでいません |
| [Typst](https://typst.app/) | 理論仕様書 `docs/theory.typ` の組版 |
| [Rust](https://www.rust-lang.org/) / [rayon](https://github.com/rayon-rs/rayon) | 実装言語と、2電子積分・数値積分の並列化 |
| [serde_json](https://github.com/serde-rs/json) | テストで参照値の JSON を読む（開発時のみの依存） |

## 背景となる論文

実装の各部分がどの論文に基づくかを示します。式そのものは `docs/theory.typ` にまとめてあります。
教科書としては Szabo & Ostlund *Modern Quantum Chemistry* と
Helgaker, Jørgensen & Olsen *Molecular Electronic-Structure Theory*（Wiley, 2000）の範囲です。

**分子積分**（`src/hermite.rs`、`src/integrals.rs`、`src/boys.rs`）

- S. F. Boys, *Proc. R. Soc. A* **200**, 542 (1950) — Gauss 関数を基底に使う提案。多中心積分が解析的に求まる根拠で、Boys 関数の名の由来。
- L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) — Gauss 関数の積を Hermite Gauss 関数で展開する方法。重なり・運動エネルギー・核引力・2電子積分をすべてこの漸化式で計算しています。
- M. Häser, R. Ahlrichs, *J. Comput. Chem.* **10**, 104 (1989) — Schwarz の不等式による2電子積分の選別。

**SCF**（`src/scf.rs`）

- C. C. J. Roothaan, *Rev. Mod. Phys.* **23**, 69 (1951) — Hartree–Fock 方程式を基底関数展開で行列の固有値問題にした Roothaan 方程式。
- P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980); *J. Comput. Chem.* **3**, 556 (1982) — DIIS による SCF の収束加速。後者の交換子 FPS − SPF を誤差に使う形を実装しています。

**密度汎関数法**（`src/dft.rs`、`src/functional/`）

- P. Hohenberg, W. Kohn, *Phys. Rev.* **136**, B864 (1964); W. Kohn, L. J. Sham, *Phys. Rev.* **140**, A1133 (1965) — 密度汎関数理論と Kohn–Sham 方程式。
- P. A. M. Dirac, *Math. Proc. Cambridge Philos. Soc.* **26**, 376 (1930) — 一様電子ガスの交換エネルギー（いわゆる Slater 交換の係数）。
- J. C. Slater, *Phys. Rev.* **81**, 385 (1951) — Hartree–Fock の交換を占有状態で平均した局所交換ポテンシャルで置き換える方法。そのポテンシャルはここで使うものの 3/2 倍です。
  ここで使う交換ポテンシャル（エネルギー密度の ρ 微分）は Kohn–Sham 1965 の式 (2.19) のものです。
- S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980) — 一様電子ガスの相関エネルギーの当てはめ式（VWN5 と VWN-RPA）。
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — 勾配補正つき交換汎関数 B88。
- C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — 相関汎関数 LYP。
- B. Miehlich, A. Savin, H. Stoll, H. Preuss, *Chem. Phys. Lett.* **157**, 200 (1989) — LYP から密度のラプラシアンを消した形と、その ω, δ による書き方。実装の書き方はこの論文に従いますが、閉殻形そのものは Lee–Yang–Parr 1988 の式 (21) から部分積分で従います。
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — 厳密交換を混ぜる3パラメータ混成汎関数。元の論文の相関は Perdew–Wang の局所相関に PW91 の勾配補正を加えたものです。
- P. J. Stephens, F. J. Devlin, C. F. Chabalowski, M. J. Frisch, *J. Phys. Chem.* **98**, 11623 (1994) — その相関を LYP + VWN にした B3LYP。
- H. Iikura, T. Tsuneda, T. Yanai, K. Hirao, *J. Chem. Phys.* **115**, 3540 (2001) — GGA 交換の長距離補正。誤差関数で分けた短距離部分の交換エネルギーを、汎関数の K_s と減衰因子で書く方法（短距離 B88 の式）。
- T. Yanai, D. P. Tew, N. C. Handy, *Chem. Phys. Lett.* **393**, 51 (2004) — CAM-B3LYP。Coulomb 演算子を α + β erf(ωr) で分ける形と、そのパラメータ（α = 0.19、β = 0.46、ω = 0.33）。

**数値積分格子**（`src/grid.rs`）

- A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988) — 分子の積分を原子ごとの球面格子に分ける重み関数（Becke 分割）と、原子の大きさによる補正、Gauss–Chebyshev 動径格子の写像。
- J. C. Slater, *J. Chem. Phys.* **41**, 3199 (1964) — 分割と動径格子の尺度に使う原子半径（Bragg–Slater 半径）。

**解析的勾配と構造最適化**（`src/gradient.rs`、`src/optimize.rs`）

- P. Pulay, *Mol. Phys.* **17**, 197 (1969) — 基底関数が核とともに動くことによる項（Pulay 力）を含む SCF エネルギーの解析的勾配。
- B. G. Johnson, P. M. W. Gill, J. A. Pople, *J. Chem. Phys.* **98**, 5612 (1993) — DFT のエネルギーと勾配を数値積分で実装する方法（交換相関項の勾配、格子重みの微分の扱い）。
- C. G. Broyden, R. Fletcher, D. Goldfarb, D. F. Shanno（いずれも 1970）— 準 Newton 法の BFGS 更新式。教科書としては J. Nocedal, S. J. Wright, *Numerical Optimization*（Springer）。

**分散補正 DFT-D3**（`src/dispersion.rs`、`data/dispersion/`）

- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104 (2010) — DFT-D3。配位数、C6 の参照値とその補間、C8、ゼロ減衰と原子対の R0。
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011) — D3 への Becke–Johnson 減衰の適用（D3(BJ)）とそのパラメータ。
- E. R. Johnson, A. D. Becke, *J. Chem. Phys.* **123**, 024101 (2005) — 有理（Becke–Johnson）減衰の元の形。
- P. Pyykkö, M. Atsumi, *Chem. Eur. J.* **15**, 186 (2009) — 配位数に使う共有結合半径。
- M. Friede, C. Hölzer, S. Ehlert, S. Grimme, *J. Chem. Phys.* **161**, 062501 (2024) — tad-dftd3 / tad-mctc。取り込んだ D3 の数表（H–Ar）の入手元です。
- 汎関数ごとの減衰パラメータは Grimme 研究室の DFT-D3 公開ページとプログラム説明書（Ver. 3.2 Rev 0, 2016）によります。
- エネルギー勾配の式（連鎖律）は `docs/theory.typ` で導いたもので、文献の式ではありません。

**基底関数**（`data/basis/`）

- W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657 (1969) — STO-3G。
- R. Ditchfield, W. J. Hehre, J. A. Pople, *J. Chem. Phys.* **54**, 724 (1971); W. J. Hehre, R. Ditchfield, J. A. Pople, *J. Chem. Phys.* **56**, 2257 (1972) — 分割原子価基底 6-31G。
- P. C. Hariharan, J. A. Pople, *Theor. Chim. Acta* **28**, 213 (1973) — 分極関数（6-31G\*、6-31G\*\*）。
- M. M. Francl ほか, *J. Chem. Phys.* **77**, 3654 (1982) — 第3周期元素（Na–Ar）の 6-31G\*。
- B. P. Pritchard ほか, *J. Chem. Inf. Model.* **59**, 4814 (2019) — Basis Set Exchange。

**検証に使ったソフトウェア**

- Q. Sun ほか, *WIREs Comput. Mol. Sci.* **8**, e1340 (2018); *J. Chem. Phys.* **153**, 024109 (2020) — PySCF。
- S. Lehtola ほか, *SoftwareX* **7**, 1 (2018) — Libxc。
- L.-P. Wang, C. Song, *J. Chem. Phys.* **144**, 214108 (2016) — geomeTRIC。
DOI つきの一覧と、各論文と実装を照合した結果は [docs/papers/README.md](docs/papers/README.md) にあります。

## 基底関数データ

`data/basis/` の Gaussian94 形式ファイルは [Basis Set Exchange](https://www.basissetexchange.org/) から取得したもので、
コンパイル時にライブラリに埋め込まれます。指数と係数は原論文で公表された数値です（上の「基底関数」の文献）。
Basis Set Exchange のソフトウェアとそこに収められたデータは BSD 3-Clause で配布されています
（Copyright (c) 2020 The Molecular Sciences Software Institute, Virginia Tech）。利用するときは
Pritchard ほか (2019) を引用してください。データ自体の再配布条件は Basis Set Exchange のサイトには明記されていません。

## ライセンス

BSD 3-Clause（`LICENSE` を参照）。

次の2つのデータは qc_rs の著作物ではありません（どちらもコンパイル時にライブラリに埋め込まれます）。

- 基底関数のデータ（`data/basis/`）は Basis Set Exchange に帰属します。
  出典・ライセンス・引用すべき文献は [data/basis/README.md](data/basis/README.md) にあります。
- 分散補正 DFT-D3 の数表（`data/dispersion/`）は tad-dftd3 0.7.0 と tad-mctc 0.8.0 から取り出したもので、
  両プロジェクトの作者に帰属し、Apache License 2.0 で配布されています（[data/dispersion/LICENSE-APACHE-2.0.txt](data/dispersion/LICENSE-APACHE-2.0.txt)）。
  値は Grimme ら（2010）にさかのぼります。出典・qc_rs が加えた変更・引用すべき文献は [data/dispersion/README.md](data/dispersion/README.md) にあります。
  ライブラリを組み込んだプログラムを配布するときは、この帰属表示と Apache License 2.0 の本文を同梱してください。
