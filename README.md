# qc_rs

Gauss 基底による閉殻 Hartree–Fock（RHF）と Kohn–Sham DFT（RKS）の Rust 実装です。
1点エネルギー・解析的エネルギー勾配・構造最適化ができます。依存クレートは `rayon`（並列化）と `faer`（対称行列の固有値分解と行列積）です。
数式はすべて `docs/theory.typ`（理論仕様書）に従い、コード中のコメントは仕様書の式ラベル（`eq:grad` など）を参照しています。

純 Rust で書かれており、C/Fortran のライブラリやシステムの BLAS を必要としません。
Windows・Linux・macOS のどれでも `cargo build` だけでビルドできることを要件にしています
（開発と検証は Windows で行いました。Linux と macOS での実行はまだ確かめていません）。

## できること・制限

- 閉殻（電子数が偶数、RHF/RKS）のみ。開殻、ECP、振動数計算はありません。
- 元素は H–Ar。基底は STO-3G、6-31G、6-31G\*、6-31G\*\*（殻は d まで、d は既定で球面 5 成分、デカルト 6 成分も可）。
- 2電子積分は8重対称性と Schwarz 選別を使ってすべてメモリに保持します（メモリは基底数の4乗に比例するため、小〜中程度の分子向け）。
  勾配の微分積分は保存せず、その場で縮約します。
- 汎関数：HF、Slater、SVWN5、SVWN-RPA、B88、BLYP、B3LYP（VWN-RPA 版 = PySCF の `b3lypg`）、B3LYP5。
- DFT の格子は Becke 分割＋Gauss–Chebyshev 動径格子＋**直積型の角度格子**（Gauss–Legendre × 等間隔 φ、Lebedev ではない）で、刈り込みはしません。
  勾配では格子重みの核座標微分を無視します。その誤差（合力が 0 からずれる大きさ）は細かい格子で 1e-9〜2e-6（最大は H2CO）、既定の格子で 1e-6 程度です。
- 構造最適化は**デカルト座標の BFGS**（信頼半径つき）です。内部座標・対称性・遷移状態探索はありません。

## 使い方

```rust
use qc_rs::{Calculation, Molecule, OptimizerSettings, Unit};

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
    Ok(())
}
```

DFT の格子は `.with_grid(GridSettings::fine())`（既定は 75 × 18 × 36 点/原子）、デカルト d は `.with_cartesian_d(true)`、
SCF の設定は `.with_scf_options(...)` で変えられます。勾配と最適化では SCF を DIIS 誤差 1e-10 まで収束させます
（勾配の誤差は密度の誤差の1次なので、既定の 1e-7 では 1e-7 程度の誤差が残るため）。
最適化が収束しなければエラー（`QcError::OptimizationNotConverged`）になります。
低水準の関数（`run_rhf`、`run_rks`、`gradient::rhf_gradient`、`gradient::rks_gradient`、`bfgs_minimize` など）も公開しています。
勾配の各項（1電子・2電子・重なり/W・交換相関・核間反発）は `GradientComponents` で個別に得られます。

水分子の例（RHF/6-31G\* と B3LYP/6-31G の1点計算・勾配・構造最適化）：

```sh
cargo run --release --example h2o
```

## ビルドとテスト

```sh
cargo build --release
cargo test          # 全テスト（数十秒、テストプロファイルは opt-level 3）
```

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

PySCF に依存しない検査として、エネルギーの中心差分（h = 1e-3 Bohr）との差は RHF/6-31G\* で 1.5e-7
（Richardson 外挿で 3e-10）、B3LYP/6-31G の細かい格子で 1.5e-7（同 2.5e-9）、既定の格子で 1.3e-6
（格子重みの微分を無視しているため）です。RHF 勾配の並進・回転不変性（合力・トルク）は 1e-11 以下です。

## 支えているプロジェクト

qc_rs 自体が実行時に使うのは Rust の標準ライブラリと rayon、faer だけですが、開発と検証は次のプロジェクトに支えられています。

| プロジェクト | 何に使っているか |
|---|---|
| [PySCF](https://pyscf.org/) | 参照値の生成（エネルギー・軌道エネルギー・勾配）。ブラックボックスとして実行するだけで、コードは読んでいません。テストの正しさの基準です |
| [Libxc](https://libxc.gitlab.io/) | PySCF が内部で使う汎関数ライブラリ。`b3lypg`、`vwn_rpa` などの名前はこのライブラリの定義に対応します（qc_rs からは使いません） |
| [geomeTRIC](https://github.com/leeping/geomeTRIC) | PySCF 側の構造最適化（参照となる最適化構造とエネルギー） |
| [Basis Set Exchange](https://www.basissetexchange.org/) | 基底関数データ（`data/basis/*.gbs`）の入手元 |
| [Typst](https://typst.app/) | 理論仕様書 `docs/theory.typ` の組版 |
| [Rust](https://www.rust-lang.org/) / [rayon](https://github.com/rayon-rs/rayon) | 実装言語と、2電子積分・数値積分の並列化 |
| [faer](https://codeberg.org/sarah-quinones/faer) | 対称行列の固有値分解と行列積（純 Rust、MIT）。連立方程式と、検証用の Jacobi 法は自前のまま |
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
- P. A. M. Dirac, *Math. Proc. Cambridge Philos. Soc.* **26**, 376 (1930); J. C. Slater, *Phys. Rev.* **81**, 385 (1951) — 一様電子ガスの交換エネルギー（Slater 交換）。
- S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980) — 一様電子ガスの相関エネルギーの当てはめ式（VWN5 と VWN-RPA）。
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — 勾配補正つき交換汎関数 B88。
- C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — 相関汎関数 LYP。
- B. Miehlich, A. Savin, H. Stoll, H. Preuss, *Chem. Phys. Lett.* **157**, 200 (1989) — LYP から密度のラプラシアンを消した形。実装はこの閉殻形です。
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — 厳密交換を混ぜる3パラメータ混成汎関数。
- P. J. Stephens, F. J. Devlin, C. F. Chabalowski, M. J. Frisch, *J. Phys. Chem.* **98**, 11623 (1994) — その相関を LYP + VWN にした B3LYP。

**数値積分格子**（`src/grid.rs`）

- A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988) — 分子の積分を原子ごとの球面格子に分ける重み関数（Becke 分割）と、原子の大きさによる補正、Gauss–Chebyshev 動径格子の写像。
- J. C. Slater, *J. Chem. Phys.* **41**, 3199 (1964) — 分割と動径格子の尺度に使う原子半径（Bragg–Slater 半径）。

**解析的勾配と構造最適化**（`src/gradient.rs`、`src/optimize.rs`）

- P. Pulay, *Mol. Phys.* **17**, 197 (1969) — 基底関数が核とともに動くことによる項（Pulay 力）を含む SCF エネルギーの解析的勾配。
- B. G. Johnson, P. M. W. Gill, J. A. Pople, *J. Chem. Phys.* **98**, 5612 (1993) — DFT のエネルギーと勾配を数値積分で実装する方法（交換相関項の勾配、格子重みの微分の扱い）。
- C. G. Broyden, R. Fletcher, D. Goldfarb, D. F. Shanno（いずれも 1970）— 準 Newton 法の BFGS 更新式。教科書としては J. Nocedal, S. J. Wright, *Numerical Optimization*（Springer）。

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

書誌情報は記憶に基づいて書いたもので、巻・ページは原典と照合していません。引用する際は原典で確認してください。

## 基底関数データ

`data/basis/` の Gaussian94 形式ファイルは [Basis Set Exchange](https://www.basissetexchange.org/) から取得したもので、
コンパイル時にライブラリに埋め込まれます。

## ライセンス

BSD 3-Clause（`LICENSE` を参照）。
