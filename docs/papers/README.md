# 論文置き場と検証状況

実装を原論文と照合するための PDF をここに置きます。`docs/papers/*.pdf` と `docs/papers/typst/` は
`.gitignore` に入っており、リポジトリには含めません（再配布しないため）。

## 進め方

1. 下の表の「ファイル名」で PDF を `docs/papers/` に置く。
2. 論文の式と定数を Typst に書き起こす（`docs/papers/typst/<同じ名前>.typ`）。古い論文は PDF が画像や
   崩れた OCR になっているので、照合はこの Typst 版を参照して行う。書き起こすのは式・定数表・記号の定義だけで、本文は写さない。
3. Typst 版と `docs/theory.typ`・コードを照合し、タグを書き換える。

### 照合の結果

集めた論文ごとに、照合済みのラベル（「導出」は論文の式から導いて確かめたもの）と、その論文では確かめられないことをまとめる。

| キー | 照合済み | 導出による照合 | この論文では確かめられないこと |
|---|---|---|---|
| McMurchie1978 | — | `eq:gpt`、`eq:hermite-exp`、`eq:E-rec`、`eq:ovlp-1d`、`eq:ovlp`、`eq:kin-1d`、`eq:kin`、`eq:dip-1d`、`eq:R-rec`、`eq:nuc-prim`、`eq:eri`（式番号・ページはプレプリントのもの） | `eq:nuc` の −Σ_C Z_C の重み（出典を Szabo1996 に変更）。雑誌版の式番号。プレプリントの誤植の疑い：式 (2.41) の1項の符号、式 (4.3) の因子 (−2α)^(N+L+M−n−ℓ−m) の欠落 |
| Adamson1999 | — | `eq:eri-erf`（論文は erfc 積分を与える。式 (1) と式 (11) の第2項から） | erf の積分表示、核の正定値性と Schwarz の不等式、中心微分の規則（`docs/theory.typ` で別に根拠を示した） |
| Gill1996 | — | — | `eq:eri-erf` の出典ではなかった |
| Kohn1965 | `eq:exc`（連続の LDA 積分、式 (2.3)） | `eq:slater`（式 (2.7)、(2.19) の μ_x から）、`eq:e-rks`、`eq:fock-rks`（どちらも a_HF = 0） | AO・密度行列による形、V_nn、GGA の項、厳密交換の混合、格子による求積 |
| Slater1951 | — | — | 本実装の係数（平均交換ポテンシャルは 3/2 倍で、エネルギー汎関数はない） |
| Vosko1980 | `eq:vwn`（式 [4.4]、Table 5 の説明と p. 1207 のパラメータ） | — | 「VWN5」「VWN-RPA」の名前、Gaussian の B3LYP が RPA 版を使うこと（Stephens1994） |
| Becke1988 | `eq:b88`、`eq:slater`（二次的な出典として）、`eq:cam-K` の B88 の K_s | 閉殻形（ρ_s = ρ/2） | — |
| Lee1988 | パラメータ a, b, c, d（p. 785）、C_F（式 (16)） | `eq:lyp`（式 (21) と式 (8) の t_W から部分積分） | ω, δ による書き方（Miehlich1989、未収集） |
| Becke1993 | a_0, a_x, a_c（式 (3)） | B3LYP の交換の係数 0.08 Slater + 0.72 B88（式 (2) から） | LYP + VWN の相関と VWN-RPA の選択（Stephens1994、未収集） |
| Yanai2004 | `eq:cam-b3lyp`（式 (9)、p. 53、Table 1）、`eq:cam-split`（式 (7)） | `eq:cam-k`、`eq:cam-fock`（式 (6)、(7)）、`eq:cam-b88`（式 (2)–(5)、(7)） | 解析的勾配（`eq:cam-grad` は DERIVED）、μ の単位（原子単位と仮定） |
| Iikura2001 | `eq:cam-K`（式 (2) と K^LSDA） | — | 式 (8) の添字 K_a は K_σ の誤植 |
| Tawada2004 | `eq:cam-a`（式 (5)–(7)）、`eq:cam-att`（式 (4)） | — | 大きい a_s での級数展開（`eq:cam-series` は DERIVED） |

## 入手できないもの

次の 4 つは手に入らないことが分かっています（2026-11-18、利用者の確認による）。探し直さないこと。

| キー | 文献 | 影響 | いまの扱い |
|---|---|---|---|
| McMurchie1978（雑誌版） | L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) の雑誌掲載版 | 式番号・ページが雑誌版と同じか確認できない | 手元のプレプリント（UCRL-13791）で照合済み。タグには「(preprint)」と書いてある。式の内容は数値でも確かめてある |
| GillAdamsonPople1996 | P. M. W. Gill, R. D. Adamson, J. A. Pople, *Mol. Phys.* **88**, 1005 (1996) | erf 側の減衰積分を直接載せた出典で確認できない | `eq:eri-erf` は Adamson1999 の erfc の式からの導出と数値積分で照合済み。これ以上の確認はしない |
| Helgaker2000 | T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory* (Wiley, 2000) | Gauss 関数の規格化、Boys 関数の級数と漸化式、基底関数の微分（タグ 12 か所）が文献と照合できない | `UNCHECKED(Helgaker2000)` のまま。別の出典を探すか、導出と数値検査で `DERIVED` に切り替える |
| Szabo1996 | A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry* (Dover, 1996) | 核間反発、対称直交化、2電子積分の対称性、双極子、核引力行列の重み（タグ 8 か所）が文献と照合できない | `UNCHECKED(Szabo1996)` のまま。同上 |

Boys 関数の定義と漸化式（`eq:boys`、`eq:boys-down`、`eq:boys-up`）は McMurchie1978 のプレプリント（式 (4.5)、(4.10)、(4.16)）にも載っているので、Helgaker2000 の代わりに使える。

## コード中のタグ

`docs/theory.typ` の式を引用しているコメント（各ファイルで式ごとに最初の1か所）に付けています。

| タグ | 意味 |
|---|---|
| `UNCHECKED(キー)` | 出典の論文はあるが、まだその論文と照合していない |
| `VERIFIED(キー, 場所)` | 論文と照合済み。「場所」は式番号・表・ページ |
| `DERIVED` | 出典なし。`docs/theory.typ` で導出し、数値的に確かめたもの |

- 一覧：`python tools/verification_tags.py status`、未検証の箇所：`grep -rn "UNCHECKED(" src`
- 照合が済んだら：`python tools/verification_tags.py verify eq:b88 Becke1988 "Eq. (8)"`
- 式と論文キーの対応は `tools/verification_tags.py` の `SOURCES` にあります。

PySCF との数値一致（`tests/`）は全項目で確認済みですが、それは「論文どおり」の確認とは別です。
タグは論文との照合だけを表します。

## 論文一覧

論文の DOI はすべて doi.org で解決し、著者・題名・巻・ページが一致することを確認済みです（書籍は未確認）。
「PDF」欄は 2026-11-18 時点で `docs/papers/` にあるかどうか。

### 分子積分

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| McMurchie1978 | L. E. McMurchie, E. R. Davidson, *J. Comput. Phys.* **26**, 218 (1978) | 10.1016/0021-9991(78)90092-X | `mcmurchie1978.pdf` | Hermite 展開係数と Hermite Coulomb 積分の漸化式、重なり・運動エネルギー・核引力・2電子積分の式 | あり（ただしタイプ打ちのプレプリント UCRL-13791、頁 -1-〜-20-。雑誌版の式番号は未確認） |
| Haser1989 | M. Häser, R. Ahlrichs, *J. Comput. Chem.* **10**, 104 (1989) | 10.1002/jcc.540100111 | `haser1989.pdf` | Schwarz の不等式による選別 | なし |
| Schlegel1995 | H. B. Schlegel, M. J. Frisch, *Int. J. Quantum Chem.* **54**, 83 (1995) | 10.1002/qua.560540202 | `schlegel1995.pdf` | デカルト d から球面調和 d への変換係数 | なし |
| Adamson1999 | R. D. Adamson, J. P. Dombroski, P. M. W. Gill, *J. Comput. Chem.* **20**, 921 (1999) | 10.1002/(SICI)1096-987X(19990715)20:9<921::AID-JCC3>3.0.CO;2-K | `adamson1999.pdf` | erfc（短距離）Coulomb 積分の計算法（PRISM の基本積分 [m]_ω、式 (11)–(13)）。erf の積分は式 (1) と式 (11) の第2項から従う。`eq:eri-erf` の出典（導出による）。Tawada 2004 が積分の評価について引用している論文 | あり |
| GillAdamsonPople1996 | P. M. W. Gill, R. D. Adamson, J. A. Pople, *Mol. Phys.* **88**, 1005 (1996) | 10.1080/00268979609484488 | `gill1996_molphys.pdf` | Coulomb 減衰交換汎関数と、Gauss 関数上の減衰した2電子積分（Yanai 2004 が引用している論文） | なし |
| Gill1996 | P. M. W. Gill, R. D. Adamson, *Chem. Phys. Lett.* **261**, 105 (1996) | 10.1016/0009-2614(96)00931-1 | `gill1996_attenuated.pdf` | 減衰演算子の族 CAP(m) の論文。書き起こしてみると、`eq:eri-erf`（一般の角運動量での erf 減衰積分）の直接の出典ではなかった。照合には上の2本が要る | あり |
| Helgaker2000 | T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory* (Wiley, 2000) | （書籍） | （書籍） | Gauss 関数の規格化、Boys 関数の級数と漸化式、基底関数の微分 | なし |
| Boys1950 | S. F. Boys, *Proc. R. Soc. A* **200**, 542 (1950) | 10.1098/rspa.1950.0036 | `boys1950.pdf` | Gauss 関数基底の原典（式の照合には Helgaker2000 を使う） | なし |

### SCF

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| Roothaan1951 | C. C. J. Roothaan, *Rev. Mod. Phys.* **23**, 69 (1951) | 10.1103/RevModPhys.23.69 | `roothaan1951.pdf` | Roothaan 方程式、密度行列、Coulomb・交換行列、エネルギー式 | なし |
| Pulay1980 | P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) | 10.1016/0009-2614(80)80396-4 | `pulay1980_diis.pdf` | DIIS の連立方程式 | なし |
| Pulay1982 | P. Pulay, *J. Comput. Chem.* **3**, 556 (1982) | 10.1002/jcc.540030413 | `pulay1982_diis.pdf` | 誤差ベクトル FPS − SPF | なし |
| VanLenthe2006 | J. H. Van Lenthe ほか, *J. Comput. Chem.* **27**, 926 (2006) | 10.1002/jcc.20393 | `vanlenthe2006_sad.pdf` | 原子密度の重ね合わせによる初期値 | なし |
| Almlof1982 | J. Almlöf, K. Faegri, K. Korsell, *J. Comput. Chem.* **3**, 385 (1982) | 10.1002/jcc.540030314 | `almlof1982.pdf` | 原子密度の初期値の原典 | なし |
| Szabo1996 | A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry* (Dover, 1996) | （書籍） | （書籍） | 核間反発、対称直交化、2電子積分の対称性、双極子 | なし |
| Golub2013 | G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed. (2013) | （書籍） | （書籍） | Householder 三重対角化、陰的シフト QL 法 | なし |

### 密度汎関数法

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| Kohn1965 | W. Kohn, L. J. Sham, *Phys. Rev.* **140**, A1133 (1965) | 10.1103/PhysRev.140.A1133 | `kohn1965.pdf` | KS 方程式 (2.6)–(2.10)、LDA の E_xc = ∫nε_xc (2.3)、交換ポテンシャル μ_x (2.19) | あり |
| Johnson1993 | B. G. Johnson, P. M. W. Gill, J. A. Pople, *J. Chem. Phys.* **98**, 5612 (1993) | 10.1063/1.464906 | `johnson1993.pdf` | 格子上の密度、交換相関行列、交換相関項の勾配 | なし |
| Dirac1930 | P. A. M. Dirac, *Math. Proc. Cambridge Philos. Soc.* **26**, 376 (1930) | 10.1017/S0305004100016108 | `dirac1930.pdf` | Slater（Dirac）交換の係数 | なし |
| Slater1951 | J. C. Slater, *Phys. Rev.* **81**, 385 (1951) | 10.1103/PhysRev.81.385 | `slater1951.pdf` | 平均交換ポテンシャル −3(3ρ/8π)^(1/3)（本実装の 3/2 倍。係数の出典ではない） | あり |
| Vosko1980 | S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980) | 10.1139/p80-159 | `vwn1980.pdf` | VWN の関数形と、VWN5・VWN-RPA のパラメータ | あり |
| Becke1988 | A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) | 10.1103/PhysRevA.38.3098 | `becke1988_b88.pdf` | B88 交換の式と β = 0.0042 | あり |
| Lee1988 | C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) | 10.1103/PhysRevB.37.785 | `lyp1988.pdf` | LYP 相関の式とパラメータ a, b, c, d | あり |
| Miehlich1989 | B. Miehlich, A. Savin, H. Stoll, H. Preuss, *Chem. Phys. Lett.* **157**, 200 (1989) | 10.1016/0009-2614(89)87234-3 | `miehlich1989.pdf` | ラプラシアンを消した LYP の形の ω, δ による書き方（実装の記法の出典。閉殻形そのものは Lee 1988 式 (21) から部分積分で従う） | なし |
| Becke1993 | A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) | 10.1063/1.464913 | `becke1993_b3.pdf` | 3パラメータ混成の係数 0.20, 0.72, 0.81（元の相関は Perdew–Wang の局所相関 + PW91 の勾配補正） | あり |
| Stephens1994 | P. J. Stephens ほか, *J. Phys. Chem.* **98**, 11623 (1994) | 10.1021/j100096a001 | `stephens1994_b3lyp.pdf` | B3LYP の定義（LYP + VWN の混合） | なし |
| Yanai2004 | T. Yanai, D. P. Tew, N. C. Handy, *Chem. Phys. Lett.* **393**, 51 (2004) | 10.1016/j.cplett.2004.06.011 | `yanai2004_cam-b3lyp.pdf` | CAM-B3LYP の定義、α = 0.19、β = 0.46、μ = 0.33 | あり |
| Iikura2001 | H. Iikura, T. Tsuneda, T. Yanai, K. Hirao, *J. Chem. Phys.* **115**, 3540 (2001) | 10.1063/1.1383587 | `iikura2001_lc.pdf` | GGA 交換の短距離化（減衰因子） | あり |
| Tawada2004 | Y. Tawada ほか, *J. Chem. Phys.* **120**, 8425 (2004) | 10.1063/1.1688752 | `tawada2004_lc-tddft.pdf` | 同じ減衰因子の再掲 | あり |
| Griewank2008 | A. Griewank, A. Walther, *Evaluating Derivatives*, 2nd ed. (SIAM, 2008) | （書籍） | （書籍） | 前進型自動微分（双対数）の規則 | なし |

### 数値積分格子

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| Becke1988grid | A. D. Becke, *J. Chem. Phys.* **88**, 2547 (1988) | 10.1063/1.454033 | `becke1988_grid.pdf` | Becke 分割の重み関数、原子の大きさの補正、動径格子の写像 | なし |
| Slater1964 | J. C. Slater, *J. Chem. Phys.* **41**, 3199 (1964) | 10.1063/1.1725697 | `slater1964_radii.pdf` | Bragg–Slater 半径の値 | なし |
| Press2007 | W. H. Press ほか, *Numerical Recipes*, 3rd ed. (2007) | （書籍） | （書籍） | Gauss–Legendre 求積の節点と重み | なし |

### 解析的勾配と構造最適化

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| Pulay1969 | P. Pulay, *Mol. Phys.* **17**, 197 (1969) | 10.1080/00268976900100941 | `pulay1969_forces.pdf` | SCF エネルギーの解析的勾配、エネルギー重み付き密度行列 | なし |
| Fletcher1970 | R. Fletcher, *Comput. J.* **13**, 317 (1970) | 10.1093/comjnl/13.3.317 | `fletcher1970_bfgs.pdf` | BFGS 更新式 | なし |
| Broyden1970 | C. G. Broyden, *IMA J. Appl. Math.* **6**, 76 (1970) | 10.1093/imamat/6.1.76 | `broyden1970.pdf` | 同上（4本のうち1本あれば足りる） | なし |
| Goldfarb1970 | D. Goldfarb, *Math. Comp.* **24**, 23 (1970) | 10.1090/S0025-5718-1970-0258249-6 | `goldfarb1970.pdf` | 同上 | なし |
| Shanno1970 | D. F. Shanno, *Math. Comp.* **24**, 647 (1970) | 10.1090/S0025-5718-1970-0274029-X | `shanno1970.pdf` | 同上 | なし |

### 分散補正

`docs/theory.typ` の節 `<sec:d3>`。2026-11-18 時点で 4 本とも未収集で、`eq:d3-*` はどれも論文と照合していない。
パラメータ表と式の一部（エネルギーの形、BJ・ゼロ減衰の関数、R_0 = sqrt(C_8/C_6)、s_r,8、α_n）は
Grimme 研究室の公開ページ（下の GrimmeD3web）で確かめた。

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| Grimme2010 | S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104 (2010) | 10.1063/1.3382344 | `grimme2010_d3.pdf` | D3 のエネルギー、配位数と k_1, k_2（共有結合半径のスケールの扱い）、C_6 の Gauss 補間と k_3、C_8 と Q_A の定義（係数の有無）、ゼロ減衰と α_n、原子対の R_0 の数表、C_6 参照値（補足資料）、B3LYP のパラメータを当てはめた汎関数の版（VWN-RPA / VWN5） | なし |
| Grimme2011 | S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011) | 10.1002/jcc.21759 | `grimme2011_d3bj.pdf` | D3(BJ) の式、R_0 = sqrt(C_8/C_6)、a_1, s_8, a_2 の表と a_2 の単位 | なし |
| Johnson2005 | E. R. Johnson, A. D. Becke, *J. Chem. Phys.* **123**, 024101 (2005) | 10.1063/1.1949201 | `johnson2005_bj.pdf` | 有理（BJ）減衰の元の形 | なし |
| Pyykko2009 | P. Pyykkö, M. Atsumi, *Chem. Eur. J.* **15**, 186 (2009) | 10.1002/chem.200800987 | `pyykko2009_rcov.pdf` | 単結合の共有結合半径の値 | なし |
| GrimmeD3web | Grimme 研究室 DFT-D3 の公開ページ（chemie.uni-bonn.de/grimme/de/software/dft-d3：bj_damping、zero_damping）と同所のプログラム説明書 man.pdf（Ver. 3.2 Rev 0, 2016） | — | — | 汎関数ごとのパラメータ（B3LYP、CAM-B3LYP）、式 (3)–(8)、既定の打ち切り（95 a_0、配位数 40 a_0）。2026-11-18 に閲覧 | — |

DOI は Crossref（Pyykko2009 は doi.org のハンドル解決）で著者・題名・巻・ページを確認した。

### 基底関数

データは Basis Set Exchange から取得したもので、原論文の表の値との照合はしていません。

| キー | 論文 | DOI | ファイル名 | 確認する内容 | PDF |
|---|---|---|---|---|---|
| Hehre1969 | W. J. Hehre, R. F. Stewart, J. A. Pople, *J. Chem. Phys.* **51**, 2657 (1969) | 10.1063/1.1672392 | `hehre1969_sto3g.pdf` | STO-3G の指数と係数 | なし |
| Ditchfield1971 | R. Ditchfield, W. J. Hehre, J. A. Pople, *J. Chem. Phys.* **54**, 724 (1971) | 10.1063/1.1674902 | `ditchfield1971.pdf` | 分割原子価基底の方法 | なし |
| Hehre1972 | W. J. Hehre, R. Ditchfield, J. A. Pople, *J. Chem. Phys.* **56**, 2257 (1972) | 10.1063/1.1677527 | `hehre1972_631g.pdf` | 6-31G の指数と係数 | なし |
| Hariharan1973 | P. C. Hariharan, J. A. Pople, *Theor. Chim. Acta* **28**, 213 (1973) | 10.1007/BF00533485 | `hariharan1973.pdf` | 分極関数の指数（6-31G\*、6-31G\*\*） | なし |
| Francl1982 | M. M. Francl ほか, *J. Chem. Phys.* **77**, 3654 (1982) | 10.1063/1.444267 | `francl1982.pdf` | 第3周期元素の 6-31G\* | なし |

### 定数

| キー | 出典 | 確認する内容 |
|---|---|---|
| CODATA2018 | NIST CODATA 2018 推奨値 | Bohr 半径 0.529177210903 Å |
