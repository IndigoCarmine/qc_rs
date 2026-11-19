// qc_rs 理論仕様書 — 実装はこの文書の式番号を参照する
#set document(title: "qc_rs 理論仕様書")
#set page(paper: "a4", margin: (x: 22mm, y: 24mm), numbering: "1")
#set text(lang: "ja", font: ("New Computer Modern", "Yu Mincho", "MS Mincho"), size: 10pt)
#set par(justify: true, leading: 0.8em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#show heading: it => block(above: 1.4em, below: 0.8em, it)

#let vb(x) = $bold(#x)$
#let bra(x) = $lr(chevron.l #x |)$
#let ket(x) = $lr(| #x chevron.r)$
#let asinh = math.op("asinh")
#let erf = math.op("erf")
#let Tr = math.op("Tr")

#align(center)[
  #text(size: 17pt, weight: "bold")[qc_rs 理論仕様書] \
  #v(2mm)
  閉殻 Hartree–Fock / Kohn–Sham DFT の1点エネルギー・解析的勾配・構造最適化
]

#outline(depth: 2)

= 記法と単位 <sec:units>

すべて原子単位（エネルギー: Hartree $E_h$、長さ: Bohr $a_0$）で計算する。入力がオングストロームのときは

$ 1 thin a_0 = 0.529177210903 thin angstrom $ <eq:bohr>

で換算する（CODATA 2018）。

- $A, B, C, D$：原子（核）。位置 $vb(R)_A = (A_x, A_y, A_z)$、核電荷 $Z_A$、原子数 $N_"atom"$
- $mu, nu, lambda, sigma$：原子軌道（AO）基底関数 $chi_mu (vb(r))$。実関数。個数 $N$
- $i, j$：占有分子軌道。閉殻なので占有軌道数は $N_"occ" = N_"elec" slash 2$（$N_"elec"$ は偶数であること）
- $(2k-1)!!$：二重階乗。$(-1)!! = 1$、$1!! = 1$、$3!! = 3$、$5!! = 15$

核間反発エネルギーとその微分：

$ V_"nn" = sum_(A < B) (Z_A Z_B) / R_(A B), quad R_(A B) = abs(vb(R)_A - vb(R)_B) $ <eq:vnn>

$ (partial V_"nn") / (partial vb(R)_A) = - sum_(B != A) Z_A Z_B (vb(R)_A - vb(R)_B) / R_(A B)^3 $ <eq:vnn-grad>

= 基底関数 <sec:basis>

== 原始デカルト Gauss 関数

中心 $vb(A)$、指数 $a$、角運動量指数 $(l, m, n)$ の（規格化していない）原始関数

$ g_(l m n)(vb(r); a, vb(A)) = x_A^l thin y_A^m thin z_A^n thin e^(-a r_A^2), quad vb(r)_A = vb(r) - vb(A) $ <eq:prim>

全角運動量は $L = l + m + n$。規格化定数は

$ N(a; l, m, n) = (2a / pi)^(3 slash 4) sqrt((4a)^(l+m+n) / ((2l-1)!! (2m-1)!! (2n-1)!!)) $ <eq:prim-norm>

== 短縮殻

角運動量 $L$ の殻は指数 $a_k$ と短縮係数 $d_k$（$k = 1 ... K$）を持つ。基底ファイルの $d_k$ は
*規格化された原始関数* に対する係数である。殻の成分 $(l, m, n)$ の基底関数は

$ chi_(l m n)(vb(r)) = c_(l m n) sum_(k=1)^K tilde(d)_k thin g_(l m n)(vb(r); a_k, vb(A)) $ <eq:contracted>

$ tilde(d)_k = tilde(N) thin d_k thin N(a_k; L, 0, 0), quad
  tilde(N) = [sum_(k, k') d_k d_(k') ((2 sqrt(a_k a_(k'))) / (a_k + a_(k')))^(L + 3 slash 2)]^(-1 slash 2) $ <eq:contr-norm>

$ c_(l m n) = sqrt(((2L-1)!!) / ((2l-1)!! (2m-1)!! (2n-1)!!)) $ <eq:cart-norm>

@eq:contr-norm の $tilde(N)$ は成分 $(L, 0, 0)$ を規格化し、$c_(l m n)$ は他のデカルト成分も
それぞれ規格化する（$bra(chi_(l m n)) ket(chi_(l m n)) = 1$）。$tilde(d)_k$ は殻ごとに一度だけ計算しておく。

デカルト成分の順序は辞書式とする：
p: $(x, y, z)$、d: $(x x, x y, x z, y y, y z, z z)$、
一般に $l$ を $L$ から $0$ へ、その中で $m$ を $L - l$ から $0$ へ。

*SP 殻*（Gaussian94 形式の `SP`）は同じ指数を共有する s 殻と p 殻に分けて扱う
（2列目が s の係数、3列目が p の係数）。

== 球面調和 d・f 関数（5d・7f）

d 殻は既定で実球面調和関数 5 成分に変換する（デカルト 6 成分のまま使うこともできる）。
規格化済みデカルト成分 $chi_(x x), chi_(x y), ...$ を用いて、$m = -2, ..., 2$ の順に

$ d_(-2) &= chi_(x y), quad d_(-1) = chi_(y z), quad d_(+1) = chi_(x z), \
  d_0 &= chi_(z z) - 1/2 chi_(x x) - 1/2 chi_(y y), \
  d_(+2) &= sqrt(3)/2 (chi_(x x) - chi_(y y)) $ <eq:sph-d>

すべて規格化されている（$bra(chi_(x x)) ket(chi_(y y)) = 1 slash 3$ を使う）。

f 殻も同様に実球面調和関数 7 成分に変換する（デカルト 10 成分のまま使うこともできる）。
規格化済みデカルト成分（辞書式に $x x x, x x y, x x z, x y y, x y z, x z z, y y y, y y z, y z z, z z z$）を用い、
$a = sqrt(5 slash 8)$、$b = sqrt(3 slash 8)$ として、$m = -3, ..., 3$ の順に

$ f_(-3) &= (3a) / sqrt(5) chi_(x x y) - a chi_(y y y), quad f_(-2) = chi_(x y z), \
  f_(-1) &= (4b) / sqrt(5) chi_(y z z) - b / sqrt(5) chi_(x x y) - b chi_(y y y), \
  f_0 &= chi_(z z z) - 3 / (2 sqrt(5)) (chi_(x x z) + chi_(y y z)), \
  f_(+1) &= (4b) / sqrt(5) chi_(x z z) - b chi_(x x x) - b / sqrt(5) chi_(x y y), \
  f_(+2) &= sqrt(3)/2 (chi_(x x z) - chi_(y y z)), quad
  f_(+3) = a chi_(x x x) - (3a) / sqrt(5) chi_(x y y) $ <eq:sph-f>

*導出*：実立体調和関数（正の定数倍を除く）
$y(3x^2 - y^2)$、$x y z$、$y(4z^2 - x^2 - y^2)$、$z(2z^2 - 3x^2 - 3y^2)$、$x(4z^2 - x^2 - y^2)$、
$z(x^2 - y^2)$、$x(x^2 - 3y^2)$ の各単項式 $x^l y^m z^n$ を、@eq:cart-norm により
$chi_(l m n) slash c_(l m n)$ に置き換え（$L = 3$ では $c_(x x x) = 1$、$c_(x x y) = sqrt(5)$、$c_(x y z) = sqrt(15)$ など）、
デカルト成分の一中心重なり $S$ で規格化する。同じ手順を $L = 2$ に当てはめると @eq:sph-d が再現され、
符号と順序の規約は d と同じである。$m > 0$ が $cos$ 型（$x$ を含む側）、$m < 0$ が $sin$ 型。

*数値確認*：@eq:sph-f の $7 times 10$ 行列 $T$ について $T S T^T = I_7$（誤差 $10^(-15)$ 程度）、
また $T$ の各行は $r^2$ を掛けた p 型の組 $r^2 x, r^2 y, r^2 z$ と $S$ について直交する
（球面調和 f は 10 成分から $r^2 p$ の 3 成分を除いた残りである）。
出典は Schlegel・Frisch（1995）だが、論文はまだ入手しておらず係数は*未照合*である
（上の導出と数値確認による）。

s, p 殻はデカルトのまま。積分・微分積分はデカルト成分で計算し、最後に @eq:sph-d、@eq:sph-f の
$(2L + 1) times (L + 1)(L + 2) slash 2$ 行列を各添字に掛ける。基底殻は $L <= 3$ を対象とするが、
原始積分の漸化式は任意の $L$ で動くように実装する（勾配で $L + 1 = 4$ まで必要になる）。

== 基底ファイル（Gaussian94 形式）

`data/basis/*.gbs`（Basis Set Exchange より取得）。`!` で始まる行はコメント。元素ごとに
`記号 0` の行で始まり `****` で終わる。殻は `種類 原始数 スケール` の行（種類は `S`, `P`, `D`, `F`, `SP`）に
続いて原始数ぶんの行（指数、係数…）が並ぶ。数値の指数部は `D` で書かれている（`0.18D+02`）。

= Gauss 積と Hermite 展開 <sec:hermite>

McMurchie–Davidson 法を用いる。指数 $a$（中心 $vb(A)$）と $b$（中心 $vb(B)$）の原始関数について

$ p = a + b, quad vb(P) = (a vb(A) + b vb(B)) / p, quad mu = (a b) / p, quad
  vb(X)_(A B) = vb(A) - vb(B) $ <eq:gpt>

$ vb(X)_(P A) = vb(P) - vb(A) = -b/p vb(X)_(A B), quad vb(X)_(P B) = vb(P) - vb(B) = a/p vb(X)_(A B) $

1次元の重なり分布を Hermite Gauss 関数 $Lambda_t (x) = (partial slash partial P_x)^t e^(-p x_P^2)$ で展開する：

$ x_A^i thin x_B^j thin e^(-a x_A^2) e^(-b x_B^2) = sum_(t=0)^(i+j) E_t^(i j) Lambda_t (x) $ <eq:hermite-exp>

展開係数の漸化式（$t < 0$ または $t > i + j$ では $E_t^(i j) = 0$）：

$ E_0^(00) &= exp(-mu X_(A B)^2) \
  E_t^(i+1, j) &= 1/(2p) E_(t-1)^(i j) + X_(P A) E_t^(i j) + (t+1) E_(t+1)^(i j) \
  E_t^(i, j+1) &= 1/(2p) E_(t-1)^(i j) + X_(P B) E_t^(i j) + (t+1) E_(t+1)^(i j) $ <eq:E-rec>

ここでの $Lambda_t$ は Gauss 因子 $e^(-p x_P^2)$ を含む。McMurchie と Davidson（1978）の $Lambda_t$ は多項式部分だけで、
係数 $d_t^(i j)$ は因子 $exp(-mu X_(A B)^2)$ を含まないので、$E_t^(i j) = exp(-mu X_(A B)^2) d_t^(i j)$ である（漸化式は同じ）。

$x, y, z$ それぞれについて独立に求め、$E_t^(l_1 l_2)$（$x$）、$E_u^(m_1 m_2)$（$y$）、$E_v^(n_1 n_2)$（$z$）と書く。
以下 $E_(t u v)^(a b) = E_t^(l_1 l_2) E_u^(m_1 m_2) E_v^(n_1 n_2)$。

= 1電子積分 <sec:oneel>

この節の式は *規格化していない原始関数* @eq:prim の対に対するものである。基底関数間の積分は
@eq:contracted に従って $c_(l m n) tilde(d)_k$ を掛けて和をとる。

== 重なり積分

1次元：

$ S_(i j) = E_0^(i j) sqrt(pi / p) $ <eq:ovlp-1d>

3次元： $ bra(g_a) ket(g_b) = S_(l_1 l_2) S_(m_1 m_2) S_(n_1 n_2) $ <eq:ovlp>

== 運動エネルギー積分

$hat(T) = -1/2 nabla^2$。1次元の2階微分の行列要素は

$ D_(i j)^2 = bra(i) dif^2 / (dif x^2) ket(j) = j(j-1) S_(i, j-2) - 2b(2j+1) S_(i j) + 4b^2 S_(i, j+2) $ <eq:kin-1d>

（$j < 2$ のとき第1項は 0）。
McMurchie と Davidson（1978）は運動エネルギーを $nabla phi dot nabla phi$ の形（その式 (2.37)）で書いているが、部分積分で上の2階微分の形と等しい。

$ bra(g_a) hat(T) ket(g_b) = -1/2 [D_(l_1 l_2)^2 S_(m_1 m_2) S_(n_1 n_2) + S_(l_1 l_2) D_(m_1 m_2)^2 S_(n_1 n_2)
  + S_(l_1 l_2) S_(m_1 m_2) D_(n_1 n_2)^2] $ <eq:kin>

== 双極子積分

原点 $vb(C)$ に対する1次元の1次モーメント：

$ bra(i) x - C_x ket(j) = (E_1^(i j) + X_(P C) E_0^(i j)) sqrt(pi / p), quad X_(P C) = P_x - C_x $ <eq:dip-1d>

3次元では該当方向だけ @eq:dip-1d、他の方向は @eq:ovlp-1d を用いる。双極子モーメント（原点 $vb(0)$）は

$ vb(mu) = sum_A Z_A vb(R)_A - sum_(mu nu) P_(mu nu) bra(chi_mu) vb(r) ket(chi_nu) $ <eq:dipole>

== Boys 関数

$ F_n (x) = integral_0^1 t^(2n) e^(-x t^2) dif t $ <eq:boys>

必要な最大次数を $n_"max"$ とする。

- $x < 35$：最高次を級数で求め、下降漸化式で残りを得る。
  $ F_(n_"max")(x) = e^(-x) sum_(k=0)^infinity (2x)^k / ((2 n_"max" + 1)(2 n_"max" + 3) dots.c (2 n_"max" + 2k + 1)) $ <eq:boys-series>
  項が和の $10^(-17)$ 倍より小さくなったら打ち切る。
  $ F_(n-1)(x) = (2x F_n (x) + e^(-x)) / (2n - 1) $ <eq:boys-down>
- $x >= 35$：漸近形から上昇漸化式。
  $ F_0 (x) = 1/2 sqrt(pi / x), quad F_(n+1)(x) = ((2n+1) F_n (x) - e^(-x)) / (2x) $ <eq:boys-up>

$x = 0$ では $F_n (0) = 1 slash (2n+1)$（@eq:boys-series はこれを含む）。

直接法（@sec:direct）では SCF の反復ごとに Boys 関数を評価し直すので、$x < 35$ では表を使う。
@eq:boys-series と @eq:boys-down で前もって $F_n (x_i)$（$x_i = i Delta$、$Delta = 0.05$）を作っておき、
最寄りの格子点 $x_i$（$abs(delta) <= Delta slash 2$、$delta = x - x_i$）のまわりで
$dif F_n slash dif x = -F_(n+1)$ を使って Taylor 展開する：

$ F_(n_"max")(x) = sum_(k=0)^6 F_(n_"max" + k)(x_i) (-delta)^k / k! $ <eq:boys-taylor>

打ち切り誤差は $(Delta slash 2)^7 slash 7! approx 10^(-15)$（相対）以下。残りの次数は @eq:boys-down で求める。

== Hermite Coulomb 積分

指数 $p$、変位 $vb(R) = (X, Y, Z)$ に対する補助積分 $R_(t u v)^n (p, vb(R))$：

$ R_(000)^n &= (-2p)^n F_n (p R^2) \
  R_(t+1, u, v)^n &= t thin R_(t-1, u, v)^(n+1) + X thin R_(t u v)^(n+1) \
  R_(t, u+1, v)^n &= u thin R_(t, u-1, v)^(n+1) + Y thin R_(t u v)^(n+1) \
  R_(t, u, v+1)^n &= v thin R_(t, u, v-1)^(n+1) + Z thin R_(t u v)^(n+1) $ <eq:R-rec>

最終的に使うのは $R_(t u v) equiv R_(t u v)^0$。$t + u + v <= L_"tot"$ まで必要なら Boys 関数は $n <= L_"tot"$ まで要る。

== 核引力積分

点電荷の位置を $vb(C)$ とすると

$ bra(g_a) 1 / abs(vb(r) - vb(C)) ket(g_b) = (2 pi) / p sum_(t u v) E_(t u v)^(a b) R_(t u v)(p, vb(P) - vb(C)) $ <eq:nuc-prim>

$ V_(mu nu) = - sum_C Z_C bra(chi_mu) 1 / abs(vb(r) - vb(R)_C) ket(chi_nu) $ <eq:nuc>

コアハミルトニアンは $h_(mu nu) = T_(mu nu) + V_(mu nu)$。

= 2電子反発積分 <sec:eri>

化学者の記法 $(mu nu | lambda sigma) = integral.double chi_mu (vb(r)_1) chi_nu (vb(r)_1) r_(12)^(-1) chi_lambda (vb(r)_2) chi_sigma (vb(r)_2) dif vb(r)_1 dif vb(r)_2$。
原始関数 $a, b$（電子1、$p$, $vb(P)$）と $c, d$（電子2、$q = c + d$, $vb(Q)$）について

$ (g_a g_b | g_c g_d) = (2 pi^(5 slash 2)) / (p q sqrt(p + q))
  sum_(t u v) E_(t u v)^(a b) sum_(tau nu phi) (-1)^(tau + nu + phi) E_(tau nu phi)^(c d)
  R_(t + tau, u + nu, v + phi)(alpha, vb(P) - vb(Q)), quad alpha = (p q) / (p + q) $ <eq:eri>

8重の置換対称性

$ (mu nu | lambda sigma) = (nu mu | lambda sigma) = (mu nu | sigma lambda) = (lambda sigma | mu nu) = dots.c $ <eq:eri-sym>

を使って独立な要素だけを計算・保存する。Schwarz の不等式

$ abs((mu nu | lambda sigma)) <= sqrt((mu nu | mu nu)) sqrt((lambda sigma | lambda sigma)) $ <eq:schwarz>

により、殻対ごとの上限の積が閾値（$10^(-12)$）未満の殻四つ組は省略してよい。

= 閉殻 Hartree–Fock（RHF） <sec:rhf>

== 方程式

密度行列（$C$ は MO 係数行列）

$ P_(mu nu) = 2 sum_i^"occ" C_(mu i) C_(nu i) $ <eq:density>

Coulomb 行列と交換行列

$ J_(mu nu) = sum_(lambda sigma) P_(lambda sigma) (mu nu | lambda sigma), quad
  K_(mu nu) = sum_(lambda sigma) P_(lambda sigma) (mu lambda | nu sigma) $ <eq:jk>

Fock 行列（RHF）

$ F = h + J - 1/2 K $ <eq:fock-rhf>

Roothaan 方程式と全エネルギー

$ F C = S C epsilon $ <eq:roothaan>

$ E_"RHF" = sum_(mu nu) P_(mu nu) h_(mu nu) + 1/2 sum_(mu nu) P_(mu nu) (J_(mu nu) - 1/2 K_(mu nu)) + V_"nn" $ <eq:e-rhf>

== 直交化と対角化

$S = U s U^T$（対称行列の固有値分解）から対称直交化行列

$ X = U s^(-1 slash 2) U^T $ <eq:orth>

を作り、$F' = X^T F X$ を対角化して $F' C' = C' epsilon$、$C = X C'$。
固有値は昇順に並べ、下から $N_"occ"$ 個を占有する。$s$ の最小固有値が $10^(-8)$ 未満ならエラーとする
（線形従属）。対称行列 $A$（$n times n$）の固有値分解は、Householder 法による三重対角化と
陰的シフト付き QL 法で実装する。まず $k = 0, dots, n - 3$ の順に、第 $k$ 列の対角より下の部分
$x = (a_(k+1, k), dots, a_(n-1, k))^T$ を第1成分だけに縮める鏡映

$ P_k = I - beta v v^T, quad v = x - alpha e_1, quad alpha = -"sgn"(x_1) norm(x), quad
  beta = 2 / (v^T v) $ <eq:householder>

（添字 $k+1, dots, n-1$ にだけ作用し、$"sgn"(0) = +1$。$x$ の第2成分以降がすべて 0 なら $P_k = I$）
を $A <- P_k A P_k$ と施し、三重対角行列 $T = Q^T A Q$（$Q = P_0 P_1 dots.c P_(n-3)$）を得る。
$T$ の対角を $d_i$、副対角を $e_i$（$i$ と $i+1$ の結合）とする。次に、$abs(e_m) <= epsilon_"mach" (abs(d_m) + abs(d_(m+1)))$
となる $e_m$ を 0 とみなして $T$ を分割し、分割されていない先頭ブロック $[l, m]$ に
シフト

$ delta = (d_(l+1) - d_l) / (2 e_l), quad
  mu = d_l - e_l / (delta + "sgn"(delta) sqrt(delta^2 + 1)) $ <eq:ql-shift>

（先頭の $2 times 2$ ブロックの固有値のうち $d_l$ に近いほう、$"sgn"(0) = +1$）で陰的 QL 掃引（平面回転の列）を繰り返す。
各回転を $Q$ に右から掛けていき、すべての $e$ が 0 になったときの $d$ が固有値、$Q$ の列が固有ベクトルである。
1つの固有値あたり 50 回の掃引で収束しなければエラーとする。

== SCF 手順

+ $S, T, V$、ERI（保存法の場合。直接法は @sec:direct）、$X$ を計算する。
+ 初期値：原子密度の重ね合わせ（@sec:sad）で $P$ を作る。前の構造の密度行列などが与えられていればそれを使う。
+ $P$ から $F$ を作り、エネルギーを計算する。
+ DIIS で $F$ を外挿する（下記）。
+ 外挿した $F$ を対角化して新しい $P$ を得る。
+ 収束判定：$abs(Delta E) < 10^(-10)$ かつ $max abs(e_(mu nu)) < 10^(-7)$。未収束なら 3 へ。
  最大反復回数（既定 200）を超えたらエラーを返す。

収束後、収束した $P$ から作った $F$ を対角化した $epsilon, C$ を結果とする。
密度行列とエネルギーもこの $C$ から作り直し、$E$、$P$、$epsilon$、$C$ が互いに整合するようにする
（勾配の @eq:W がこの整合性を必要とする）。

== 初期値：原子密度の重ね合わせ <sec:sad>

コアハミルトニアン $h$ の固有ベクトルから作る初期値は電子間反発を無視するので、
ヘテロ原子の多い分子では電子が偏りすぎ、DIIS が収束しないことがある
（24 原子の分子で実際に発散した）。そこで、各原子を単独で解いた密度を並べたものを初期値にする：

$ P^((0)) = plus.o.big_A P^A $ <eq:sad>

$P^A$ は原子 $A$ の基底関数だけで張ったブロックで、原子間のブロックは 0 とする。
$P^A$ は中性原子を球対称に平均した Hartree–Fock 計算で求める。原子 $A$ の基底だけを使い、
$F^A = h^A + J[P^A] - 1/2 K[P^A]$ を対角化して、軌道エネルギーの低い順に電子を入れる。
縮退した軌道の組（縮退度 $g$、軌道エネルギーの差が $10^(-6)$ 未満）には、残りの電子を等分に入れる：

$ P^A_(mu nu) = sum_i n_i C_(mu i) C_(nu i), quad
  n_i = min(N_"残り", 2g) / g quad (i in "縮退した組") $ <eq:sad-occ>

たとえば炭素は $1s^2 2s^2$ のあと、3つの $2p$ 軌道に $2 slash 3$ 個ずつ入る。こうすると密度は球対称に保たれる。
反復は $h^A$ から始め、密度を新旧 $1:1$ で混ぜながら、密度行列の変化が $10^(-6)$ 未満になるまで行う。
同じ元素の原子には同じ $P^A$ を使う。

$P^((0))$ は中性分子の電子数を持つが、べき等ではない（$P S P != 2P$）。最初の対角化で
正しい電子数のべき等な密度になる。交換相関汎関数によらず、原子の計算は Hartree–Fock で行う
（初期値なので精度は要らない）。収束した結果は初期値に依らない。

== DIIS

誤差行列（直交化基底での交換子）

$ e = X^T (F P S - S P F) X $ <eq:diis-err>

直近 $m <= 8$ 回の $(F_k, e_k)$ を保持し、$B_(k l) = sum_(mu nu) (e_k)_(mu nu) (e_l)_(mu nu)$ として

$ mat(B, -vb(1); -vb(1)^T, 0) vec(vb(c), lambda) = vec(vb(0), -1), quad F_"ext" = sum_k c_k F_k $ <eq:diis>

を解く（部分ピボット付き Gauss 消去）。保持数が 2 以上になったら外挿を使う。
連立方程式が特異なら最古の組を捨ててやり直す。

== 2電子積分の直接法 <sec:direct>

独立な ERI をすべて保存する方法（保存法）はメモリが基底数の4乗に比例する（約 $N^4 slash 8$ 個の倍精度数）。
直接法では積分を保存せず、反復ごとに殻四つ組 $(a b|c d)$ を計算して、その場で $J$ と $K$ に足し込む。
@eq:eri-sym の独立な組（$a >= b$、$c >= d$、殻対の番号で $(a b) >= (c d)$）だけを回り、
重み $w = 2^(-[a = b] - [c = d] - [(a b) = (c d)])$ をかけて、$mu in a, nu in b, lambda in c, sigma in d$ について

$ & tilde(J)_(mu nu) += 2 w (mu nu|lambda sigma) P_(lambda sigma), quad
    tilde(J)_(lambda sigma) += 2 w (mu nu|lambda sigma) P_(mu nu), \
  & tilde(K)_(mu lambda) += w (mu nu|lambda sigma) P_(nu sigma), quad
    tilde(K)_(nu lambda) += w (mu nu|lambda sigma) P_(mu sigma), \
  & tilde(K)_(mu sigma) += w (mu nu|lambda sigma) P_(nu lambda), quad
    tilde(K)_(nu sigma) += w (mu nu|lambda sigma) P_(mu lambda), \
  & J = tilde(J) + tilde(J)^T, quad K = tilde(K) + tilde(K)^T $ <eq:jk-quartet>

とする（$J, K$ は @eq:jk に一致する。8つの置換のうち互いに転置の関係にある項を最後の対称化でまとめ、
同じ置換が重なる分を $w$ で割り引く）。範囲分離混成汎関数では $K$ の式の積分を
$alpha (mu nu|lambda sigma) + beta (mu nu|lambda sigma)_omega$ に置き換えて $K^"x"$（@eq:cam-k）を作る。
同じ原始関数の組の Hermite 展開係数 $E$ を2つの演算子で共有し、$R$ だけを2回求める。
計算は規格化したデカルト成分で行い、密度を $P^"cart" = T^T P T$、結果を $J = T J^"cart" T^T$ で変換する
（$T$ は @eq:sph-d、@eq:sph-f の行列、デカルト d・f では単位行列）。

*密度で重みづけた Schwarz 選別*：四つ組が $J$ と $K$ に寄与するときに掛かる密度のブロックは
$P_(a b), P_(c d)$（$J$）と $P_(a c), P_(a d), P_(b c), P_(b d)$（$K$）なので、
殻対の Schwarz 因子 $Q_(a b) = max_(mu in a, nu in b) sqrt((mu nu|mu nu))$ を使って

$ Q_(a b) Q_(c d) max(norm(P_(a b)), norm(P_(c d)), norm(P_(a c)), norm(P_(a d)), norm(P_(b c)), norm(P_(b d))) < tau $ <eq:schwarz-dens>

なら四つ組を省略する（$norm(dot)$ はブロック内の最大絶対値。$K$ が要らないときは $J$ の2つだけ。既定 $tau = 10^(-12)$）。
@eq:schwarz により省略した項は1つあたり $tau$ 以下である。
$erf(omega r) slash r$ と $"erfc"(omega r) slash r$ はどちらも正定値の核なので $(mu nu|mu nu)_omega <= (mu nu|mu nu)$ であり、
Coulomb 積分の $Q$ は減衰した積分の上限にもなる。殻四つ組の中の原始関数の四つ組も、原始関数の積ごとの
Schwarz 因子で同じ判定をし、閾値 $tau slash 10$ 未満なら省略する。

*差分による Fock 行列の構築*：$J$ と $K$ は密度について線形なので、反復 $k$ では前回の密度との差から

$ Delta P = P_k - P_(k-1), quad J[P_k] = J[P_(k-1)] + J[Delta P], quad K[P_k] = K[P_(k-1)] + K[Delta P] $ <eq:fock-incr>

と作る。収束につれて $Delta P$ が小さくなり、@eq:schwarz-dens で $P$ を $Delta P$ に置き換えた判定で省略される四つ組が増える。
全体から作るときの省略の誤差は毎回同じ $P$ の関数だが、差分で作ると各回の誤差が反復をまたいで足し合わさり、
Fock 行列に $P$ だけでは決まらない雑音が乗る（閾値 $tau$ のままでは DIIS 誤差が $10^(-10)$ 前後で止まった）。そこで

- 差分の構築では閾値を $tau dot "clamp"(e slash (10^6 tau), 1 slash 100, 1)$ にする（$e$ は前の反復の DIIS 誤差 @eq:diis-err）。
  収束から遠いうちは雑音が問題にならないので $tau$ のまま安く作り、収束に近づくと $tau slash 100$ まで締める。
- 差分による構築を10回続けたら全体を作り直す。
- 前の反復の DIIS 誤差（@eq:diis-err）が $100 tau$ 未満になったら、以後は毎回全体から作る
  （既定の $tau$ ではふつう起こらない。大きな $tau$ を指定したときに収束を妨げないため）。
- 収束後のエネルギー（@sec:rhf の SCF 手順の最後）も差分ではなく全体から作る。

保存法と直接法は `ScfOptions::integral_strategy` で選ぶ。既定（自動）では、保存するテンソル
（範囲分離混成汎関数では2つ）の大きさが上限（既定 0.5 GiB）以下なら保存法、超えるなら直接法を使う。

= 閉殻 Kohn–Sham DFT（RKS） <sec:rks>

== エネルギーと Fock 行列

交換相関エネルギー密度を $f(rho, sigma)$（体積あたり）、厳密交換の混合率を $a_"HF"$ とすると

$ E_"RKS" = sum_(mu nu) P_(mu nu) h_(mu nu) + 1/2 sum_(mu nu) P_(mu nu) J_(mu nu)
  - a_"HF" / 4 sum_(mu nu) P_(mu nu) K_(mu nu) + E_"xc" + V_"nn" $ <eq:e-rks>

$ F = h + J - a_"HF" / 2 K + V^"xc" $ <eq:fock-rks>

RHF は $a_"HF" = 1$、$f = 0$ の特別な場合である。SCF 手順は @sec:rhf と同じ。

== 格子上の密度

$ rho(vb(r)) = sum_(mu nu) P_(mu nu) chi_mu (vb(r)) chi_nu (vb(r)), quad
  nabla rho(vb(r)) = 2 sum_(mu nu) P_(mu nu) chi_mu (vb(r)) nabla chi_nu (vb(r)), quad
  sigma = abs(nabla rho)^2 $ <eq:rho>

基底関数の値と微分は @eq:prim から解析的に求める：

$ partial / (partial x) (x_A^l e^(-a r_A^2)) = (l x_A^(l-1) - 2a x_A^(l+1)) e^(-a r_A^2) $ <eq:ao-deriv>

（2階微分はこれを繰り返し適用する。）

== 数値積分

$ E_"xc" = sum_g w_g f(rho_g, sigma_g) $ <eq:exc>

$ V_(mu nu)^"xc" = sum_g w_g [v_rho chi_mu chi_nu
  + 2 v_sigma nabla rho dot (chi_mu nabla chi_nu + chi_nu nabla chi_mu)]_g, quad
  v_rho = (partial f) / (partial rho), quad v_sigma = (partial f) / (partial sigma) $ <eq:vxc>

$rho_g < 10^(-10)$ の点は寄与を 0 とする。$sigma$ は $10^(-40)$ 以上に切り上げる（$sqrt(sigma)$ の微分のため）。
電子数 $sum_g w_g rho_g approx N_"elec"$ を格子精度の診断値として返す。

== 原子中心格子

分子の格子は原子ごとの球面格子の和で、点 $g$ の重みは
$w_g = w_"rad" w_"ang" w_"Becke"$。

*動径格子*（Gauss–Chebyshev 第2種 + Becke 写像）。$n_r$ 点、$i = 1, ..., n_r$：

$ x_i = cos((i pi) / (n_r + 1)), quad r_i = R_m (1 + x_i) / (1 - x_i), quad
  w_i^"rad" = pi / (n_r + 1) sin((i pi) / (n_r + 1)) (2 R_m) / (1 - x_i)^2 r_i^2 $ <eq:radial>

$R_m$ は Bragg–Slater 半径 $R_"BS"$ の半分（水素だけは $R_"BS"$ そのもの）。

*角度格子*（直積格子）。$cos theta_j = u_j$ を $n_theta$ 点 Gauss–Legendre 求積の節点（重み $omega_j$）、
$phi_k = 2 pi (k + 1 slash 2) slash n_phi$（$k = 0, ..., n_phi - 1$、$n_phi = 2 n_theta$）とする：

$ hat(vb(n))_(j k) = (sqrt(1 - u_j^2) cos phi_k, sqrt(1 - u_j^2) sin phi_k, u_j), quad
  w_(j k)^"ang" = omega_j (2 pi) / n_phi $ <eq:angular>

重みの総和は $4 pi$。次数 $2 n_theta - 1$ までの球面調和関数を厳密に積分する。
Gauss–Legendre の節点は Legendre 多項式 $P_n$ の零点で、初期値
$u_j^((0)) = cos(pi (j - 1 slash 4) slash (n + 1 slash 2))$（$j = 1, ..., n$）から Newton 法で求める：

$ (k+1) P_(k+1)(u) = (2k+1) u P_k (u) - k P_(k-1)(u), quad
  P'_n (u) = n (u P_n (u) - P_(n-1)(u)) / (u^2 - 1), quad
  omega_j = 2 / ((1 - u_j^2) P'_n (u_j)^2) $ <eq:gauss-legendre>

格子点は $vb(r) = vb(R)_A + r_i hat(vb(n))_(j k)$。

*Becke 分割*。原子 $A$ の格子点 $vb(r)$ に対し、$r_A = abs(vb(r) - vb(R)_A)$ として

$ mu_(A B) = (r_A - r_B) / R_(A B), quad nu_(A B) = mu_(A B) + a_(A B)(1 - mu_(A B)^2) $ <eq:becke-mu>

$ a_(A B) = u_(A B) / (u_(A B)^2 - 1), quad u_(A B) = (chi_(A B) - 1) / (chi_(A B) + 1), quad
  chi_(A B) = R_"BS"^A / R_"BS"^B, quad abs(a_(A B)) <= 1/2 "に制限" $ <eq:becke-a>

$ s(nu) = 1/2 [1 - p(p(p(nu)))], quad p(t) = 3/2 t - 1/2 t^3 $ <eq:becke-s>

$ P_A (vb(r)) = product_(B != A) s(nu_(A B)), quad w_"Becke" = P_A (vb(r)) / (sum_C P_C (vb(r))) $ <eq:becke-w>

（1原子分子では $w_"Becke" = 1$。）

*既定の格子*：$n_r = 75$、$n_theta = 18$（$n_phi = 36$）。精密な比較には $n_r = 150$、$n_theta = 30$ を使う。

*Bragg–Slater 半径*（Å、@eq:bohr で Bohr に換算。希ガスは便宜上の値）：

#align(center, table(
  columns: 9, stroke: 0.4pt, inset: 4pt, align: center,
  [H], [He], [Li], [Be], [B], [C], [N], [O], [F],
  [0.35], [0.35], [1.45], [1.05], [0.85], [0.70], [0.65], [0.60], [0.50],
  [Ne], [Na], [Mg], [Al], [Si], [P], [S], [Cl], [Ar],
  [0.45], [1.80], [1.50], [1.25], [1.10], [1.00], [1.00], [1.00], [0.90],
))

= 交換相関汎関数 <sec:xc>

閉殻（$rho_alpha = rho_beta = rho slash 2$）での体積あたりエネルギー密度 $f(rho, sigma)$ を与える。
微分 $v_rho, v_sigma$ は *前進型自動微分* で求める：値と2つの偏微分を持つ双対数

$ hat(x) = (x, x_rho, x_sigma), quad
  hat(x) hat(y) = (x y, x_rho y + x y_rho, x_sigma y + x y_sigma), quad
  g(hat(x)) = (g(x), g'(x) x_rho, g'(x) x_sigma) $ <eq:dual>

を用意し、$hat(rho) = (rho, 1, 0)$、$hat(sigma) = (sigma, 0, 1)$ から下の式をそのまま評価する。
必要な初等関数は四則、冪、$sqrt(dot)$、$exp$、$ln$、$arctan$、$asinh$
（$asinh'(x) = 1 slash sqrt(1 + x^2)$、$arctan'(x) = 1 slash (1 + x^2)$）。

== Slater 交換（LDA）

$ f_x^"S" = -3/4 (3 / pi)^(1 slash 3) rho^(4 slash 3) $ <eq:slater>

係数は Dirac（1930）の一様電子気体の交換エネルギーのもので、Kohn と Sham（1965）の交換ポテンシャル
$mu_x = -(3 rho slash pi)^(1 slash 3)$（その式 (2.19)）は $partial f_x^"S" slash partial rho$ に一致する。
Slater（1951）の平均交換ポテンシャル $-3 (3 rho slash 8 pi)^(1 slash 3)$ はこの $3 slash 2$ 倍で、ここで使う係数の出典ではない（「Slater 交換」は慣用名）。

== Becke 88 交換

スピンごとの密度 $rho_s = rho slash 2$、勾配 $abs(nabla rho_s) = sqrt(sigma) slash 2$ を用いて

$ x = (sqrt(sigma) slash 2) / (rho slash 2)^(4 slash 3), quad
  f_x^"B88" = f_x^"S" - 2 beta (rho / 2)^(4 slash 3) x^2 / (1 + 6 beta x asinh(x)), quad beta = 0.0042 $ <eq:b88>

== VWN 相関（LDA）

$ r_s = (3 / (4 pi rho))^(1 slash 3), quad x = sqrt(r_s), quad X(x) = x^2 + b x + c, quad Q = sqrt(4c - b^2) $

$ epsilon_c^"VWN" = A {ln(x^2 / X(x)) + (2b) / Q arctan(Q / (2x + b))
  - (b x_0) / X(x_0) [ln((x - x_0)^2 / X(x)) + (2(b + 2 x_0)) / Q arctan(Q / (2x + b))]} $ <eq:vwn>

$ f_c^"VWN" = rho thin epsilon_c^"VWN" $

常磁性（閉殻）のパラメータ。$A$ は Hartree 単位で、論文の値 $A^P = 0.0621814$ Ry の半分である。
VWN5 は Ceperley–Alder の値への当てはめ、VWN-RPA は RPA の値への当てはめで、
どちらの名前も論文のものではなく後の慣用である：

#align(center, table(
  columns: 5, stroke: 0.4pt, inset: 5pt, align: center,
  [], [$A$], [$x_0$], [$b$], [$c$],
  [VWN5], [0.0310907], [$-0.10498$], [3.72744], [12.9352],
  [VWN-RPA], [0.0310907], [$-0.409286$], [13.0720], [42.7198],
))

== Lee–Yang–Parr 相関（閉殻形）

$ omega = exp(-c rho^(-1 slash 3)) / (1 + d rho^(-1 slash 3)) rho^(-11 slash 3), quad
  delta = c rho^(-1 slash 3) + (d rho^(-1 slash 3)) / (1 + d rho^(-1 slash 3)), quad
  C_F = 3/10 (3 pi^2)^(2 slash 3) $

$ f_c^"LYP" = -(a rho) / (1 + d rho^(-1 slash 3))
  - a b omega rho^2 [C_F rho^(8 slash 3) - sigma (3 + 7 delta) / 72] $ <eq:lyp>

$ a = 0.04918, quad b = 0.132, quad c = 0.2533, quad d = 0.349 $

== 汎関数の定義

#align(center, table(
  columns: 3, stroke: 0.4pt, inset: 5pt, align: (left, left, center),
  [名前], [$f$], [$a_"HF"$],
  [HF], [$0$], [1],
  [Slater], [$f_x^"S"$], [0],
  [SVWN5], [$f_x^"S" + f_c^"VWN5"$], [0],
  [SVWN-RPA], [$f_x^"S" + f_c^"VWN-RPA"$], [0],
  [B88], [$f_x^"B88"$], [0],
  [BLYP], [$f_x^"B88" + f_c^"LYP"$], [0],
  [B3LYP], [$0.08 f_x^"S" + 0.72 f_x^"B88" + 0.19 f_c^"VWN-RPA" + 0.81 f_c^"LYP"$], [0.20],
  [B3LYP5], [$0.08 f_x^"S" + 0.72 f_x^"B88" + 0.19 f_c^"VWN5" + 0.81 f_c^"LYP"$], [0.20],
))

B3LYP は Gaussian と同じ VWN-RPA 版（Stephens ら 1994）、B3LYP5 は VWN5 版である。

元の Becke（1993）の3パラメータ混成は（その式 (2)、(3)）

$ E_"xc" = E_"xc"^"LSDA" + a_0 (E_x^"exact" - E_x^"LSDA") + a_x Delta E_x^"B88" + a_c Delta E_c^"PW91", quad
  a_0 = 0.20, quad a_x = 0.72, quad a_c = 0.81 $

で、$Delta E_x^"B88"$ は B88 の勾配補正だけである。交換は $0.80 f_x^"S" + 0.72 Delta f_x^"B88"$ と厳密交換 $0.20$ で、
$f_x^"B88" = f_x^"S" + Delta f_x^"B88"$ を使うと $0.80 - 0.72 = 0.08$ から上の表の $0.08 f_x^"S" + 0.72 f_x^"B88"$ になる。
元の論文の相関は Perdew–Wang の局所相関に PW91 の勾配補正の $0.81$ 倍を加えたもので、LYP と VWN は出てこない。
相関を $(1 - a_c) f_c^"VWN" + a_c f_c^"LYP"$ に置き換えたのは Stephens ら（1994、未照合）である。
実装は「$f$ の線形結合（係数と成分の列）＋ $a_"HF"$」という一般形にしておく。

= 距離分離混成汎関数（CAM-B3LYP） <sec:cam>

== 演算子の分割

電子間の Coulomb 演算子を、誤差関数で短距離部分と長距離部分に分ける（Yanai, Tew, Handy 2004）：

$ 1 / r_(12) = (1 - [alpha + beta erf(omega r_(12))]) / r_(12) + (alpha + beta erf(omega r_(12))) / r_(12) $ <eq:cam-split>

第2項を厳密交換（Hartree–Fock 型）で、第1項を密度汎関数の交換で扱う。厳密交換の割合は
近距離で $alpha$、遠距離で $alpha + beta$ になる。$beta = 0$ なら通常の混成汎関数（$a_"HF" = alpha$）に戻る。
Coulomb 項 $J$ は分割しない。

== 厳密交換

交換行列を、減衰した積分を含む形に一般化する：

$ K_(mu nu)^"x" = sum_(lambda sigma) P_(lambda sigma)
  [alpha (mu lambda | nu sigma) + beta (mu lambda | nu sigma)_omega], quad
  (mu nu | lambda sigma)_omega = integral.double chi_mu chi_nu erf(omega r_(12)) / r_(12) chi_lambda chi_sigma $ <eq:cam-k>

@eq:fock-rks と @eq:e-rks の $a_"HF" K$ を $K^"x"$ に置き換える：

$ F = h + J - 1/2 K^"x" + V^"xc", quad
  E = sum P h + 1/2 sum P J - 1/4 sum_(mu nu) P_(mu nu) K_(mu nu)^"x" + E_"xc" + V_"nn" $ <eq:cam-fock>

実装では核 $[alpha + beta erf(omega r)] slash r$ の積分を1つのテンソルとして持てばよい
（$beta = 0$ のときは通常の積分の $alpha$ 倍で、追加のテンソルは要らない）。

== 減衰した2電子積分

$erf(omega r) slash r$ の積分は、@eq:eri で Boys 関数の引数と前因子を変えるだけで得られる。
$alpha_"PQ" = p q slash (p + q)$ として

$ alpha' = (alpha_"PQ" omega^2) / (alpha_"PQ" + omega^2), quad
  (g_a g_b | g_c g_d)_omega = (2 pi^(5 slash 2)) / (p q sqrt(p + q)) sqrt(omega^2 / (alpha_"PQ" + omega^2))
  sum_(t u v) E_(t u v)^(a b) sum_(tau nu phi) (-1)^(tau + nu + phi) E_(tau nu phi)^(c d)
  R_(t + tau, u + nu, v + phi)(alpha', vb(P) - vb(Q)) $ <eq:eri-erf>

（Hermite 展開係数 $E$ は変わらない。$omega -> infinity$ で @eq:eri に戻る。）
これは $erf(omega r) slash r = (2 slash sqrt(pi)) integral_0^omega e^(-t^2 r^2) dif t$ から従う。
8重対称性 @eq:eri-sym と Schwarz の不等式 @eq:schwarz は減衰した積分にもそのまま成り立つ
（$erf(omega r) slash r$ は正定値の核である）。中心微分 @eq:center-deriv も同じ規則で求まる。

Adamson, Dombroski, Gill（1999）が式 (11)–(13) で与えているのは短距離の $"erfc"(omega r) slash r$ の積分で、
上の式はその式 (1)（$erf = 1 slash r - "erfc"$）により、式 (11) の引かれる第2項として導いたものである。
次の3点はこの論文にはなく、ここで確かめたことである：

- 積分表示：$erf(omega r) slash r$ を $omega$ で微分すると $(2 slash sqrt(pi)) e^(-omega^2 r^2)$ で、$erf(0) = 0$ なので上の積分表示が成り立つ（初等的）。
- 正定値性：$erf(omega r) slash r$ の Fourier 変換は $(4 pi slash k^2) exp(-k^2 slash (4 omega^2)) > 0$ なので核は正定値で、正定値の核についての Schwarz の不等式（Häser と Ahlrichs 1989）が使える。
- 中心微分：核は $vb(r)_1 - vb(r)_2$ だけの関数で、中心は Hermite 展開係数と $vb(P) - vb(Q)$ を通してしか入らないので、@eq:center-deriv がそのまま使える。

== CAM で減衰した B88 交換

交換汎関数を $E_x = -1/2 sum_s integral rho_s^(4 slash 3) K_s dif vb(r)$ と書いたときの $K_s$ を使い、
Iikura ら（2001、式 (8)）の短距離 GGA 交換を、@eq:cam-split の第1項に一般化する。
$alpha = 0$、$beta = 1$ のとき Iikura ら・Tawada ら（2004）の長距離補正の交換に一致する。
$a_s, b_s, c_s$ を使う @eq:cam-att の形は Tawada ら（式 (4)–(7)）と Yanai ら（式 (2)–(5)）のもので、Iikura らの式 (8) と代数的に同じである。
一般の $alpha, beta$ での因子 $1 - alpha - beta A$ はどの論文にも印刷されておらず、構成が演算子について線形であること
（$1 - alpha - beta erf = (1 - alpha - beta) dot 1 + beta op("erfc")$）から従う。
$K_s$ の B88 での具体形は Becke（1988）による。B88 では

$ K_s = 2 [3/2 (3 / (4 pi))^(1 slash 3) + beta_"B88" x_s^2 / (1 + 6 beta_"B88" x_s asinh(x_s))], quad
  x_s = abs(nabla rho_s) / rho_s^(4 slash 3), quad beta_"B88" = 0.0042 $ <eq:cam-K>

$ a_s = (omega sqrt(K_s)) / (6 sqrt(pi) rho_s^(1 slash 3)), quad
  b_s = exp(-1 / (4 a_s^2)) - 1, quad c_s = 2 a_s^2 b_s + 1/2 $ <eq:cam-a>

$ A(a) = 8/3 a [sqrt(pi) erf(1 / (2a)) + 2a (b - c)] $ <eq:cam-att>

$ f_x^"CAM-B88" = -sum_s 1/2 rho_s^(4 slash 3) K_s [1 - alpha - beta A(a_s)]
  = -(rho / 2)^(4 slash 3) K [1 - alpha - beta A(a)] quad ("閉殻、" rho_s = rho slash 2, abs(nabla rho_s) = sqrt(sigma) slash 2) $ <eq:cam-b88>

$A(a)$ は交換のうち長距離側に回る割合で、$a -> 0$（高密度）で $0$、$a -> infinity$（低密度）で $1$ に近づく。
$alpha = beta = 0$ なら $f_x^"CAM-B88" = f_x^"B88"$ である。

*数値上の注意。* $a$ が大きいと @eq:cam-att は桁落ちする（$1 - A$ の相対誤差は $a = 1$ で $10^(-14)$、
$a = 6$ で $10^(-11)$ 程度）。そこで $a >= 0.5$ では、$u = 1 slash (4 a^2)$ についての級数

$ 1 - A(a) = sum_(n=1)^infinity (-1)^(n+1) (2 u^n) / (n! thin (2n+1)(n+1)(n+2))
  = 1 / (36 a^2) - 1 / (960 a^4) + 1 / (26880 a^6) - 1 / (829440 a^8) + dots.c $ <eq:cam-series>

を項が丸め誤差以下になるまで足す（すべての $a$ で収束し、$a = 0.5$ で約 17 項）。
級数は 60 桁の多倍長計算で @eq:cam-att と照合済みで、$a in [0.25, 1]$ では両者が $3 times 10^(-14)$ 以内で一致する。
$a < 0.5$ では @eq:cam-att を直接使い、$b$ は $e^x - 1$ を桁落ちなく求める関数（`exp_m1`）で計算する。
$1 slash (4a^2) > 700$ となるところでは $b = -1$、$erf = 1$ としてよい。

$erf$ は Boys 関数から求める： $erf(z) = (2z slash sqrt(pi)) F_0 (z^2)$。
双対数 @eq:dual には $erf'(z) = (2 slash sqrt(pi)) e^(-z^2)$ を加える。

== CAM-B3LYP の定義

$ f^"CAM-B3LYP" = f_x^"CAM-B88" + 0.81 f_c^"LYP" + 0.19 f_c^"VWN5", quad
  alpha = 0.19, quad beta = 0.46, quad omega = 0.33 thin a_0^(-1) $ <eq:cam-b3lyp>

論文（Yanai ら 2004）の記号では $omega$ は $mu$ である。論文は $mu$ の単位を書いておらず、ここでは原子単位と解釈している。
交換は B88 全体（Slater と B88 補正に分けた B3LYP 型の混合ではない）である。

（相関の局所部分は VWN5。）実装の一般形は「$f$ の線形結合 ＋ 厳密交換の核 $(alpha, beta, omega)$」とし、
これまでの $a_"HF"$ は $alpha = a_"HF"$、$beta = 0$ に当たる。

== 勾配

@eq:grad と @eq:gamma の交換部分を、減衰した積分の微分に置き換える：

$ 1/2 sum_(mu nu lambda sigma) P_(mu nu) P_(lambda sigma) (mu nu | lambda sigma)^((X_A))
  - 1/4 sum_(mu nu lambda sigma) P_(mu lambda) P_(nu sigma)
  [alpha (mu nu | lambda sigma)^((X_A)) + beta (mu nu | lambda sigma)_omega^((X_A))] $ <eq:cam-grad>

交換相関項 @eq:xc-grad は $f$ が変わるだけで形は同じ。

= 解析的エネルギー勾配 <sec:grad>

== 全体の式

核座標 $X_A$（原子 $A$ の $x, y, z$ のいずれか）による微分。上付き $(X_A)$ は
*基底関数の中心と演算子の核位置だけ* を微分した積分（密度行列は固定）を表す：

$ (dif E) / (dif X_A) = sum_(mu nu) P_(mu nu) h_(mu nu)^((X_A))
  + 1/2 sum_(mu nu lambda sigma) [P_(mu nu) P_(lambda sigma) - a_"HF" / 2 P_(mu lambda) P_(nu sigma)]
  (mu nu | lambda sigma)^((X_A))
  - sum_(mu nu) W_(mu nu) S_(mu nu)^((X_A)) + E_"xc"^((X_A)) + (partial V_"nn") / (partial X_A) $ <eq:grad>

エネルギー重み付き密度行列：

$ W_(mu nu) = 2 sum_i^"occ" epsilon_i C_(mu i) C_(nu i) $ <eq:W>

勾配は密度行列の誤差の1次で変わる（エネルギーは2次）。そのため勾配を計算する SCF は
1点計算の既定値より厳しく、$max abs(e_(mu nu)) < 10^(-10)$ まで収束させる。

== 原始関数の中心微分

$ partial / (partial A_x) g_(l m n)(vb(r); a, vb(A)) = 2a thin g_(l+1, m, n) - l thin g_(l-1, m, n) $ <eq:center-deriv>

（$l = 0$ のとき第2項はない。$y, z$ も同様。）規格化因子 $c_(l m n) tilde(d)_k$ は中心に依らないので、
微分積分は角運動量を $plus.minus 1$ した原始関数の積分の線形結合で得られる。
たとえば原始関数 $a$ の中心に関する重なり積分の微分は

$ partial / (partial A_x) bra(g_(l_1 m_1 n_1)^a) ket(g_b) = 2a bra(g_(l_1 + 1, m_1 n_1)^a) ket(g_b) - l_1 bra(g_(l_1 - 1, m_1 n_1)^a) ket(g_b) $

運動エネルギー・核引力・ERI も同じ規則に従う。球面調和への変換 @eq:sph-d、@eq:sph-f は微分と可換。

== 1電子項

原子 $A$ に属する基底関数の集合を $mu in A$ と書く。重なりと運動エネルギー：

$ S_(mu nu)^((X_A)) = [mu in A] bra((partial chi_mu) / (partial X_A)) ket(chi_nu)
  + [nu in A] bra(chi_mu) ket((partial chi_nu) / (partial X_A)) $ <eq:dS>

（$T$ も同じ形。$[dot]$ は条件が真なら 1。）核引力は、核 $C$ ごとの積分
$I_(mu nu)^C = bra(chi_mu) abs(vb(r) - vb(R)_C)^(-1) ket(chi_nu)$ が3つの中心（$mu$ の中心、$nu$ の中心、$vb(R)_C$）に依存する。
並進不変性より

$ (partial I_(mu nu)^C) / (partial vb(R)_C) = -[bra(nabla_A chi_mu) abs(vb(r) - vb(R)_C)^(-1) ket(chi_nu)
  + bra(chi_mu) abs(vb(r) - vb(R)_C)^(-1) ket(nabla_B chi_nu)] $ <eq:hf-force>

ここで $nabla_A, nabla_B$ は基底関数の中心に関する微分。したがって核 $C$・基底対 $(mu, nu)$ ごとに
中心微分 $vb(d)_mu = bra(nabla_A chi_mu) dots.c ket(chi_nu)$、$vb(d)_nu = bra(chi_mu) dots.c ket(nabla_B chi_nu)$ を計算し、
$-Z_C P_(mu nu) vb(d)_mu$ を $mu$ の原子へ、$-Z_C P_(mu nu) vb(d)_nu$ を $nu$ の原子へ、
$+Z_C P_(mu nu) (vb(d)_mu + vb(d)_nu)$ を原子 $C$ へ加える。

== 2電子項

4つの基底関数の中心 $A, B, C, D$ についてそれぞれ中心微分
$(mu^((X)) nu | lambda sigma)$ などを @eq:center-deriv と @eq:eri で計算する。
並進不変性から4つの中心微分の和は 0 なので、3つ計算して残りは符号を変えた和で得てよい。
各微分を密度の因子

$ Gamma_(mu nu lambda sigma) = 1/2 [P_(mu nu) P_(lambda sigma) - a_"HF" / 2 P_(mu lambda) P_(nu sigma)] $ <eq:gamma>

と縮約し、微分した関数の属する原子に加える。@eq:eri-sym の対称性を使って独立な殻四つ組だけを回るときは、
$Gamma$ を対称化（$P_(mu lambda) P_(nu sigma)$ と $P_(mu sigma) P_(nu lambda)$ の平均）し、重複度を掛ける。
微分積分は保存せず、その場で縮約する。

== 交換相関項

格子の重みの核座標依存性は無視する（格子が十分密なら誤差は小さい）。
基底関数は中心とともに動くので $partial chi_mu slash partial X_A = -partial_x chi_mu$（$mu in A$）であり、

$ E_"xc"^((X_A)) = -2 sum_(mu in A) sum_nu P_(mu nu) sum_g w_g
  [v_rho (partial_x chi_mu) chi_nu
  + 2 v_sigma ((nabla rho dot nabla partial_x chi_mu) chi_nu + (partial_x chi_mu)(nabla rho dot nabla chi_nu))]_g $ <eq:xc-grad>

$partial_x$ は電子座標での微分。基底関数の2階微分 $nabla partial_x chi_mu$ が必要になる。

== 検証用の恒等式

- 並進不変性： $sum_A dif E slash dif vb(R)_A = vb(0)$（RHF では $10^(-10)$ 以下、DFT では格子誤差の範囲）
- 中心差分との一致： $(E(X_A + h) - E(X_A - h)) slash (2h)$、$h = 10^(-3) a_0$

= 構造最適化 <sec:opt>

デカルト座標 $vb(x) in RR^(3 N_"atom")$、勾配 $vb(g) = dif E slash dif vb(x)$、近似 Hessian $H$ による準 Newton 法。

+ $H_0 = 0.5 thin I$（$E_h slash a_0^2$）、信頼半径 $Delta = 0.3 thin a_0$。
+ ステップ $vb(s) = -H^(-1) vb(g)$。$abs(vb(s)) > Delta$ なら $vb(s) <- vb(s) Delta slash abs(vb(s))$。
+ $vb(x)_"new" = vb(x) + vb(s)$ でエネルギーと勾配を計算する。
+ エネルギーが上がった場合（$E_"new" > E + 10^(-8)$）：ステップを棄却し、
  $Delta <- min(Delta, abs(vb(s))) slash 2$ として同じ $vb(g), H$ で 2 に戻る
  （$Delta < 10^(-5)$ になったらエラー）。棄却したステップより必ず短くするために $abs(vb(s))$ との最小値をとる。
+ 受理した場合：$vb(y) = vb(g)_"new" - vb(g)$ として BFGS 更新
  $ H <- H + (vb(y) vb(y)^T) / (vb(y)^T vb(s)) - (H vb(s) vb(s)^T H) / (vb(s)^T H vb(s)) $ <eq:bfgs>
  ただし $vb(y)^T vb(s) <= 10^(-10)$ のときは更新しない。ステップが信頼半径で切られずに受理されたら
  $Delta <- min(2 Delta, 0.5)$。
+ 収束判定（すべて満たすこと。括弧内は既定値）：
  $max abs(g_i) < 4.5 times 10^(-4)$、$"rms"(vb(g)) < 3 times 10^(-4)$、
  $max abs(s_i) < 1.8 times 10^(-3)$、$"rms"(vb(s)) < 1.2 times 10^(-3)$。
  閾値は利用者が変更できるようにする。ステップの判定には受理したステップが必要なので、
  出発点が極小でも少なくとも1ステップは進む。

SCF の初期値には前の構造の密度行列を使ってよい。$H^(-1) vb(g)$ は連立方程式を解いて求める。

= 分散補正 DFT-D3 <sec:d3>

Grimme ら（2010）の DFT-D3 の2体項を、Becke–Johnson 減衰（D3(BJ)、Grimme, Ehrlich, Goerigk 2011）と
ゼロ減衰（D3(0)、Grimme ら 2010）の2通りで扱う。補正は核の位置と元素だけで決まり、密度に依らないので、
SCF には入らず全エネルギーと勾配に足すだけである：

$ E_"tot" = E_"SCF" + E_"disp", quad (dif E_"tot") / (dif vb(R)_A) = (dif E_"SCF") / (dif vb(R)_A) + (dif E_"disp") / (dif vb(R)_A) $

対象外とするもの：3体項（Axilrod–Teller–Muto）、D2・D4、改良版の D3M、周期系。
距離の打ち切りはせず、すべての原子対を足す（$N_"atom"^2$ 程度の計算で、分子では問題にならない）。

*出典の状況。* Grimme 研究室が公開している DFT-D3 プログラムの説明書（Ver. 3.2 Rev 0、2016）で確かめたのは、
@eq:d3-e の形、@eq:d3-bj と @eq:d3-zero の減衰関数、$R_0^(A B) = sqrt(C_8^(A B) slash C_6^(A B))$（BJ のみ）、
ゼロ減衰の $s_(r,8) = 1$、$alpha_6 = 14$、$alpha_8 = alpha_6 + 2$、それに下のパラメータ表（研究室の公開ページ）である。
配位数 @eq:d3-cn、$C_6$ の補間 @eq:d3-c6、$C_8$ @eq:d3-c8 は原論文（Grimme ら 2010）の式だが、論文はまだ入手しておらず*未照合*である
（`docs/papers/README.md`）。照合までは式・定数を確定扱いにしない。
説明書によるとプログラムの既定値には打ち切り（分散 95 $a_0$、配位数 40 $a_0$）があり、これより大きい分子では参照値との比較に差が出うる。

== エネルギー

原子対 $A < B$、距離 $r = r_(A B) = abs(vb(R)_A - vb(R)_B)$ について

$ E_"disp" = - sum_(A < B) sum_(n = 6, 8) s_n C_n^(A B) / r_(A B)^n f_n (r_(A B)) $ <eq:d3-e>

（説明書の $-1/2 sum_(A != B)$ と同じ。）$C_n^(A B)$ は原子対の（等方的な）分散係数（$E_h a_0^n$）、
$s_n$ は汎関数ごとの係数、$f_n$ は減衰関数。BJ 減衰は $C_n^(A B) slash r^n dot f_n$ をまとめて書く方が見やすいので、
@eq:d3-bj ではその形で与える。

== 配位数

$ "CN"_A = sum_(B != A) 1 / (1 + exp[-k_1 (k_2 (R_A^"cov" + R_B^"cov") / r_(A B) - 1)]), quad
  k_1 = 16, quad k_2 = 4 slash 3 $ <eq:d3-cn>

$R^"cov"$ は Pyykkö と Atsumi（2009）の単結合の共有結合半径で、金属の値は 10 % 小さくしてある（データの出典の説明による）。
取り込んだ数表（`data/dispersion/d3_h_ar.dat` の `rcov`）は $k_2$ を*掛けた後*の値 $k_2 R^"cov"$（$a_0$）である
（`data/dispersion/README.md`：H と C で $0.32$ Å、$0.75$ Å の $4 slash 3$ 倍と一致し、これ以上スケールせずに参照プログラムのエネルギーを再現する）。
したがって実装では数表の値をそのまま $k_2 R^"cov"$ として使い、$k_2$ を重ねて掛けない。
*未照合：* この半径とスケールが原論文と同じかは、論文と照合していない。

== $C_6$ の補間

元素ごとに参照の配位数 $"CN"_A^i$（$i = 1 ... N_A$）があり、参照の組ごとに係数 $C_(6, "ref")^(A B)("CN"_A^i, "CN"_B^j)$ が与えられている（Grimme ら 2010 の補足資料）。
実際の配位数での値は Gauss 重みによる平均である：

$ C_6^(A B) = (sum_(i j) C_(6, "ref")^(A B, i j) L_(i j)) / (sum_(i j) L_(i j)), quad
  L_(i j) = exp(-k_3 d_(i j)), quad
  d_(i j) = ("CN"_A - "CN"_A^i)^2 + ("CN"_B - "CN"_B^j)^2, quad k_3 = 4 $ <eq:d3-c6>

*アンダーフロー。* 配位数が参照から遠いと（$k_3 d_(i j) > 745$ 程度）すべての $L_(i j)$ が倍精度で 0 になり、0/0 になる。
比は $L$ を共通の正の因子で割っても変わらないので、$d_min = min_(i j) d_(i j)$ として
$L_(i j) = exp(-k_3 (d_(i j) - d_min))$ で計算する（最大の重みが 1 になり、分母は 1 以上）。
これは @eq:d3-c6 と数学的に同じで、規則を足すものではない。微分（@eq:d3-grad）も同じ置き換えでよい。
ただし参照プログラムがこの領域で同じ値を返すかは確かめていない（通常の分子では起きない）。

== $C_8$

$ C_8^(A B) = 3 C_6^(A B) sqrt(Q_A Q_B) $ <eq:d3-c8>

$Q_A$ は元素ごとの定数で、原子の $chevron.l r^4 chevron.r slash chevron.l r^2 chevron.r$ と $sqrt(Z_A)$ から作られる。
取り込んだ数表の `r4r2`（$a_0$）は $sqrt(Q_A)$ そのものであり、$C_8^(A B) = 3 C_6^(A B) dot "r4r2"_A dot "r4r2"_B$ と使うと
参照プログラムのエネルギーを再現する（`data/dispersion/README.md`）。He、Ne、Ar の値は後に計算し直された版（2018）である。
*未確認：* $Q_A$ の定義そのもの（$sqrt(Z_A) chevron.l r^4 chevron.r slash chevron.l r^2 chevron.r$ に掛かる係数の有無と値）は論文と照合していないが、
数表の値をそのまま使う限り実装には影響しない。$C_8$ は $C_6$ に比例するので、$Q$ は配位数に依らない。

== Becke–Johnson 減衰

$ E_"disp"^"BJ" = - sum_(A < B) sum_(n = 6, 8) s_n C_n^(A B) / (r_(A B)^n + (a_1 R_0^(A B) + a_2)^n), quad
  R_0^(A B) = sqrt(C_8^(A B) / C_6^(A B)) $ <eq:d3-bj>

（Johnson と Becke 2005 の有理減衰を Grimme ら 2011 が D3 に使ったもの。）@eq:d3-c8 より
$R_0^(A B) = (9 Q_A Q_B)^(1 slash 4)$ で、配位数に依らない。$a_2$ は長さで、$a_0$ として扱う
（*未確認：* 公開ページの表は単位を書いていない。Grimme ら 2011 の表の単位と照合する）。

== ゼロ減衰

$ f_n (r_(A B)) = 1 / (1 + 6 (r_(A B) slash (s_(r, n) R_0^(A B)))^(-alpha_n)), quad
  alpha_6 = 14, quad alpha_8 = 16, quad s_(r, 8) = 1 $ <eq:d3-zero>

ここでの $R_0^(A B)$ は BJ の $sqrt(C_8 slash C_6)$ *ではなく*、Grimme ら（2010）が原子対ごとに計算した
打ち切り半径の数表である（説明書：「the $R_0^(A B)$ used with this damping are from Ref. 1」）。
数表（94 元素の対）は補足資料にあり、取り込むときにライセンスを確かめる。

== パラメータ

$s_6 = 1$（GGA・混成汎関数）。値は Grimme 研究室の公開ページ（BJ・ゼロ減衰の表）による：

#align(center, table(
  columns: 7, stroke: 0.4pt, inset: 5pt, align: center,
  [汎関数], [D3(BJ) $s_6$], [$a_1$], [$s_8$], [$a_2$ ($a_0$)], [D3(0) $s_(r,6)$], [$s_8$],
  [B3LYP], [1.000], [0.3981], [1.9889], [4.4211], [1.261], [1.703],
  [CAM-B3LYP], [1.000], [0.3708], [2.0674], [5.4743], [1.378], [1.217],
))

*未確認：* これらが当てはめられた B3LYP が VWN-RPA 版・VWN5 版のどちらか（@sec:xc の B3LYP は VWN-RPA）。
公開ページと説明書には書かれていない。差は小さいと見込まれるが、論文で確かめるまでは
B3LYP（VWN-RPA）と B3LYP5 のどちらにも同じ値を使い、その旨を利用者に示す。

== 勾配

$E_"disp"$ は距離 $r_(A B)$ と配位数を通して核座標に依る。対ごとのエネルギーを
$e_(A B) = -sum_n s_n C_n^(A B) slash r^n f_n$ と書く。

*$C_6$ についての微分。* @eq:d3-bj でも @eq:d3-zero でも $R_0^(A B)$ は配位数に依らず、
$C_8$ は $C_6$ に比例するので、$e_(A B)$ は $C_6^(A B)$ の1次式である：$partial e_(A B) slash partial C_6^(A B) = e_(A B) slash C_6^(A B)$。

*距離についての微分。*

$ (partial e_(A B)) / (partial r)
  = sum_n s_n C_n^(A B) n r^(n-1) / (r^n + (a_1 R_0 + a_2)^n)^2 quad ("BJ"), quad
  (partial e_(A B)) / (partial r)
  = sum_n s_n C_n^(A B) f_n / r^(n+1) [n - 6 alpha_n t_n^(-alpha_n) f_n], quad t_n = r / (s_(r,n) R_0) quad ("ゼロ") $ <eq:d3-dedr>

*配位数の微分。* @eq:d3-cn の各項を $g_(A B) = 1 slash (1 + e^u)$、$u = -k_1 (R_(A B)^c slash r - 1)$、$R_(A B)^c = k_2 (R_A^"cov" + R_B^"cov")$ とすると

$ (dif g_(A B)) / (dif r) = -g_(A B) (1 - g_(A B)) k_1 R_(A B)^c / r^2 $

$g_(A B)$ は $"CN"_A$ と $"CN"_B$ の両方に入る。@eq:d3-c6 の分子・分母を $Z, W$ とすると

$ (partial C_6^(A B)) / (partial "CN"_A) = (Z' W - Z W') / W^2, quad
  Z' = sum_(i j) C_(6, "ref")^(A B, i j) (partial L_(i j)) / (partial "CN"_A), quad
  W' = sum_(i j) (partial L_(i j)) / (partial "CN"_A), quad
  (partial L_(i j)) / (partial "CN"_A) = -2 k_3 ("CN"_A - "CN"_A^i) L_(i j) $

（$"CN"_B$ も同様。）各原子の $D_A = partial E_"disp" slash partial "CN"_A = sum_(B != A) (e_(A B) slash C_6^(A B)) partial C_6^(A B) slash partial "CN"_A$ を先に求めておくと、

$ (dif E_"disp") / (dif vb(R)_A) = sum_(B != A) [(partial e_(A B)) / (partial r) + (D_A + D_B) (dif g_(A B)) / (dif r)]
  (vb(R)_A - vb(R)_B) / r_(A B) $ <eq:d3-grad>

（連鎖律による導出で、文献の式ではない。）各対は $vb(R)_A - vb(R)_B$ だけに依るので並進不変性
$sum_A dif E_"disp" slash dif vb(R)_A = vb(0)$ が厳密に成り立つ。検証は中心差分（@sec:grad の検証用の恒等式と同じ）で行う。

== 全エネルギーと全勾配への加え方

$E_"disp"$ は密度に依らないので、SCF（Fock 行列・収束）には入れず、収束後に足すだけである：

$ E_"tot" = E_"SCF" + E_"disp", quad
  (dif E_"tot") / (dif vb(R)_A) = (dif E_"SCF") / (dif vb(R)_A) + (dif E_"disp") / (dif vb(R)_A) $ <eq:d3-total>

$E_"SCF"$ は @eq:e-rks（HF では @eq:e-rhf）、その勾配は @eq:grad、$E_"disp"$ の勾配は @eq:d3-grad。
構造最適化（@sec:opt）はこの $E_"tot"$ とその勾配を最小化する。分散補正を指定しなければ $E_"tot" = E_"SCF"$ のままである。

= 振動数解析と熱化学 <sec:freq>

解析的勾配（@eq:grad、分散補正があれば @eq:d3-total）を中心差分して Hessian を作り、
質量加重して並進・回転を除いた内部空間で対角化し、調和振動数・基準振動・IR 強度を求める。
さらに理想気体・剛体回転子・調和振動子の近似で零点エネルギー、エンタルピー、Gibbs エネルギーを求める
（Wilson, Decius, Cross 1955；Ochterski 1999「Vibrational Analysis in Gaussian」；McQuarrie, Simon 1999；Ochterski 2000「Thermochemistry in Gaussian」）。

対象外とするもの：解析的 Hessian、低振動数の補正（qRRHO など）、対称数の自動判定、
途中からの再開、変位ごとの計算の並列化、開殻（電子の縮重度は 1 に固定）。

振動数は停留点（勾配がほぼ 0 の構造、@sec:opt で最適化した構造）でだけ意味を持つ。
停留点でない構造では回転の分離（@eq:tr-vectors）が厳密でなくなるので、実装は勾配の大きさを利用者に示す。

== 中心差分による Hessian

デカルト座標 $vb(x) in RR^(3 N_"atom")$（$a_0$）、全エネルギー $E_"tot"$（@eq:d3-total）の勾配 $vb(g)(vb(x))$ について

$ H_(i j) = (g_i (vb(x) + h vb(e)_j) - g_i (vb(x) - h vb(e)_j)) / (2 h), quad h = 5 times 10^(-3) thin a_0 $ <eq:hess-fd>

（$vb(e)_j$ は座標 $j$ の単位ベクトル。）1回の変位で勾配の全成分が得られるので、列 $j$ ごとに2回、
合計 $6 N_"atom"$ 回の勾配計算になる。変位した構造の SCF の初期値には基準構造の収束した密度行列を使う
（収束先は変わらず、反復が減るだけ）。単位は $E_h slash a_0^2$。

*誤差。* Taylor 展開より $H_(i j)$ の打ち切り誤差は $h^2 slash 6 dot partial^3 g_i slash partial x_j^3 = O(h^2)$。
勾配の誤差を $epsilon$ とすると、差分による誤差は $epsilon slash h$ 程度である。
勾配を計算する SCF は $max abs(e_(mu nu)) < 10^(-10)$ まで収束させる（@sec:grad）ので $epsilon tilde 10^(-10)$ 程度と見込み、
$h = 5 times 10^(-3)$ では $epsilon slash h tilde 2 times 10^(-8) thin E_h slash a_0^2$、打ち切り誤差は $h^2 slash 6 approx 4 times 10^(-6)$ に3階微分を掛けたものとなる。
実測では打ち切り誤差のほうが支配的である。水/STO-3G の RHF（勾配用の SCF 収束条件 $max abs(e_(mu nu)) < 10^(-10)$）で
$h = 10^(-2), 5 times 10^(-3), 2.5 times 10^(-3), 1.25 times 10^(-3) thin a_0$ と変えると、
非対称性 $delta_"asym"$（@eq:hess-sym）は $3.43 times 10^(-5), 8.57 times 10^(-6), 2.14 times 10^(-6), 5.35 times 10^(-7) thin E_h slash a_0^2$ となり、
$h$ を半分にするたびにちょうど $1 slash 4$ になる。SCF の収束条件を $10^(-10)$ から $10^(-12)$ に締めても4桁目しか変わらない。
つまり HF の既定の $h$ では誤差は $O(h^2)$ で、$epsilon slash h$ の項は見えない。
DFT では格子による勾配の誤差が加わる。$h$ は利用者が変更できるようにする。
（この式と誤差の見積もりは差分の定義から導いたもので、文献の式ではない。）

== 対称化

差分の誤差のため $H$ は厳密には対称でない。対称化した行列を使い、非対称性を診断値として利用者に示す：

$ H <- 1/2 (H + H^T), quad delta_"asym" = max_(i j) abs(H_(i j) - H_(j i)) $ <eq:hess-sym>

DFT では格子の重みの核座標依存性を無視している（@eq:xc-grad）ので、勾配はエネルギーの厳密な微分ではなく、
$delta_"asym"$ は格子が粗いほど大きくなる。並進不変性 $sum_A H_(A_x, j) = 0$ も同じ理由で近似的にしか成り立たない。

HF でも並進不変性は打ち切り誤差の範囲でしか成り立たない。勾配はどの構造でも $sum_A dif E slash dif vb(R)_A = vb(0)$ を満たすので、
対称化する前の $H$ の各列は $sum_A H_(A_x, j) = 0$ を $epsilon slash h$ の範囲で満たす。
ところが対称化した $H$ の和には転置した側 $sum_A H_(j, A_x)$ も入り、その打ち切り誤差
$h^2 slash 6 dot partial^3 g_j slash partial x_(A_x)^3$ は $A$ ごとに異なって打ち消し合わない。
そのため対称化した $H$ の並進の和 $max_(alpha, j) abs(sum_A H_(A_alpha, j))$ も $O(h^2)$ で小さくなるだけである。
@eq:hess-fd と同じ水/STO-3G の RHF で、$h = 10^(-2), 5 times 10^(-3), 2.5 times 10^(-3), 1.25 times 10^(-3) thin a_0$ に対し
$5.78 times 10^(-6), 1.45 times 10^(-6), 3.61 times 10^(-7), 9.03 times 10^(-8) thin E_h slash a_0^2$ であった（$h$ を半分にするたびに $1 slash 4$）。
並進・回転の成分は @eq:tr-vectors で取り除く。

== 質量加重

原子 $A$ の質量を $m_A$（統一原子質量単位 u）とし、座標 $i$ の属する原子の質量を $m_i$ と書く：

$ tilde(H)_(i j) = H_(i j) / sqrt(m_i m_j) $ <eq:mass-weight>

（単位は $E_h slash (a_0^2 thin "u")$。）質量加重座標は $q_i = sqrt(m_i) x_i$ である。

*質量の既定値。* IUPAC/CIAAW 2013 の標準原子量（Meija ら 2016。範囲で与えられる元素は慣用値）を使う：

#align(center, table(
  columns: 6, stroke: 0.4pt, inset: 4pt, align: center,
  [元素], [u], [元素], [u], [元素], [u],
  [H], [1.008], [He], [4.002602], [Li], [6.94],
  [Be], [9.0121831], [B], [10.81], [C], [12.011],
  [N], [14.007], [O], [15.999], [F], [18.998403163],
  [Ne], [20.1797], [Na], [22.98976928], [Mg], [24.305],
  [Al], [26.9815385], [Si], [28.085], [P], [30.973761998],
  [S], [32.06], [Cl], [35.45], [Ar], [39.948],
))

利用者は原子ごとに質量を与えてよい（同位体置換など）。
PySCF の振動解析の既定の質量（参照値に書き出した `masses_u`）は、H、C、N、O についてこの表と一致した
（PySCF のコードは読まず、出力した数値で比べた）。

== 並進と回転の分離

質量の中心 $vb(R)_"cm" = sum_A m_A vb(R)_A slash M$（$M = sum_A m_A$）を原点とし、慣性テンソル
$I = sum_A m_A (abs(vb(r)_A)^2 bb(1) - vb(r)_A vb(r)_A^T)$、$vb(r)_A = vb(R)_A - vb(R)_"cm"$ を対角化して
主慣性モーメント $I_1 <= I_2 <= I_3$（$"u" thin a_0^2$）と主軸 $vb(u)_1, vb(u)_2, vb(u)_3$ を得る。
質量加重座標での並進 $vb(t)_alpha$（$alpha = x, y, z$）と主軸まわりの回転 $vb(r)_a$ のベクトルの、原子 $A$ の成分は

$ (vb(t)_alpha)_A = sqrt(m_A) thin vb(e)_alpha, quad
  (vb(r)_a)_A = sqrt(m_A) thin vb(u)_a times vb(r)_A $ <eq:tr-vectors>

である。重心を原点にとり主軸を使うと、これら6本は互いに直交し、$abs(vb(t)_alpha)^2 = M$、$abs(vb(r)_a)^2 = I_a$ となる
（Eckart 条件。Wilson ら 1955）。並進・回転のベクトルの作り方は Ochterski（1999）と同じ（同文書は残りの基底を Schmidt 直交化で作るが、ここでは下のとおり射影の対角化で作る。内部ブロックの固有値はどちらでも同じ）。

- 回転 $a$ は $I_a > 10^(-6) I_3$ かつ $I_3 > 10^(-20) sum_A m_A abs(vb(R)_A)^2$ のときだけ残す（$vb(R)_A$ は入力の座標原点からの位置）。直線分子では $I_1 approx 0$ なので 5 本、原子（$I_3 = 0$）では回転はなく 3 本になる。2つ目の条件は、原子1個のとき $vb(R)_"cm"$ の丸め誤差で $vb(r)_A$ に残る $abs(vb(R)_A)$ の数 ulp 程度の残差（$I_3 tilde 10^(-31) m_A abs(vb(R)_A)^2$）から回転を作らないためのものである。
  これらの閾値は数値上の取り決めで、文献の値ではない。直線分子かどうか（@eq:thermo-rot）もこの判定で決め、残した回転がちょうど 2 本のとき直線分子とする。
- 残したベクトルを正規化し、念のため修正 Gram–Schmidt 法で直交化して $n_"tr"$ 本の正規直交系 $D_"tr"$ を作る。
- 射影 $P = bb(1) - D_"tr" D_"tr"^T$ を対角化し、固有値が 1 の固有ベクトル（$3 N_"atom" - n_"tr"$ 本）を内部空間の正規直交基底 $D_"int"$ とする
  （固有値は 0 と 1 だけなので、$1 slash 2$ を境に分ければよい）。
- 内部空間のブロック $D_"int"^T tilde(H) D_"int"$ を対角化し、固有値 $lambda_k$ と固有ベクトル $vb(c)_k$ から
  質量加重の基準振動 $vb(l)_k = D_"int" vb(c)_k$（$abs(vb(l)_k) = 1$）を得る。$P tilde(H) P$ を対角化して 0 に近い固有値を捨てる方法は、
  振動数の小さい本物の振動と区別がつかないので使わない。

デカルト座標での変位は $x_i = l_(k, i) slash sqrt(m_i)$ に比例する。
*正規化の約束。* 実装が返す基準振動は質量加重の単位ベクトル $vb(l)_k$（$abs(vb(l)_k) = 1$、単位なし）である。
デカルト変位のベクトル $vb(d)_k$（$d_(k, i) = l_(k, i) slash sqrt(m_i)$、単位 $"u"^(-1 slash 2)$）は正規化しない。
$vb(d)_k$ は、質量加重の基準座標 $Q_k$（$sqrt("u") thin a_0$）を 1 だけ動かしたときのデカルト変位（$a_0$）にあたる。
$vb(l)_k$ の符号と、縮退した振動の部分空間の中での基底の選び方は任意である。
換算質量は

$ mu_k = (sum_i l_(k, i)^2 / m_i)^(-1) = 1 / abs(vb(d)_k)^2 $ <eq:reduced-mass>

（u）である。$vb(d)_k$ を単位ベクトルに正規化した変位で動く有効質量にあたる（Ochterski 1999 の換算質量と同じ考え方だが、同文書の式との照合はしていない）。$mu_k$ は $vb(l)_k$ の符号には依らないが、縮退した振動では部分空間の基底の選び方に依る。

== 調和振動数

$lambda_k$ の単位 $E_h slash (a_0^2 thin "u")$ を原子単位（$E_h slash (a_0^2 m_e)$）に直すと角振動数 $omega_k = sqrt(lambda_k m_e slash m_u)$（$E_h slash ħ$）、
波数は $tilde(nu)_k = omega_k E_h slash (2 pi ħ c)$ である：

$ tilde(nu)_k = cases(
  c_nu sqrt(lambda_k) & (lambda_k >= 0),
  -c_nu sqrt(-lambda_k) & (lambda_k < 0)
), quad
c_nu = E_h / (h c) sqrt(m_e / m_u) = 5140.487 thin "cm"^(-1) slash sqrt(E_h slash (a_0^2 thin "u")) $ <eq:freq>

（定数は CODATA 2018：$E_h slash h c = 219474.6313632 thin "cm"^(-1)$、$m_e slash m_u = 5.48579909065 times 10^(-4)$。
実装ではこの2つから $c_nu$ を計算する。）$lambda_k < 0$（虚数振動数）は負の波数で表し、熱化学からは除く。
振動数の小さい正の振動はそのまま熱化学に入れる（低振動数の補正は対象外）。

== IR 強度

双極子モーメント $vb(mu)$（@eq:dipole、$e thin a_0$）を Hessian と同じ変位で計算し、中心差分で微分する：

$ (partial vb(mu)) / (partial x_j) = (vb(mu)(vb(x) + h vb(e)_j) - vb(mu)(vb(x) - h vb(e)_j)) / (2 h), quad
  (partial vb(mu)) / (partial Q_k) = sum_j (partial vb(mu)) / (partial x_j) l_(k, j) / sqrt(m_j) $

（$Q_k$ は質量加重の基準座標、単位 $sqrt("u") thin a_0$。$partial vb(mu) slash partial Q_k$ の単位は $e slash sqrt("u")$。）
基本音の積分吸収係数（自然対数による、二重調和近似）は

$ A_k = N_A / (12 epsilon_0 c^2) abs((partial vb(mu)) / (partial Q_k))^2 $ <eq:ir-intensity>

（SI 単位。CGS（Gauss 単位）では $4 pi epsilon_0 = 1$ として $A_k = N_A pi slash (3 c^2) abs(partial vb(mu) slash partial Q_k)^2$ と同じ式になる。）
$A_k$ は振動 $k$ の吸収帯の、自然対数のモル吸光係数を波数で積分したものである。$partial vb(mu) slash partial Q_k$ を $"C" slash sqrt("kg")$ で入れると $A_k$ は m/mol になる。
原子単位からの換算は

$ A_k ["km/mol"] = (N_A e^2) / (12 epsilon_0 c^2 m_u) times 10^(-3) dot abs((partial vb(mu)) / (partial Q_k))^2 [e^2 slash "u"]
  = 974.8801 dot abs((partial vb(mu)) / (partial Q_k))^2 [e^2 slash "u"] $

（CODATA 2018 の $N_A$、$e$、$epsilon_0$、$c$、$m_u$ から。$("D"slash angstrom)^2 slash "u"$ 単位なら係数は $42.2561$。）
実装は定数から係数を計算する。電荷 $q$ を持つ分子では $vb(mu)$ が原点に依るが（原点を $vb(c)$ ずらすと $-q vb(c)$ だけ変わる）、
差は定数なので $partial vb(mu) slash partial x_j$ は原点に依らない。また基準振動 $vb(l)_k$ は質量加重の並進と直交する（重心を動かさない）ので、
分子全体の並進による $vb(mu)$ の変化は $A_k$ に入らない。
$A_k$ はすべての内部振動（虚数振動数のものを含む。その値に物理的な意味はない）について、振動数と同じ順に出す。
双極子は Hessian の変位の SCF ごとに計算するので、余分な SCF は要らない。
縮退した振動では個々の $A_k$ が縮退部分空間の基底の選び方に依るが、縮退群での和は基底に依らない。
*未確認：* この式と係数を Wilson ら（1955）の該当箇所で照合していない。
使った PySCF には IR 強度の機能がないので、比較テストの参照値は PySCF の双極子と基準振動にこの式を当てたもので、
双極子の微分と基準振動を確かめるが、この式と係数の定義そのものは確かめない。

== 熱化学

温度 $T$（既定 298.15 K）、圧力 $p$（既定 101325 Pa）。理想気体・剛体回転子・調和振動子の近似で、
すべて1分子あたり（$E_h$、$E_h slash "K"$）で計算する。$k_B$ は CODATA 2018 の $E_h slash k_B = 315775.02480407 thin "K"$ から、
並進の分配関数だけは SI（$h$、$k_B$ は定義値、$m_u = 1.66053906660 times 10^(-27)$ kg）で計算する。
以下は Ochterski（2000）の式（McQuarrie と Simon 1999 の式を引いたもの）で、そこではモルあたり（$R = N_A k_B$）で書かれている。

*並進。* $M$ は分子の質量：

$ q_"t" = ((2 pi M k_B T) / h^2)^(3 slash 2) (k_B T) / p, quad
  E_"t" = 3/2 k_B T, quad S_"t" = k_B (ln q_"t" + 5/2), quad C_"v,t" = 3/2 k_B $ <eq:thermo-trans>

*回転。* 回転温度 $Theta_a = ħ^2 slash (2 I_a k_B)$（原子単位では $Theta_a = (E_h slash k_B) slash (2 I_a m_u slash m_e)$、$I_a$ は $"u" thin a_0^2$）。
対称数 $sigma$ は利用者が与え、既定は 1（自動判定は対象外）。上の行が直線分子（@eq:tr-vectors の判定で回転が2本、$Theta = Theta_2 = Theta_3$）、
下の行が非直線分子：

$ q_"r" = T / (sigma Theta), quad E_"r" = k_B T, quad S_"r" = k_B (ln q_"r" + 1), quad C_"v,r" = k_B \
  q_"r" = sqrt(pi) / sigma (T^(3 slash 2)) / sqrt(Theta_1 Theta_2 Theta_3), quad
  E_"r" = 3/2 k_B T, quad S_"r" = k_B (ln q_"r" + 3/2), quad C_"v,r" = 3/2 k_B $ <eq:thermo-rot>

原子では回転の寄与はすべて 0。

*振動。* 実の振動数（$tilde(nu)_k > 0$）ごとに $Theta_k = omega_k E_h slash k_B$（$omega_k$ は @eq:freq の $E_h slash ħ$ 単位）、$x_k = Theta_k slash T$ として、
エネルギーの原点を電子エネルギーの極小（ポテンシャルの底）にとる：

$ E_"ZPE" = sum_k 1/2 k_B Theta_k, quad
  E_"v" = sum_k k_B Theta_k (1/2 + 1 / (e^(x_k) - 1)), \
  S_"v" = k_B sum_k [x_k / (e^(x_k) - 1) - ln(1 - e^(-x_k))], quad
  C_"v,v" = k_B sum_k e^(x_k) (x_k / (e^(x_k) - 1))^2 $ <eq:thermo-vib>

$E_"v"$ は零点エネルギーを含む。虚数振動数は和に入れない。

*電子。* 閉殻一重項だけを扱うので電子状態の縮重度は 1、励起状態は無視する：

$ q_"e" = 1, quad E_"e" = 0, quad S_"e" = 0 $ <eq:thermo-elec>

*合計。* 電子エネルギー $E_"el"$ は基準構造の $E_"tot"$（@eq:d3-total。分散補正なしなら $E_"SCF"$）とする：

$ U = E_"el" + E_"t" + E_"r" + E_"v" + E_"e", quad
  H = U + k_B T, quad
  S = S_"t" + S_"r" + S_"v" + S_"e", quad
  G = H - T S $ <eq:thermo-total>

あわせて $E_"el" + E_"ZPE"$ と $C_"v" = C_"v,t" + C_"v,r" + C_"v,v"$、$C_"p" = C_"v" + k_B$（理想気体）を示す。

*実装上の注意。* 振動の $Theta_k$ は波数から $Theta_k = tilde(nu)_k (E_h slash k_B) slash (E_h slash h c)$ で求める（@eq:freq の定数）。
$tilde(nu)_k <= 0$ の振動（虚数と、ちょうど 0 のもの）は和に入れず、その数を利用者に示す。
$x_k$ が大きいときに桁あふれしないよう、$C_"v,v"$ の項は $x_k^2 e^(-x_k) slash (1 - e^(-x_k))^2$ と、
$ln(1 - e^(-x_k))$ と $e^(x_k) - 1$ は $"ln1p"$、$"expm1"$ で計算する（式は同じ）。
振動の数は $3 N_"atom" - n_"tr"$（@eq:tr-vectors）でなければならない。

*PySCF との比較。* PySCF の `thermo.thermo` の出力から $k_B T = H - U$ を逆算すると、$E_h slash k_B$ が
CODATA 2018 の値より相対 $3.3 times 10^(-7)$ 大きい $315775.129 thin "K"$ になる（CODATA 2014 の値 $315775.13 thin "K"$ に近い。PySCF のコードは読まず、出力した数値から求めた）。
そのまま比べると $k_B T$ に比例する量が相対 $3$〜$4 times 10^(-7)$ ずれる。上の式の温度依存はすべて $k_B T$ を通してだけ入るので、
比較テストでは $k_B T$ が PySCF と等しくなる温度 $T' = T (H - U)_"PySCF" slash (k_B T)$ で計算し、
エネルギー（$E$、$H$、$G$）はそのまま、$S$ と $C_"v"$ は $T' S$ と $T S_"PySCF"$ のように温度を掛けて $E_h$ で比べる。
