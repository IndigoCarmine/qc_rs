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

== 球面調和 d 関数（5d）

d 殻は既定で実球面調和関数 5 成分に変換する（デカルト 6 成分のまま使うこともできる）。
規格化済みデカルト成分 $chi_(x x), chi_(x y), ...$ を用いて、$m = -2, ..., 2$ の順に

$ d_(-2) &= chi_(x y), quad d_(-1) = chi_(y z), quad d_(+1) = chi_(x z), \
  d_0 &= chi_(z z) - 1/2 chi_(x x) - 1/2 chi_(y y), \
  d_(+2) &= sqrt(3)/2 (chi_(x x) - chi_(y y)) $ <eq:sph-d>

すべて規格化されている（$bra(chi_(x x)) ket(chi_(y y)) = 1 slash 3$ を使う）。
s, p 殻はデカルトのまま。積分・微分積分はデカルト成分で計算し、最後に @eq:sph-d の
$5 times 6$ 行列を各添字に掛ける。基底殻は $L <= 2$ を対象とするが、
原始積分の漸化式は任意の $L$ で動くように実装する（勾配で $L + 1$ が必要になる）。

== 基底ファイル（Gaussian94 形式）

`data/basis/*.gbs`（Basis Set Exchange より取得）。`!` で始まる行はコメント。元素ごとに
`記号 0` の行で始まり `****` で終わる。殻は `種類 原始数 スケール` の行（種類は `S`, `P`, `D`, `SP`）に
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
（線形従属）。対称行列の固有値分解は faer クレートに任せる（三重対角化と分割統治法）。
検証用の参照実装として巡回 Jacobi 法も持ち、テストで両者を照合する。Jacobi 法では
非対角要素 $a_(p q)$ を消す回転角は

$ theta = (a_(q q) - a_(p p)) / (2 a_(p q)), quad t = "sgn"(theta) / (abs(theta) + sqrt(theta^2 + 1)), quad
  c = 1 / sqrt(t^2 + 1), quad s = t c $ <eq:jacobi>

で、非対角要素の2乗和が十分小さく（全体の $10^(-22)$ 倍以下）なるまで掃引を繰り返し、
条件を満たしたあとさらに1回掃引する（Jacobi 法は2次収束なので、これで丸め誤差の水準に達する）。
回転は $A <- G^T A G$（$G_(p p) = G_(q q) = c$、$G_(p q) = s$、$G_(q p) = -s$）、$"sgn"(0) = +1$ とする。

== SCF 手順

+ $S, T, V$、ERI、$X$ を計算する。
+ 初期値：コアハミルトニアン $h$ を @eq:roothaan の $F$ として解き、$P$ を作る。
+ $P$ から $F$ を作り、エネルギーを計算する。
+ DIIS で $F$ を外挿する（下記）。
+ 外挿した $F$ を対角化して新しい $P$ を得る。
+ 収束判定：$abs(Delta E) < 10^(-10)$ かつ $max abs(e_(mu nu)) < 10^(-7)$。未収束なら 3 へ。
  最大反復回数（既定 200）を超えたらエラーを返す。

収束後、収束した $P$ から作った $F$ を対角化した $epsilon, C$ を結果とする。
密度行列とエネルギーもこの $C$ から作り直し、$E$、$P$、$epsilon$、$C$ が互いに整合するようにする
（勾配の @eq:W がこの整合性を必要とする）。

== DIIS

誤差行列（直交化基底での交換子）

$ e = X^T (F P S - S P F) X $ <eq:diis-err>

直近 $m <= 8$ 回の $(F_k, e_k)$ を保持し、$B_(k l) = sum_(mu nu) (e_k)_(mu nu) (e_l)_(mu nu)$ として

$ mat(B, -vb(1); -vb(1)^T, 0) vec(vb(c), lambda) = vec(vb(0), -1), quad F_"ext" = sum_k c_k F_k $ <eq:diis>

を解く（部分ピボット付き Gauss 消去）。保持数が 2 以上になったら外挿を使う。
連立方程式が特異なら最古の組を捨ててやり直す。

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

== Becke 88 交換

スピンごとの密度 $rho_s = rho slash 2$、勾配 $abs(nabla rho_s) = sqrt(sigma) slash 2$ を用いて

$ x = (sqrt(sigma) slash 2) / (rho slash 2)^(4 slash 3), quad
  f_x^"B88" = f_x^"S" - 2 beta (rho / 2)^(4 slash 3) x^2 / (1 + 6 beta x asinh(x)), quad beta = 0.0042 $ <eq:b88>

== VWN 相関（LDA）

$ r_s = (3 / (4 pi rho))^(1 slash 3), quad x = sqrt(r_s), quad X(x) = x^2 + b x + c, quad Q = sqrt(4c - b^2) $

$ epsilon_c^"VWN" = A {ln(x^2 / X(x)) + (2b) / Q arctan(Q / (2x + b))
  - (b x_0) / X(x_0) [ln((x - x_0)^2 / X(x)) + (2(b + 2 x_0)) / Q arctan(Q / (2x + b))]} $ <eq:vwn>

$ f_c^"VWN" = rho thin epsilon_c^"VWN" $

常磁性（閉殻）のパラメータ：

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
実装は「$f$ の線形結合（係数と成分の列）＋ $a_"HF"$」という一般形にしておく。

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

運動エネルギー・核引力・ERI も同じ規則に従う。球面調和への変換 @eq:sph-d は微分と可換。

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
