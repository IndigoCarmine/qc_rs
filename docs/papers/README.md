# 論文置き場

実装を原論文と照合するための PDF をここに置きます。`docs/papers/*.pdf` は `.gitignore` に入っており、
リポジトリには含めません（再配布しないため）。ファイル名は下の表のとおりにしてください。

## CAM-B3LYP（`Feat/cam-b3lyp`）

| ファイル名 | 論文 | DOI | 確認する内容 |
|---|---|---|---|
| `yanai2004_cam-b3lyp.pdf` | T. Yanai, D. P. Tew, N. C. Handy, *Chem. Phys. Lett.* **393**, 51 (2004) | 10.1016/j.cplett.2004.06.011 | CAM-B3LYP の定義。演算子の分割と α = 0.19、β = 0.46、μ = 0.33、相関の混合 |
| `iikura2001_lc.pdf` | H. Iikura, T. Tsuneda, T. Yanai, K. Hirao, *J. Chem. Phys.* **115**, 3540 (2001) | 10.1063/1.1383587 | GGA 交換の短距離化（a、b、c を使う減衰因子の式） |
| `tawada2004_lc-tddft.pdf` | Y. Tawada, T. Tsuneda, S. Yanagisawa, T. Yanai, K. Hirao, *J. Chem. Phys.* **120**, 8425 (2004) | 10.1063/1.1688752 | 同じ減衰因子の再掲と、長距離補正の実装上の注意 |
| `gill1996_attenuated.pdf` | P. M. W. Gill, R. D. Adamson, *Chem. Phys. Lett.* **261**, 105 (1996) | 10.1016/0009-2614(96)00931-1 | erf で減衰させた Coulomb 演算子の積分 |
| `becke1988_b88.pdf` | A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) | 10.1103/PhysRevA.38.3098 | B88 交換（減衰因子の中の K_σ の元） |
| `lyp1988.pdf` | C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) | 10.1103/PhysRevB.37.785 | LYP 相関 |
| `vwn1980.pdf` | S. H. Vosko, L. Wilk, M. Nusair, *Can. J. Phys.* **58**, 1200 (1980) | 10.1139/p80-159 | VWN5 のパラメータ |

7 件とも、doi.org で DOI を解決して著者・題名・巻・ページが上の表と一致することを確認済みです。
