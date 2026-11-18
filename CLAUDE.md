# qc_rs

- ブランチ方針は `CONTRIBUTING.md` に従う：結果を変えない高速化は `Imp/`、式・実装の誤りの修正は `Bug/`、計算理論の追加は `Feat/`。作業を始める前に、どの種類かを決めてブランチを切る。
- 数式の仕様は `docs/theory.typ`。式を変える・足すときは仕様書を先に直し、コードのコメントで式ラベル（`eq:...`）を引く。
- PySCF は参照値を作るブラックボックスとして実行するだけで、コードは読まない（libxc、libcint なども同じ）。
- 依存は BSD 3-Clause と両立し、Windows・Linux・macOS で素の `cargo build` が通るものに限る。
