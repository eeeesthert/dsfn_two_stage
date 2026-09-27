# Equations implemented

Normalized DLT estimates `x_ref ~ H x_target`. RANSAC minimizes Euclidean reprojection error. Formula-mode kernels implement the real part of the article's DC-compensated Gabor expression with `k_v=2^{-(v+2)/2}π` and `φ_u=uπ/6`. Local energy is a configurable window mean of squared response. Texture energy is `Eg=Σ||Vg_ref−Vg_target||²`; Eq. (10) global spread is `S=sqrt(Σ||p_i−mean(p)||²/k)`; refinement stacks `[sqrt(λg)Rg, sqrt(λs)Rs]`. Fusion uses `w1=D1/(D1+D2+eps)` and `w2=1-w1` in overlap.
