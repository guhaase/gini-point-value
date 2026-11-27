import numpy as np
from scipy.stats import norm
pi, lgd, acc = 0.05, 0.45, 0.80
s = np.linspace(-8, 10, 200001); ds = s[1]-s[0]
def irb_retail(pd, lgd=0.45, R=0.15):
    pd = np.clip(pd, 1e-6, 1-1e-6)
    return lgd*(norm.cdf((norm.ppf(pd)+np.sqrt(R)*norm.ppf(0.999))/np.sqrt(1-R)) - pd)
print("AUC   KS    Gini  BR_aprov  EL_aprov(bps)  K_med(%)")
for auc in [0.70,0.72,0.74,0.76,0.78,0.80,0.82]:
    d = np.sqrt(2)*norm.ppf(auc)
    fG, fB = norm.pdf(s), norm.pdf(s-d)   # high score = riskier
    mix = (1-pi)*fG + pi*fB
    pd_s = pi*fB/mix
    ks = 2*norm.cdf(d/2)-1
    # approve lowest-risk 80%
    cdf = np.cumsum(mix)*ds; cut = s[np.searchsorted(cdf, acc)]
    m = s <= cut
    br = (pi*fB[m]).sum()*ds/(mix[m].sum()*ds)
    K = (irb_retail(pd_s)*mix).sum()*ds
    print(f"{auc:.2f}  {ks:.3f} {2*auc-1:.2f}  {br:.4f}   {br*lgd*1e4:7.1f}       {K*100:.3f}")
