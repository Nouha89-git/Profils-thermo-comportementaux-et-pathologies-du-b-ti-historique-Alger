# =============================================================================
# THERMO-BEHAVIORAL ANALYSIS — ALGIERS
# =============================================================================

# ── CELL 1: Install ──────────────────────────────────────────
# !pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels openpyxl -q


# ── CELL 2: Imports + Journal Style Config ──────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import svd
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score,
                              classification_report, confusion_matrix)
from itertools import combinations
import warnings, os
warnings.filterwarnings("ignore")

# ── Palette (luminance-checked for greyscale print) ─────────
NAVY, STEEL, TEAL, GREEN, AMBER = '#1a4f6b','#3d7a99','#2e8b8b','#4a8c5f','#c99a2e'
ORANGE, RED, GREY, PURPLE       = '#d17a37','#b3452e','#6b6b6b','#7d5ba6'
CLUSTER_COLORS = {'C0': GREY, 'C1': ORANGE, 'C2': GREEN, 'C3': PURPLE}
PAL = {0: GREY, 1: ORANGE, 2: GREEN, 3: PURPLE}

cmap_journal = LinearSegmentedColormap.from_list("journal", ["#FFFFFF", AMBER, RED])
cmap_blue    = LinearSegmentedColormap.from_list("journalblue", ["#FFFFFF", STEEL, NAVY])

plt.rcParams.update({
    'font.family':'Arial', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.grid':True, 'grid.alpha':0.25, 'savefig.dpi':600,
})

def save(name):
    """Save PNG at 600dpi, no title, journal style already applied."""
    plt.tight_layout()
    plt.savefig(f"/content/{name}.png", format="png", dpi=600, bbox_inches="tight")
    plt.show()
    print(f"saved: /content/{name}.png")

SURVEY_YEAR = 2024
print("Journal style loaded: Arial | no titles | PNG @600dpi | fixed palette")


# ── CELL 3: Load + Clean ─────────────────────────────────────
df_raw = pd.read_excel("article_rymel.xlsx", sheet_name="nouha_draft")

df = df_raw.rename(columns={
    "Age":"age","Gender":"gender","Occupant status":"occupant_status",
    "Occupation since":"occ_since","Housing type":"housing_type",
    "Area (m²)":"area","Floor":"floor",
    "Are you on the top floor ?":"top_floor",
    "Number of shared walls":"shared_walls",
    "Facade orientation":"orientation",
    "Windows material":"window_material","Glass type":"glass_type",
    "Shutter state":"shutter_state","Roof waterproofing status":"roof_state",
    "Does air get through closed windows?":"air_infiltr",
    "Water infiltration":"water_infiltr","Moisture and mold":"moisture",
    "Do you experience cold drafts frequently?":"cold_drafts",
    "How would you rate the temperature in the rooms facing the street (or the main facade)?":"thermal_sensation",
    "Summer thermal satisfaction":"sat_summer",
    "Winter thermal satisfaction":"sat_winter",
    "Rate thermal comfort":"comfort_score",
    "Turn on a fan/the AC":"beh_fan_ac",
    " Put on lighter clothes":"beh_light_clothes",
    "Close the shutters":"beh_close_shutters",
    "Open the windows":"beh_open_windows",
    "Turn on or turn up the heating":"beh_heating",
    "Put on warmer clothes":"beh_warm_clothes",
    "Close the windows":"beh_close_windows",
})

sat_map = {"-2 dissatisfied":-2,"-1 Somewhat dissatisfied":-1,
           "0 Acceptable":0,"+1 Somewhat satisfied":1,"+2 Satisfied":2}
for col in ["sat_summer","sat_winter"]:
    df[col+"_n"] = df[col].map(sat_map)

sens_map = {"-2 Freezing":-2,"-1 Slightly chilly":-1,
            "0 Neutral":0,"+1 Slightly hot":1,"+2 Hot":2}
df["sensation_n"] = df["thermal_sensation"].map(sens_map)

for col in ["air_infiltr","water_infiltr","moisture","cold_drafts"]:
    df[col+"_b"] = (df[col]=="Yes").astype(int)

df["single_glaz_b"] = df["glass_type"].apply(
    lambda x: 0 if "Double" in str(x) and "Single" not in str(x) else 1)

for col, m in [("shutter_state",{"Good":2,"Medium":1,"Degraded":0}),
               ("roof_state",   {"Good":2,"Medium":1,"Degraded":0})]:
    df[col+"_n"] = df[col].map(m)

df["occ_years"] = SURVEY_YEAR - df["occ_since"]

print(f"Data loaded: {df.shape[0]} households x {df.shape[1]} variables")


# ── CELL 4: fig01 — Respondent profile ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

vc = df["gender"].value_counts()
axes[0,0].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
              colors=[STEEL, RED, GREEN], startangle=90)

vc2 = df["occupant_status"].value_counts()
bars = axes[0,1].barh(vc2.index, vc2.values, color=STEEL, edgecolor="white")
for bar,v in zip(bars,vc2.values):
    axes[0,1].text(v+0.3, bar.get_y()+bar.get_height()/2,
                   f"{v} ({100*v/125:.1f}%)", va="center", fontsize=8)
axes[0,1].set_xlim(0, vc2.max()*1.35)

axes[0,2].hist(df["age"].dropna(), bins=10, color=GREEN, edgecolor="white")
axes[0,2].axvline(df["age"].mean(), color=RED, lw=1.5, linestyle="--",
                  label=f"Mean={df['age'].mean():.1f} yr")
axes[0,2].set_xlabel("Age (years)"); axes[0,2].legend(fontsize=8)

axes[1,0].hist(df["area"].dropna(), bins=10, color=RED, edgecolor="white")
axes[1,0].axvline(df["area"].mean(), color=NAVY, lw=1.5, linestyle="--",
                  label=f"Mean={df['area'].mean():.0f} m2")
axes[1,0].set_xlabel("Floor area (m2)"); axes[1,0].legend(fontsize=8)

axes[1,1].hist(df["occ_years"].dropna(), bins=10, color=PURPLE, edgecolor="white")
axes[1,1].axvline(df["occ_years"].mean(), color=RED, lw=1.5, linestyle="--",
                  label=f"Mean={df['occ_years'].mean():.1f} yr")
axes[1,1].set_xlabel("Years of occupation"); axes[1,1].legend(fontsize=8)

vc3 = df["floor"].value_counts().sort_index()
axes[1,2].bar(vc3.index.astype(str), vc3.values, color=STEEL, edgecolor="white")
axes[1,2].set_xlabel("Floor level")

save("fig01_respondent_profile")


# ── CELL 5: fig02 — Architecture + envelope ──────────────────
cs = {"Good":GREEN,"Medium":AMBER,"Degraded":RED}
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

vc = df["glass_type"].apply(
    lambda x:"Double" if "Double" in str(x) and "Single" not in str(x) else "Single"
).value_counts()
axes[0,0].pie(vc.values, labels=vc.index, autopct="%1.1f%%",
              colors=[PURPLE, STEEL], startangle=90)

for ax, col in [(axes[0,1],"shutter_state"),(axes[0,2],"roof_state")]:
    vc2 = df[col].value_counts()
    ax.bar(vc2.index, vc2.values,
           color=[cs.get(k,GREY) for k in vc2.index], edgecolor="white")
    for i,(k,v) in enumerate(vc2.items()):
        ax.text(i, v+0.5, f"{100*v/125:.1f}%", ha="center", fontsize=9)

path_l = ["Air\ninfiltration","Water\ninfiltration","Moisture\n& mold","Cold\ndrafts"]
path_p = [df["air_infiltr_b"].mean()*100, df["water_infiltr_b"].mean()*100,
          df["moisture_b"].mean()*100, df["cold_drafts_b"].mean()*100]
bars = axes[1,0].bar(path_l, path_p,
                     color=[RED, STEEL, GREEN, AMBER], edgecolor="white")
for bar,pct in zip(bars,path_p):
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                   f"{pct:.1f}%", ha="center", fontsize=9)
axes[1,0].set_ylim(0,105); axes[1,0].set_ylabel("% of households")

vc4 = df["window_material"].value_counts()
bars2 = axes[1,1].barh(vc4.index, vc4.values, color=PURPLE, edgecolor="white")
for bar,v in zip(bars2,vc4.values):
    axes[1,1].text(v+0.3, bar.get_y()+bar.get_height()/2,
                   f"{100*v/125:.1f}%", va="center", fontsize=8)

vc5 = df["orientation"].value_counts().head(5)
axes[1,2].barh(vc5.index, vc5.values, color=AMBER, edgecolor="white")

save("fig02_architecture_envelope")


# ── CELL 6: fig03 — Comfort + behaviors ──────────────────────
def cronbach_alpha(data):
    k = data.shape[1]
    return (k/(k-1))*(1 - data.var(ddof=1).sum()/data.sum(axis=1).var(ddof=1))

norm = lambda s: (s-s.min())/(s.max()-s.min()+1e-9)
items = df[["comfort_score","sat_summer_n","sat_winter_n","sensation_n"]].dropna()
alpha = cronbach_alpha(items.apply(norm))
print(f"Cronbach alpha (4-item composite) = {alpha:.3f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(df["comfort_score"].dropna(), bins=10, color=PURPLE, edgecolor="white")
axes[0,0].axvline(df["comfort_score"].mean(), color=RED, lw=1.5, linestyle="--",
                  label=f"Mean={df['comfort_score'].mean():.2f}")
axes[0,0].set_xlabel("Thermal comfort score (0-10)"); axes[0,0].legend(fontsize=8)

order = ["-2 dissatisfied","-1 Somewhat dissatisfied","0 Acceptable",
         "+1 Somewhat satisfied","+2 Satisfied"]
x = np.arange(5); w = 0.35
sc = [df["sat_summer"].value_counts().get(o,0) for o in order]
wc = [df["sat_winter"].value_counts().get(o,0) for o in order]
axes[0,1].bar(x-w/2, sc, w, label="Summer", color=RED, edgecolor="white")
axes[0,1].bar(x+w/2, wc, w, label="Winter", color=STEEL, edgecolor="white")
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(["-2","-1","0","+1","+2"])
axes[0,1].legend()

beh_s = {"Fan/AC":"beh_fan_ac","Close shutters":"beh_close_shutters",
          "Open windows":"beh_open_windows","Light clothes":"beh_light_clothes"}
beh_s = {k:v for k,v in beh_s.items() if v in df.columns}
pcts_s = [df[col].mean()*100 for col in beh_s.values()]
bars3 = axes[1,0].barh(list(beh_s.keys()), pcts_s, color=RED, edgecolor="white")
for bar,p in zip(bars3,pcts_s):
    axes[1,0].text(p+0.5, bar.get_y()+bar.get_height()/2, f"{p:.1f}%", va="center", fontsize=9)
axes[1,0].set_xlim(0,110)

beh_w = {"Turn on heating":"beh_heating","Warmer clothes":"beh_warm_clothes",
          "Close windows":"beh_close_windows"}
pcts_w = [df[col].mean()*100 for col in beh_w.values()]
bars4 = axes[1,1].barh(list(beh_w.keys()), pcts_w, color=STEEL, edgecolor="white")
for bar,p in zip(bars4,pcts_w):
    axes[1,1].text(p+0.5, bar.get_y()+bar.get_height()/2, f"{p:.1f}%", va="center", fontsize=9)
axes[1,1].set_xlim(0,110)

save("fig03_thermal_comfort_behaviors")


# ── CELL 7: MCA computation ──────────────────────────────────
acm_in = pd.DataFrame({
    "Sensation"    : df["thermal_sensation"],
    "Sat_Summer"   : df["sat_summer"],
    "Sat_Winter"   : df["sat_winter"],
    "Fan_AC"       : df["beh_fan_ac"].map({1:"Fan_Yes",0:"Fan_No"}),
    "Shutters"     : df["beh_close_shutters"].map({1:"Shut_Yes",0:"Shut_No"}),
    "WinOpen"      : df["beh_open_windows"].map({1:"WinO_Yes",0:"WinO_No"}),
    "Heating"      : df["beh_heating"].map({1:"Heat_Yes",0:"Heat_No"}),
    "WarmClothes"  : df["beh_warm_clothes"].map({1:"WC_Yes",0:"WC_No"}),
    "WinClose"     : df["beh_close_windows"].map({1:"WinC_Yes",0:"WinC_No"}),
    "Air_Infiltr"  : df["air_infiltr"],
    "Water_Infiltr": df["water_infiltr"],
    "Moisture"     : df["moisture"],
    "ColdDrafts"   : df["cold_drafts"],
    "Glazing"      : df["glass_type"].apply(
        lambda x:"Double" if "Double" in str(x) and "Single" not in str(x) else "Single"),
    "Shutter_State": df["shutter_state"],
    "Roof_State"   : df["roof_state"],
}, index=df.index).dropna()

N_MCA, Q = acm_in.shape
Z    = pd.get_dummies(acm_in, drop_first=False).astype(float)
K    = Z.shape[1]
Za   = Z.values; nt = Za.sum(); rm = Za.sum(axis=1); cm = Za.sum(axis=0)
P    = Za/nt; rc = np.outer(rm/nt, cm/nt)
S    = (P-rc)/(np.sqrt(np.outer(rm/nt, cm/nt))+1e-12)
U, sigma, Vt = svd(S, full_matrices=False)
eig  = sigma**2; IT = (K-Q)/Q
ep   = eig/IT*100; cp = np.cumsum(ep)
F    = U[:,:8]*sigma[:8]
G    = Vt[:8,:].T*sigma[:8]
mods = Z.columns.tolist()
Xc   = F[:,:5]

print(f"MCA: N={N_MCA}, Q={Q}, K={K}")
print(f"Dim1={ep[0]:.1f}%  Dim2={ep[1]:.1f}%")


# ── CELL 8: fig04 — MCA Scree plot ───────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax2 = ax.twinx()
ax.bar(range(1,11), ep[:10], color=PURPLE, edgecolor="white", alpha=0.85)
ax2.plot(range(1,11), cp[:10], "o-", color=RED, lw=2, ms=6)
ax2.axhline(60, color=GREY, lw=0.8, linestyle="--", alpha=0.6)
ax.set_xlabel("MCA Dimension"); ax.set_ylabel("Explained inertia (%)")
ax2.set_ylabel("Cumulative inertia (%)")
ax.set_xticks(range(1,11))
save("fig04_mca_scree_plot")


# ── CELL 9: fig05 — MCA Variable map ─────────────────────────
gc = {"Sensation":RED,"Sat_Summer":RED,"Sat_Winter":RED,
      "Fan_AC":GREEN,"Shutters":GREEN,"WinOpen":GREEN,
      "Heating":GREEN,"WarmClothes":GREEN,"WinClose":GREEN,
      "Air_Infiltr":STEEL,"Water_Infiltr":STEEL,
      "Moisture":STEEL,"ColdDrafts":STEEL,
      "Glazing":GREY,"Shutter_State":GREY,"Roof_State":GREY}
lp = [mpatches.Patch(color=RED, label="Thermal comfort/sensation"),
      mpatches.Patch(color=GREEN, label="Adaptive behaviors"),
      mpatches.Patch(color=STEEL, label="Envelope pathologies"),
      mpatches.Patch(color=GREY, label="Building architecture")]

fig, ax = plt.subplots(figsize=(13, 9))
ax.axhline(0, color=GREY, lw=0.5, alpha=0.4)
ax.axvline(0, color=GREY, lw=0.5, alpha=0.4)
for mod,d1,d2 in zip(mods, G[:,0], G[:,1]):
    grp = next((g for g in gc if mod.startswith(g)), "Glazing")
    col = gc.get(grp,GREY)
    ax.scatter(d1, d2, color=col, s=55, zorder=3, alpha=0.85)
    if abs(d1)>0.05 or abs(d2)>0.05:
        ax.annotate(mod.split("_")[-1] if "_" in mod else mod,
                    (d1,d2), fontsize=7.5, color=col,
                    xytext=(4,3), textcoords="offset points")
ax.legend(handles=lp, fontsize=9, title="Variable group")
ax.set_xlabel(f"Dimension 1 ({ep[0]:.1f}% of inertia)")
ax.set_ylabel(f"Dimension 2 ({ep[1]:.1f}% of inertia)")
save("fig05_mca_variable_map")


# ── CELL 10: fig06 — MCA Individual map ──────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
ax.axhline(0, color=GREY, lw=0.4, alpha=0.4)
ax.axvline(0, color=GREY, lw=0.4, alpha=0.4)
ax.scatter(F[:,0], F[:,1], color=PURPLE, s=35, alpha=0.5,
           edgecolors="white", lw=0.3, label=f"N={N_MCA} households")
ax.set_xlabel(f"Dimension 1 ({ep[0]:.1f}%)")
ax.set_ylabel(f"Dimension 2 ({ep[1]:.1f}%)")
ax.legend(fontsize=9)
save("fig06_mca_individual_map")


# ── CELL 11: fig07 — Cluster validation indices ──────────────
Zlink = linkage(Xc, method="ward")
ks = list(range(2,7))
sils, dbs, chs, aris = [], [], [], []
for k in ks:
    c = fcluster(Zlink, t=k, criterion="maxclust")-1
    m = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(Xc)
    sils.append(silhouette_score(Xc,c)); dbs.append(davies_bouldin_score(Xc,c))
    chs.append(calinski_harabasz_score(Xc,c)); aris.append(adjusted_rand_score(c,m))

print("k | Silhouette | Davies-B | Calinski | ARI")
for i,k in enumerate(ks):
    print(f"{k} | {sils[i]:.4f} | {dbs[i]:.4f} | {chs[i]:.2f} | {aris[i]:.4f}"
          + (" <- selected" if k==4 else ""))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax,(vals,col) in zip(axes.flat, [
    (sils,STEEL), (dbs,RED), (chs,GREEN), (aris,PURPLE),
]):
    ax.plot(ks, vals, "o-", color=col, lw=2, ms=8)
    ax.axvline(4, color=GREY, lw=1.5, linestyle="--", label="k=4 selected")
    ax.set_xlabel("Number of clusters k"); ax.set_xticks(ks)
    ax.legend(fontsize=8)
    idx4 = ks.index(4)
    ax.annotate(f"{vals[idx4]:.3f}", xy=(4, vals[idx4]),
                xytext=(4.2, vals[idx4]), fontsize=9, color=col)
save("fig07_cluster_validation")


# ── CELL 12: CAH k=4 + reference dataset ─────────────────────
K4      = 4
cah_lab = fcluster(Zlink, t=K4, criterion="maxclust")-1
km_lab  = KMeans(n_clusters=K4, random_state=42, n_init=20).fit_predict(Xc)

ref = acm_in.copy()
ref["cluster"] = cah_lab
for col in ["comfort_score","sat_summer_n","sat_winter_n","sensation_n",
            "air_infiltr_b","water_infiltr_b","moisture_b","cold_drafts_b",
            "single_glaz_b","shutter_state_n","roof_state_n",
            "beh_fan_ac","beh_close_shutters","beh_open_windows",
            "beh_heating","beh_warm_clothes","beh_close_windows",
            "area","floor","shutter_state","roof_state","glass_type",
            "air_infiltr","water_infiltr","moisture","cold_drafts",
            "thermal_sensation","sat_summer","sat_winter"]:
    if col in df.columns:
        ref[col] = df.loc[acm_in.index, col].values

N_REF = len(ref)
ns = [(ref["cluster"]==cl).sum() for cl in range(K4)]
print(f"Reference dataset: N={N_REF}  |  ARI(CAH/KM)={adjusted_rand_score(cah_lab,km_lab):.4f}")
for cl in range(K4):
    c = ref[ref["cluster"]==cl]["comfort_score"].mean()
    print(f"  C{cl}: n={ns[cl]} ({100*ns[cl]/N_REF:.1f}%)  comfort={c:.2f}")

print("\nKruskal-Wallis (omnibus — no correction needed):")
for lbl,col in [("Comfort","comfort_score"),
                 ("Sat.Summer","sat_summer_n"),("Sat.Winter","sat_winter_n")]:
    grps = [ref[ref["cluster"]==cl][col].dropna().values for cl in range(K4)]
    H,p  = kruskal(*grps)
    print(f"  {lbl}: H={H:.3f}, p={p:.4f}")


# ── CELL 13: Post-hoc Bonferroni (Family F1) ─────────────────
pairs   = list(combinations(range(K4), 2))
n_pairs = len(pairs)
print(f"Post-hoc pairwise (Bonferroni alpha/{n_pairs}={0.05/n_pairs:.4f}):")
for lbl,col in [("Comfort score","comfort_score"),
                 ("Summer satisfaction","sat_summer_n"),
                 ("Winter satisfaction","sat_winter_n")]:
    p_raws, rows = [], []
    for p0,p1 in pairs:
        g0 = ref[ref["cluster"]==p0][col].dropna()
        g1 = ref[ref["cluster"]==p1][col].dropna()
        U,p = mannwhitneyu(g0, g1, alternative="two-sided")
        p_raws.append(p); rows.append((p0,p1,U,g0.mean(),g1.mean()))
    _, p_bonf, _, _ = multipletests(p_raws, method="bonferroni")
    print(f"\n  {lbl}:")
    for (p0,p1,U,m0,m1),p_r,pb in zip(rows,p_raws,p_bonf):
        sig = "***" if pb<0.001 else "**" if pb<0.01 else "*" if pb<0.05 else "ns"
        print(f"  C{p0} vs C{p1}  p_raw={p_r:.4f}  p_Bonf={pb:.4f}  {sig}")


# ── CELL 14: fig08 — Individual map by cluster ───────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax,labs in [(axes[0], cah_lab), (axes[1], km_lab)]:
    ax.axhline(0, color=GREY, lw=0.4, alpha=0.4)
    ax.axvline(0, color=GREY, lw=0.4, alpha=0.4)
    for cl in range(K4):
        mask = labs==cl
        ax.scatter(F[mask,0], F[mask,1], color=PAL[cl], s=50, alpha=0.75,
                   edgecolors="white", lw=0.4, label=f"C{cl} (n={mask.sum()})")
        ax.scatter(F[mask,0].mean(), F[mask,1].mean(),
                   color=PAL[cl], marker="*", s=280,
                   zorder=6, edgecolors="black", lw=0.5)
    ax.set_xlabel(f"Dim 1 ({ep[0]:.1f}%)")
    ax.set_ylabel(f"Dim 2 ({ep[1]:.1f}%)")
    ax.legend(fontsize=8)
save("fig08_individual_map_clusters")


# ── CELL 15: fig09 — Cluster profiles ────────────────────────
cl_lbl  = [f"C{i}\n(n={ns[i]})" for i in range(K4)]
colors4 = [PAL[i] for i in range(K4)]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

data_vio = [ref[ref["cluster"]==cl]["comfort_score"].dropna().values for cl in range(K4)]
vp = axes[0,0].violinplot(data_vio, positions=range(K4), showmedians=True)
for body,col in zip(vp["bodies"],colors4):
    body.set_facecolor(col); body.set_alpha(0.65)
vp["cmedians"].set_color("white"); vp["cmedians"].set_lw(2.5)
np.random.seed(42)
for i,(data,col) in enumerate(zip(data_vio,colors4)):
    j = np.random.uniform(-0.07,0.07,len(data))
    axes[0,0].scatter(i+j, data, color=col, s=15, alpha=0.4, zorder=3)
    axes[0,0].scatter(i, np.mean(data), color="white", marker="D",
                      s=60, zorder=5, edgecolors=col, lw=1.5)
axes[0,0].set_xticks(range(K4)); axes[0,0].set_xticklabels(cl_lbl, fontsize=8)
axes[0,0].set_ylim(-0.5,11.5); axes[0,0].set_ylabel("Comfort score (0-10)")

x = np.arange(K4); w = 0.35
axes[0,1].bar(x-w/2, [ref[ref["cluster"]==cl]["sat_summer_n"].mean() for cl in range(K4)],
              w, label="Summer", color=RED, edgecolor="white", alpha=0.85)
axes[0,1].bar(x+w/2, [ref[ref["cluster"]==cl]["sat_winter_n"].mean() for cl in range(K4)],
              w, label="Winter", color=STEEL, edgecolor="white", alpha=0.85)
axes[0,1].axhline(0, color="black", lw=0.8, linestyle="--")
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(cl_lbl, fontsize=8)
axes[0,1].set_ylabel("Mean satisfaction (-2 to +2)"); axes[0,1].legend()

pc = ["air_infiltr_b","water_infiltr_b","moisture_b","cold_drafts_b"]
pl = ["Air\ninfilt.","Water\ninfilt.","Moisture","Cold\ndrafts"]
x2 = np.arange(len(pc))
for i,col in enumerate(colors4):
    axes[1,0].bar(x2+i*0.2, [ref[ref["cluster"]==i][c].mean()*100 for c in pc],
                  0.2, color=col, edgecolor="white", alpha=0.85, label=f"C{i}")
axes[1,0].set_xticks(x2+0.3); axes[1,0].set_xticklabels(pl, fontsize=8)
axes[1,0].set_ylim(0,115); axes[1,0].set_ylabel("% of households"); axes[1,0].legend(fontsize=8)

bc = ["beh_fan_ac","beh_close_shutters","beh_heating","beh_warm_clothes","beh_close_windows"]
bl = ["Fan/\nAC","Close\nshutters","Heating","Warm\nclothes","Close\nwindows"]
x3 = np.arange(len(bc))
for i,col in enumerate(colors4):
    axes[1,1].bar(x3+i*0.18, [ref[ref["cluster"]==i][c].mean()*100 for c in bc],
                  0.18, color=col, edgecolor="white", alpha=0.85, label=f"C{i}")
axes[1,1].set_xticks(x3+0.27); axes[1,1].set_xticklabels(bl, fontsize=8)
axes[1,1].set_ylim(0,115); axes[1,1].set_ylabel("% of households"); axes[1,1].legend(fontsize=8)
save("fig09_cluster_profiles")


# ── CELL 16: fig10 — Bootstrap stability ─────────────────────
def jaccard(a, b):
    a,b = set(a),set(b)
    return len(a&b)/len(a|b) if a|b else 0

N_BOOT = 200; np.random.seed(42)
orig_s  = [set(np.where(cah_lab==cl)[0]) for cl in range(K4)]
jac_all = {cl:[] for cl in range(K4)}; ari_b = []
for _ in range(N_BOOT):
    idx   = np.random.choice(len(Xc), len(Xc), replace=True)
    lab_b = fcluster(linkage(Xc[idx], method="ward"), t=K4, criterion="maxclust")-1
    for cl in range(K4):
        bsets = [set(idx[lab_b==cb]) for cb in range(K4)]
        jac_all[cl].append(max(jaccard(orig_s[cl],bs) for bs in bsets))
    ari_b.append(adjusted_rand_score(cah_lab, lab_b))

jac_means = [np.mean(jac_all[cl]) for cl in range(K4)]
print(f"Bootstrap Jaccard = {[round(j,3) for j in jac_means]}")
print(f"Bootstrap ARI = {np.mean(ari_b):.3f} +/- {np.std(ari_b):.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bp = axes[0].boxplot([jac_all[cl] for cl in range(K4)], patch_artist=True, widths=0.5,
                     medianprops=dict(color="white", lw=2.5))
for patch,col in zip(bp["boxes"],[PAL[i] for i in range(K4)]):
    patch.set_facecolor(col); patch.set_alpha(0.75)
axes[0].axhline(0.75, color="black", lw=1.5, linestyle="--", label="Stability threshold (0.75)")
axes[0].axhline(0.60, color=GREY, lw=1.0, linestyle=":", label="Moderate threshold")
axes[0].set_xticklabels([f"C{cl}\nn={ns[cl]}" for cl in range(K4)], fontsize=8)
axes[0].set_ylabel("Jaccard similarity"); axes[0].set_ylim(0,1.05); axes[0].legend(fontsize=8)

axes[1].hist(ari_b, bins=20, color=PURPLE, edgecolor="white", alpha=0.85)
axes[1].axvline(np.mean(ari_b), color=RED, lw=2, linestyle="--",
                label=f"Mean ARI={np.mean(ari_b):.3f}")
axes[1].set_xlabel("ARI (bootstrap vs original)"); axes[1].set_ylabel("Frequency")
axes[1].legend(fontsize=9)
save("fig10_bootstrap_stability")


# ── CELL 17: fig11 — Cramér's V (FDR) ────────────────────────
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2,p,dof,_ = chi2_contingency(ct)
    n=ct.sum().sum(); r,k=ct.shape
    return np.sqrt(chi2/(n*(min(r,k)-1))), p

def cohen_lbl(v):
    return "negl." if v<0.10 else "small" if v<0.30 else "medium" if v<0.50 else "large"

arch_v = {"Glazing type":"Glazing","Shutter state":"shutter_state",
           "Roof state":"roof_state","Air infiltration":"air_infiltr",
           "Water infiltration":"water_infiltr","Moisture/mold":"moisture",
           "Cold drafts":"cold_drafts"}
tgt_v  = {"Thermal sensation":"thermal_sensation","Summer satisfaction":"sat_summer",
           "Winter satisfaction":"sat_winter","Cluster":"cluster_label"}

ref["cluster_label"] = ref["cluster"].astype(str)

rows_cv = []
for a_l,a_c in arch_v.items():
    for t_l,t_c in tgt_v.items():
        tmp = ref[[a_c,t_c]].dropna()
        if len(tmp)>10 and a_c in ref.columns and t_c in ref.columns:
            v,p = cramers_v(tmp[a_c], tmp[t_c])
            rows_cv.append({"Arch":a_l,"Target":t_l,"V":round(v,3),"p_raw":round(p,4)})

cv_df = pd.DataFrame(rows_cv)
_,p_fdr,_,_ = multipletests(cv_df["p_raw"].values, method="fdr_bh")
cv_df["p_FDR"] = p_fdr.round(4)
cv_df["sig"]   = cv_df["p_FDR"].apply(lambda p:"***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
cv_df["effect"] = cv_df["V"].apply(cohen_lbl)

for _,row in cv_df.iterrows():
    print(f"{row['Arch']:<22} x {row['Target']:<22} V={row['V']:.3f} p_FDR={row['p_FDR']:.4f} {row['sig']}")

cv_pivot = cv_df.pivot(index="Arch",columns="Target",values="V")
cv_ann   = cv_df.pivot(index="Arch",columns="Target",values="V").astype(object)
sig_piv  = cv_df.pivot(index="Arch",columns="Target",values="sig")
for i in range(cv_ann.shape[0]):
    for j in range(cv_ann.shape[1]):
        v=cv_pivot.iloc[i,j]; s=sig_piv.iloc[i,j]
        cv_ann.iloc[i,j] = f"{v:.2f}\n({cohen_lbl(v)})\n{s}"

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(cv_pivot.astype(float), annot=cv_ann, fmt="",
            cmap=cmap_journal, vmin=0, vmax=0.5, linewidths=0.5,
            annot_kws={"size":8}, ax=ax,
            cbar_kws={"label":"Cramér's V","shrink":0.8})
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xlabel(""); ax.set_ylabel("")
save("fig11_cramers_v")


# ── CELL 18: fig12 — Spearman correlations (FDR) ─────────────
corr_v = {
    "Summer satisfaction":"sat_summer_n","Winter satisfaction":"sat_winter_n",
    "Thermal sensation":"sensation_n","Air infiltration":"air_infiltr_b",
    "Water infiltration":"water_infiltr_b","Moisture/mold":"moisture_b",
    "Cold drafts":"cold_drafts_b","Single glazing":"single_glaz_b",
    "Shutter state":"shutter_state_n","Roof state":"roof_state_n",
    "Floor area":"area","Floor level":"floor",
}
rows_sp = []
for lbl,col in corr_v.items():
    tmp = ref[["comfort_score",col]].dropna()
    r,p = spearmanr(tmp["comfort_score"], tmp[col])
    rows_sp.append({"Variable":lbl,"r":round(r,3),"p_raw":round(p,4)})

sp_df = pd.DataFrame(rows_sp)
_,p_fdr3,_,_ = multipletests(sp_df["p_raw"].values, method="fdr_bh")
sp_df["p_FDR"] = p_fdr3.round(4)
sp_df["sig"]   = sp_df["p_FDR"].apply(lambda p:"***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns")
sp_df = sp_df.sort_values("r", key=abs, ascending=False)

for _,row in sp_df.iterrows():
    print(f"  {row['Variable']:<25} r={row['r']:>6.3f}  p_FDR={row['p_FDR']:.4f}  {row['sig']}")

fig, ax = plt.subplots(figsize=(9, 7))
bar_colors = [RED if r<0 else GREEN for r in sp_df["r"]]
bars = ax.barh(sp_df["Variable"], sp_df["r"], color=bar_colors, edgecolor="white", alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
for bar,row in zip(bars, sp_df.itertuples()):
    val=row.r; sig=row.sig
    txt = f"{val:.2f} {sig}" if sig!="ns" else f"{val:.2f}"
    xp  = val+(0.01 if val>=0 else -0.01)
    ax.text(xp, bar.get_y()+bar.get_height()/2, txt,
            va="center", ha="left" if val>=0 else "right", fontsize=8)
ax.set_xlabel("Spearman r"); ax.set_xlim(-0.8, 0.8)
save("fig12_spearman_correlations")


# ── CELL 19: LDA + Ordinal logit (computation) ───────────────
lda_c = {"Air infiltration":"air_infiltr_b","Water infiltration":"water_infiltr_b",
          "Moisture/mold":"moisture_b","Cold drafts":"cold_drafts_b",
          "Single glazing":"single_glaz_b","Shutter state":"shutter_state_n",
          "Roof state":"roof_state_n","Fan/AC (summer)":"beh_fan_ac",
          "Heating (winter)":"beh_heating","Floor area":"area","Floor level":"floor"}
lda_df = ref[list(lda_c.values())+["cluster"]].dropna()
X_lda  = lda_df[list(lda_c.values())].values
y_lda  = lda_df["cluster"].values

lda    = LinearDiscriminantAnalysis()
cv     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_acc = cross_val_score(lda, X_lda, y_lda, cv=cv, scoring="accuracy")
y_pred = cross_val_predict(lda, X_lda, y_lda, cv=cv)
lda.fit(X_lda, y_lda)
print(f"LDA Accuracy CV-10 = {cv_acc.mean():.3f} +/- {cv_acc.std():.3f}")

imp_lda = pd.DataFrame({"Feature":list(lda_c.keys()),
                         "Importance":np.sqrt(np.sum(lda.coef_**2, axis=0))}
                        ).sort_values("Importance", ascending=True)

pred_c = ["air_infiltr_b","cold_drafts_b","single_glaz_b",
          "shutter_state_n","roof_state_n","beh_fan_ac","beh_heating","area"]
pred_l = {"air_infiltr_b":"Air infiltration","cold_drafts_b":"Cold drafts",
          "single_glaz_b":"Single glazing","shutter_state_n":"Shutter state",
          "roof_state_n":"Roof state","beh_fan_ac":"Fan/AC (summer)",
          "beh_heating":"Heating (winter)","area":"Floor area"}

logit_r = {}
for season,dep in [("Summer","sat_summer_n"),("Winter","sat_winter_n")]:
    reg = ref[pred_c+[dep]].dropna().copy()
    cats = sorted(reg[dep].dropna().unique().astype(int))
    reg[dep+"_cat"] = pd.Categorical(reg[dep].astype(int), categories=cats, ordered=True)
    try:
        model  = OrderedModel(reg[dep+"_cat"], reg[pred_c].astype(float), distr="logit")
        result = model.fit(method="bfgs", disp=False)
        logit_r[season] = result
        pr2 = 1-result.llf/result.llnull
        print(f"\n{season} (N={len(reg)}, pseudo-R2={pr2:.3f})")
        params = result.params[pred_c]; pvals = result.pvalues[pred_c]
        conf   = result.conf_int().loc[pred_c]
        for v in pred_c:
            or_v=np.exp(params[v]); lo=np.exp(conf.loc[v,0]); hi=np.exp(conf.loc[v,1])
            p=pvals[v]
            sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "ns"
            print(f"  {pred_l.get(v,v):<22} OR={or_v:.2f} [{lo:.2f}-{hi:.2f}] p={p:.4f} {sig}")
    except Exception as e:
        print(f"  Error {season}: {e}")


# ── CELL 20: fig13 — LDA figure ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

X_proj = lda.transform(X_lda)
for cl in range(K4):
    mask = y_lda==cl
    axes[0].scatter(X_proj[mask,0], X_proj[mask,1], color=PAL[cl], s=50,
                    alpha=0.75, edgecolors="white", lw=0.4, label=f"C{cl} (n={mask.sum()})")
    axes[0].scatter(X_proj[mask,0].mean(), X_proj[mask,1].mean(),
                    color=PAL[cl], marker="*", s=250, zorder=6, edgecolors="black", lw=0.5)
axes[0].set_xlabel("LD1"); axes[0].set_ylabel("LD2"); axes[0].legend(fontsize=7)

bars = axes[1].barh(imp_lda["Feature"], imp_lda["Importance"],
                    color=PURPLE, edgecolor="white", alpha=0.85)
for bar,v in zip(bars, imp_lda["Importance"]):
    axes[1].text(v+0.003, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8)
axes[1].set_xlabel("Discriminant importance")

cm = confusion_matrix(y_lda, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap=cmap_blue, ax=axes[2],
            xticklabels=[f"C{i}" for i in range(K4)],
            yticklabels=[f"C{i}" for i in range(K4)],
            annot_kws={"size":11})
axes[2].set_xlabel("Predicted (CV-10)"); axes[2].set_ylabel("Actual")
save("fig13_LDA")


# ── CELL 21: fig14 — Odds ratios ─────────────────────────────
if len(logit_r) == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax,season,col in [(axes[0],"Summer",RED),(axes[1],"Winter",STEEL)]:
        result = logit_r[season]
        params = result.params[pred_c]; pvals = result.pvalues[pred_c]
        conf   = result.conf_int().loc[pred_c]
        or_v   = np.exp(params.values); lo_v = np.exp(conf.iloc[:,0].values)
        hi_v   = np.exp(conf.iloc[:,1].values)
        lbls   = [pred_l.get(v,v) for v in pred_c]; pv = pvals.values
        idx    = np.argsort(or_v)
        or_s,lo_s,hi_s = or_v[idx],lo_v[idx],hi_v[idx]
        lb_s = [lbls[i] for i in idx]; pv_s = pv[idx]
        bar_c = [col if p<0.05 else GREY for p in pv_s]
        ax.barh(range(len(lb_s)), or_s-1, left=1, color=bar_c,
                edgecolor="white", alpha=0.85, height=0.6)
        ax.errorbar(or_s, range(len(lb_s)), xerr=[or_s-lo_s, hi_s-or_s],
                    fmt="none", color="black", capsize=4, lw=1.2)
        ax.axvline(1, color="black", lw=1.2, linestyle="--")
        ax.set_yticks(range(len(lb_s))); ax.set_yticklabels(lb_s, fontsize=8)
        ax.set_xlabel("Odds Ratio")
        for i, (ov,p) in enumerate(zip(or_s,pv_s)):
            sig="***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else ""
            if sig:
                ax.text(max(hi_s)+0.05, i, sig, va="center", fontsize=9)
    save("fig14_odds_ratios")


# ── CELL 22: Multiple comparison summary ─────────────────────
print("""
Family  Test                     N tests  Correction       alpha
F0      Kruskal-Wallis (omnibus) 1/var    NONE             0.05
F1      Post-hoc MWU (k=4)      6        Bonferroni       0.008
F2      Cramer's V               28       FDR (BH)         variable
F3      Spearman correlations    12       FDR (BH)         variable
F4      Ordinal logit             1 model  NONE (global)    0.05/OR
""")


# ── CELL 23: Download all figures ────────────────────────────
from google.colab import files

figures = [
    "fig01_respondent_profile","fig02_architecture_envelope",
    "fig03_thermal_comfort_behaviors","fig04_mca_scree_plot",
    "fig05_mca_variable_map","fig06_mca_individual_map",
    "fig07_cluster_validation","fig08_individual_map_clusters",
    "fig09_cluster_profiles","fig10_bootstrap_stability",
    "fig11_cramers_v","fig12_spearman_correlations",
    "fig13_LDA","fig14_odds_ratios",
]
for name in figures:
    path = f"/content/{name}.png"
    if os.path.exists(path):
        files.download(path)
        print(f"downloaded: {name}.png")
    else:
        print(f"NOT FOUND: {name}.png")
