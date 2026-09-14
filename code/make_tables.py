"""Emit paper/tables.tex (booktabs tables) and paper/numbers.tex (\\newcommand
macros) directly from results/summary_v2.json, so every number in the paper is
traceable to raw data and cannot drift from it."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from common import RESULTS

PAPER = os.path.join(common.PROJ, "paper")
SHORT = {"empty-32-32": r"\texttt{empty-32-32}",
         "random-32-32-20": r"\texttt{random-32-32-20}",
         "room-32-32-4": r"\texttt{room-32-32-4}",
         "maze-32-32-4": r"\texttt{maze-32-32-4}",
         "warehouse-10-20-10-2-1": r"\texttt{warehouse}"}
ORDER = ["empty-32-32", "random-32-32-20", "room-32-32-4", "maze-32-32-4",
         "warehouse-10-20-10-2-1"]
METHS = ["unweighted", "lanes", "tolls", "tolls-online"]
MACROS = []


def mac(name, value):
    MACROS.append((name, value))


def cell(c, bold=False):
    if not c:
        return "---"
    s = f"{c['mean']:.2f} $\\pm$ {c['std']:.2f}"
    return f"\\textbf{{{s}}}" if bold else s


def headline_table(s):
    rows = {(r["map"], r["N"]): r for r in s["headline"]}
    tests = s.get("headline_tests", [])
    tmap = {(t["map"], t["N"], t["cmp"]): t for t in tests}
    out = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
           r"\caption{\textsc{Finite-horizon throughput} (tasks/step, mean $\pm$ population std, "
           r"10 seeds $\times$ 1{,}000 steps). Bold: best in row; $\dagger$: above "
           r"both baselines at family-wise $p<0.05$.}",
           r"\label{tab:headline}", r"\begin{tabular}{llcccccc}", r"\toprule",
           r"Map & $N$ (density) & Unweighted & Lanes & Tolls (ours) & "
           r"Adapt. (sync.) & solve (s) & $\lambda$ \\", r"\midrule"]
    nsig = 0
    nsig_online = 0
    for mp_ in ORDER:
        first = True
        for (m2, N), r in sorted(rows.items(), key=lambda kv: kv[0][1]):
            if m2 != mp_:
                continue
            vals = {k: r.get(k, {}).get("mean", -1) for k in METHS}
            best = max(vals, key=vals.get)
            def beats_both(meth):
                return all(
                    tmap.get((mp_, N, f"{meth}-vs-{b}"), {}).get("significant")
                    and meth in r and b in r
                    and r[meth]["mean"] > r[b]["mean"]
                    for b in ("unweighted", "lanes"))
            sig, sig_o = beats_both("tolls"), beats_both("tolls-online")
            nsig += bool(sig)
            nsig_online += bool(sig_o)
            cells = [cell(r.get(k), k == best) for k in METHS]
            if sig:
                cells[2] += r"$^\dagger$"
            if sig_o:
                cells[3] += r"$^\dagger$"
            name = SHORT.get(mp_, mp_) if first else ""
            def num(key, fmt="%.1f"):
                v = r.get(key)
                return "---" if v is None or v != v else fmt % v
            out.append(f"{name} & {N} ({r['density']:.0f}\\%) & "
                       + " & ".join(cells)
                       + f" & {num('solve_s')} & {num('lam_pred')} \\\\")
            first = False
        if mp_ != ORDER[-1]:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    mac("NumSigCells", str(nsig))
    mac("NumSigCellsOnline", str(nsig_online))
    # LaTeX control sequences are letters only, so cells are named
    # <map><Lo|Mid|Hi> by density rank rather than by agent count.
    LEVEL = ["Lo", "Mid", "Hi"]
    best_ratio = ("", 0.0)
    for mp_ in ORDER:
        cells = sorted([kv for kv in rows.items() if kv[0][0] == mp_],
                       key=lambda kv: kv[0][1])
        for lvl, ((_, N), r) in zip(LEVEL, cells):
            if "tolls" not in r or "unweighted" not in r:
                continue
            tag = mp_.split("-")[0].capitalize() + lvl
            ratio = r["tolls"]["mean"] / r["unweighted"]["mean"]
            mac(f"R{tag}", f"{ratio:.2f}")
            mac(f"N{tag}", str(N))
            for pre, key in (("T", "tolls"), ("U", "unweighted"),
                             ("L", "lanes"), ("O", "tolls-online")):
                if key in r:
                    mac(f"{pre}{tag}", f"{r[key]['mean']:.2f}")
            if ratio > best_ratio[1]:
                best_ratio = (tag, ratio)
    mac("BestRatio", f"{best_ratio[1]:.2f}")
    mac("BestRatioMap", best_ratio[0])

    # Win counts, generated rather than hand-written, so the prose cannot drift
    # from the table.
    def wins(meth):
        n = 0
        for r in rows.values():
            if meth in r and "unweighted" in r and r[meth]["mean"] > r["unweighted"]["mean"]:
                n += 1
        return n
    ncells = sum(1 for r in rows.values() if "tolls" in r and "unweighted" in r)
    mac("NumCells", str(ncells))
    mac("NumTollsWin", str(wins("tolls")))
    mac("NumOnlineWin", str(wins("tolls-online")))
    mac("NumLanesWin", str(wins("lanes")))
    # map families where tolls beat unweighted in a majority of their cells
    fams = 0
    famlist = []
    for mp_ in ORDER:
        cs = [r for (m2, _), r in rows.items() if m2 == mp_ and "tolls" in r]
        if not cs:
            continue
        w = sum(1 for r in cs if r["tolls"]["mean"] > r["unweighted"]["mean"])
        if w * 2 > len(cs):
            fams += 1
            famlist.append(mp_)
    mac("NumFamiliesTollsWin", str(fams))
    mac("NumFamilies", str(sum(1 for mp_ in ORDER
                               if any(m2 == mp_ for (m2, _) in rows))))
    return "\n".join(out)


def transfer_table(s):
    """Fleet-size transfer and demand shift as one two-panel table."""
    ac = s.get("transfer_agentcount", {})
    ds = s.get("transfer_demandshift", {})
    if not ac:
        return ""
    Ns = sorted({int(n) for v in ac.values() for n in v})
    hs = sorted({float(h) for v in ds.values() for h in v}) if ds else []
    labs = [("unweighted", "Unweighted"), ("lanes", "Lanes"),
            ("tolls-frozen", "Tolls frozen @$400$"),
            ("tolls-resolved", "Tolls re-solved"),
            ("tolls-online", "Tolls online")]
    dmap = {"unweighted": "unweighted", "lanes": "lanes",
            "tolls-frozen": "tolls", "tolls-resolved": None,
            "tolls-online": "tolls-online"}
    ncol = len(Ns) + len(hs)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{\textsc{Transfer} on \texttt{empty-32-32}, mean of 10 seeds. "
           r"Left: fleet size, weights built at $N{=}400$. Right: task-distribution "
           r"shift at $N{=}250$.}",
           r"\label{tab:transfer}", r"\scriptsize",
           r"\setlength{\tabcolsep}{2pt}",
           r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
           r"& \multicolumn{" + str(len(Ns)) + r"}{c}{fleet size $N$} & "
           r"\multicolumn{" + str(len(hs)) + r"}{c}{demand shift $p$} \\",
           r"\cmidrule(lr){2-" + str(1 + len(Ns)) + r"}"
           r"\cmidrule(lr){" + str(2 + len(Ns)) + "-" + str(1 + ncol) + r"}",
           "Method & " + " & ".join([str(n) for n in Ns]
                                    + [f"{h:g}" for h in hs]) + r" \\",
           r"\midrule"]
    for key, name in labs:
        if key not in ac:
            continue
        cs = []
        for n in Ns:
            v = ac[key].get(str(n)) or ac[key].get(n)
            cs.append(f"{v['mean']:.2f}" if v else "---")
        dk = dmap.get(key)
        for h in hs:
            v = ds.get(dk, {}).get(str(h)) if dk else None
            cs.append(f"{v['mean']:.2f}" if v else "---")
        out.append(f"{name} & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def planner_table(s):
    pt = s.get("planner_transfer", {})
    if not pt:
        return ""
    keys = sorted(pt)
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption{\textsc{Planner transfer}: the same derived weights fed to "
           r"windowed prioritized planning (space-time A$^\star$ over a bounded "
           r"horizon against a reservation table) instead of PIBT, with no "
           r"re-tuning (mean, 5 seeds).}", r"\label{tab:planner}", r"\footnotesize",
           r"\setlength{\tabcolsep}{3pt}",
           r"\begin{tabular}{l" + "c" * len(keys) + "}", r"\toprule",
           "Method & " + " & ".join(k.replace("-32-32", "").replace("_", " ")
                                    .replace("@", "\\,@\\,") for k in keys) + r" \\",
           r"\midrule"]
    for key, name in [("unweighted", "Unweighted"), ("lanes", "Lanes"),
                      ("tolls", "Tolls (ours)"), ("tolls-online", "Tolls-online (ours)")]:
        cs = [f"{pt[k][key]['mean']:.2f}" if key in pt[k] else "---" for k in keys]
        out.append(f"{name} & " + " & ".join(cs) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def numbers(s):
    cfg = s.get("config", {})
    oc = s.get("online_cost", {})
    if oc:
        mac("OnlineSmallCost", oc.get("small_range", "??"))
        mac("OnlineWarehouseCost", oc.get("warehouse_range", "??"))
    mac("GammaVal", f"{cfg.get('gamma', 2.0):g}")
    mac("MuVal", f"{cfg.get('mu', 1.0):g}")
    mac("EpsTie", f"{cfg.get('eps_tie', 1.0):g}")
    fl = s.get("fluid", {})
    if fl:
        mac("FluidR", f"{fl['pearson_r']:.3f}")
        mac("FluidRatio", f"{fl['mean_ratio']:.2f}")
        mac("FluidN", str(fl["n_points"]))
        emp = [p for p in fl["points"] if p["map"] == "empty-32-32"]
        if emp:
            worst = max(abs(1 - p["realised"] / p["predicted"]) for p in emp)
            mac("FluidEmptyErr", f"{100 * worst:.0f}")
    ct = s.get("compute_vs_tput", {})
    if "_eval_rate_s" in ct:
        mac("EvalRate", f"{ct['_eval_rate_s']:.2f}")
    for k, v in ct.items():
        if not isinstance(v, dict):
            continue
        tag = k.replace("-", "").replace("_", "")
        mac(f"C{tag}Tput", f"{v['tput']:.2f}")
        mac(f"C{tag}Std", f"{v['std']:.2f}")
        mac(f"C{tag}Time", f"{v['compute_s']:.0f}")
        if "n_evals" in v:
            mac(f"C{tag}Evals", f"{v['n_evals']:,}".replace(",", "{,}"))
    if "tolls" in ct and "es-strong-cold" in ct and "_eval_rate_s" in ct:
        # search cost at the uncontended rate, relative to one toll solve
        cost = ct["es-strong-cold"]["n_evals"] * ct["_eval_rate_s"]
        mac("ESSpeedup", f"{cost / max(ct['tolls']['compute_s'], 1e-9):.0f}")
        mac("ESEquivSec", f"{cost:.0f}")
        mac("GGOEquivHours", f"{10000 * ct['_eval_rate_s'] / 3600:.1f}")
    return MACROS


def main():
    with open(os.path.join(RESULTS, "summary_v2.json")) as f:
        s = json.load(f)
    parts = [headline_table(s), transfer_table(s)]
    with open(os.path.join(PAPER, "tables.tex"), "w") as f:
        f.write("%% generated by code/make_tables.py -- do not edit\n")
        f.write("\n\n".join(p for p in parts if p) + "\n")
    numbers(s)
    DIG = str.maketrans("0123456789", "abcdefghij")
    seen = {}
    with open(os.path.join(PAPER, "numbers.tex"), "w") as f:
        f.write("%% generated by code/make_tables.py -- do not edit\n")
        for name, val in MACROS:
            name = name.translate(DIG).replace("-", "").replace("_", "")
            if name in seen:
                continue
            seen[name] = val
            f.write(f"\\providecommand{{\\{name}}}{{}}"
                    f"\\renewcommand{{\\{name}}}{{{val}}}\n")
    print(f"wrote paper/tables.tex and paper/numbers.tex ({len(seen)} macros)")
    for k in sorted(seen):
        print(f"  \\{k} = {seen[k]}")


if __name__ == "__main__":
    main()
