#!/usr/bin/env python3
"""SPRT analysis for rc-perf-recert-06us receipts.

H0 (fail):  cv >= CV_FAIL (0.08)   H1 (certify): cv <= CV_PASS (0.05)
Indifference zone between; alpha = beta = 0.05. Sequential test on the running
sample: after each run compute the log-likelihood ratio of observing the
residual sequence under sigma = CV_PASS*mu_hat vs sigma = CV_FAIL*mu_hat
(mu_hat = running mean, plug-in). Cross A=(1-beta)/alpha => CERTIFY;
cross B=beta/(1-alpha) => FAIL-TO-CERTIFY. Exhaust runs => honest verdict from
final cv with the SPRT path recorded.
"""
import json, math, sys

CV_PASS, CV_FAIL, ALPHA, BETA = 0.05, 0.08, 0.05, 0.05
A = math.log((1 - BETA) / ALPHA)
B = math.log(BETA / (1 - ALPHA))

def sprt(values):
    n, s, llr = 0, 0.0, 0.0
    trail = []
    for v in values:
        n += 1; s += v; mu = s / n
        if mu <= 0 or n < 2:
            trail.append(None); continue
        var = sum((x - mu) ** 2 for x in values[:n]) / (n - 1)
        sd = math.sqrt(var)
        # normal log-lik at sd=CV_PASS*mu vs sd=CV_FAIL*mu (mean treated as known plug-in)
        def ll(sigma):
            return -n * math.log(sigma) - sum((x - mu) ** 2 for x in values[:n]) / (2 * sigma ** 2)
        llr = ll(CV_PASS * mu) - ll(CV_FAIL * mu)
        verdict = ("CERTIFY" if llr >= A else "FAIL" if llr <= B else "continue")
        trail.append({"n": n, "cv_pct": sd / mu * 100, "llr": round(llr, 3), "verdict": verdict})
        if verdict in ("CERTIFY", "FAIL"):
            return verdict, trail
    final_cv = (sum((x - s / n) ** 2 for x in values) / (n - 1)) ** 0.5 / (s / n) * 100 if n > 1 else None
    verdict = "CERTIFY" if final_cv is not None and final_cv <= CV_PASS * 100 else "FAIL"
    return verdict + "_EXHAUSTED", trail

receipts = [json.loads(l) for l in open(sys.argv[1]) if '"ttfa_cert"' in l]
out = {}
for cls in ("short", "long"):
    vals = [r["ttfa_audible_ms"] for r in receipts if r["class"] == cls]
    rtfs = [r["rtf"] for r in receipts if r["class"] == cls]
    v_verdict, v_trail = sprt(vals)
    r_verdict, r_trail = sprt(rtfs)
    out[cls] = {
        "ttfa_audible": {"n": len(vals), "mean_ms": round(sum(vals)/len(vals), 2),
                          "verdict": v_verdict, "trail_last": v_trail[-1] if v_trail else None},
        "rtf": {"n": len(rtfs), "mean": round(sum(rtfs)/len(rtfs), 4),
                 "verdict": r_verdict},
    }
print(json.dumps(out, indent=2))
