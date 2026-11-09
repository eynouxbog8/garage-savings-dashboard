"""
Synthetic data generator for the Garage Savings Dashboard.

ALL DATA IS SYNTHETIC. Nothing here comes from a real employer, garage,
vehicle or claim. Every number is drawn from assumptions written below.
Fixed seed => anyone running this gets identical files.

Scenario: a payments team reviews extra charges that garages add to
recovery/repair jobs. Charges that are declined or reduced create a "saving".
"""
import numpy as np
import pandas as pd

SEED = 2026
rng = np.random.default_rng(SEED)

# ---------- ASSUMPTIONS (edit these, re-run, see the dashboard change) ----------
N_GARAGES   = 120
START, END  = "2025-01-06", "2025-12-19"
N_CLAIMS    = 1000
RATE_PER_MILE = 1.20          # £ per mile paid to garages
REGIONS = {"London & South East": .22, "Midlands": .20, "North West": .15,
           "North East & Yorkshire": .14, "South West": .10, "Scotland": .08,
           "Wales": .06, "East of England": .05}
GARAGE_TYPES = {"Recovery": .55, "Mechanical": .30, "Body Shop": .15}

# charge_type: (weight, is_mileage, (min£, max£) for non-mileage)
CHARGES = {
    "Additional Mileage": (.45, True,  None),
    "Trailer Mileage":    (.06, True,  None),
    "Skates":             (.12, False, (30, 80)),
    "Winching":           (.08, False, (40, 170)),
    "Crane / Hiab":       (.05, False, (120, 480)),
    "Tolls":              (.07, False, (2, 28)),
    "Ferry":              (.02, False, (30, 150)),
    "Callout Fee":        (.05, False, (45, 100)),
    "Misc Charges":       (.10, False, (10, 120)),
}
REASONS = ["Excessive mileage request", "Not chargeable under agreement",
           "Insufficient evidence", "Not authorised", "Duplicate charge"]
# P(reason | charge_type) for non-mileage charges
NON_MILEAGE_REASON_P = [0.0, .45, .30, .15, .10]

# ---------- DIMENSIONS ----------
dim_garage = pd.DataFrame({
    "garage_id": [f"G{i:03d}" for i in range(1, N_GARAGES + 1)],
    "region": rng.choice(list(REGIONS), N_GARAGES, p=list(REGIONS.values())),
    "garage_type": rng.choice(list(GARAGE_TYPES), N_GARAGES, p=list(GARAGE_TYPES.values())),
})

dim_charge_type = pd.DataFrame({
    "charge_type_id": range(1, len(CHARGES) + 1),
    "charge_type": list(CHARGES),
    "charge_group": ["Mileage" if v[1] else "Equipment & Fees" for v in CHARGES.values()],
})

dim_decline_reason = pd.DataFrame({
    "decline_reason_id": range(1, len(REASONS) + 1),
    "decline_reason": REASONS,
})

# ---------- GARAGE BEHAVIOUR (hidden drivers an analyst should discover) ----------
# Claim volume is skewed: a few garages claim a lot (lognormal weights).
garage_w = rng.lognormal(0, 1.0, N_GARAGES); garage_w /= garage_w.sum()
# ~10% of garages routinely inflate mileage more than others.
inflators = rng.random(N_GARAGES) < 0.10
garage_uplift_mu = np.where(inflators, 0.38, 0.18)

# ---------- FACT TABLE ----------
days = pd.bdate_range(START, END)
day_w = np.linspace(0.85, 1.15, len(days))            # gentle growth over the year
day_w = day_w * np.where(days.month == 12, 0.6, 1.0)  # quieter December
day_w /= day_w.sum()

charge_names = list(CHARGES)
charge_w = np.array([v[0] for v in CHARGES.values()]); charge_w /= charge_w.sum()

rows = []
for i in range(N_CLAIMS):
    g = rng.choice(N_GARAGES, p=garage_w)
    ct = rng.choice(len(charge_names), p=charge_w)
    name = charge_names[ct]
    _, is_mileage, rng_amt = CHARGES[name]
    logged = days[rng.choice(len(days), p=day_w)]
    job = logged - pd.Timedelta(days=int(rng.integers(1, 7)))

    mapped = req_miles = None
    if is_mileage:
        mapped = float(np.clip(rng.lognormal(np.log(38), 0.7), 5, 260))
        uplift = max(0.05, rng.normal(garage_uplift_mu[g], 0.08))
        req_miles = round(mapped * (1 + uplift))
        mapped = round(mapped)
        req_miles = max(req_miles, mapped + 1)
        requested = round(req_miles * RATE_PER_MILE, 2)
        saved = round((req_miles - mapped) * RATE_PER_MILE, 2)
        reason_id = 1
    else:
        requested = round(float(rng.uniform(*rng_amt)), 2)
        # 80% fully declined, 20% partially reduced
        frac = 1.0 if rng.random() < 0.8 else rng.uniform(0.3, 0.7)
        saved = round(requested * frac, 2)
        reason_id = int(rng.choice(range(1, 6), p=NON_MILEAGE_REASON_P)) + 0
    rows.append({
        "claim_id": f"C{i+1:05d}",
        "date_logged": logged.date(),
        "job_date": job.date(),
        "garage_id": dim_garage.garage_id[g],
        "charge_type_id": ct + 1,
        "decline_reason_id": reason_id,
        "requested_amount": requested,
        "approved_amount": round(requested - saved, 2),
        "amount_saved": saved,
        "mapped_miles": mapped,
        "requested_miles": req_miles,
    })

fact = pd.DataFrame(rows).sort_values("date_logged").reset_index(drop=True)
fact["claim_id"] = [f"C{i+1:05d}" for i in range(len(fact))]

out = "data/"
dim_garage.to_csv(out + "dim_garage.csv", index=False)
dim_charge_type.to_csv(out + "dim_charge_type.csv", index=False)
dim_decline_reason.to_csv(out + "dim_decline_reason.csv", index=False)
fact.to_csv(out + "fact_claims.csv", index=False)
print(f"fact_claims: {len(fact)} rows | total saved £{fact.amount_saved.sum():,.2f}")
