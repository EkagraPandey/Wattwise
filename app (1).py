import os
import json
import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_energy_model.joblib"
META_PATH = BASE_DIR / "model_metadata.json"

# Load the already-trained model and its saved metadata.
final_model = joblib.load(MODEL_PATH)
with META_PATH.open("r", encoding="utf-8") as f:
    meta = json.load(f)

FEATURES = meta["features"]
TARIFF = {str(k): float(v) for k, v in meta["tariff_by_city"].items()}
CITY_CHOICES = sorted(str(x) for x in meta["cities"])
SEASON_CHOICES = sorted(str(x) for x in meta["seasons"])

APP_FEATURES = ["Fan", "Refrigerator", "AirConditioner", "Television", "Monitor"]
MAX_REDUCTION = 0.40
APP_DAYS_PER_MONTH = 30

# Preserve the notebook's wattage lookup when the uploaded ZIP contains a
# CSV with appliance and power_watts_max columns. Otherwise use its fallbacks.
FALLBACK_WATTS = {
    "Fan": 83.0,
    "Refrigerator": 871.5,
    "AirConditioner": 45.0,
    "Television": 54.0,
    "Monitor": 31.5,
}
ALIASES = {
    "Fan": {"fan", "fans"},
    "Refrigerator": {"refrigerator", "fridge"},
    "AirConditioner": {"airconditioner", "air conditioner", "ac"},
    "Television": {"television", "tv"},
    "Monitor": {"monitor", "screen"},
}

def _normalise_appliance(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()

APP_WATTS = FALLBACK_WATTS.copy()

# Find and extract the first CSV in the dataset ZIP, regardless of its filename.
zip_files = sorted(BASE_DIR.glob("*.zip"))
if zip_files:
    extracted_dir = BASE_DIR / "_dataset_extracted"
    extracted_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_files[0], "r") as zf:
        csv_members = [n for n in zf.namelist()
                       if n.lower().endswith(".csv") and not n.startswith("__MACOSX/")]
        if csv_members:
            member = csv_members[0]
            safe_name = Path(member).name
            extracted_csv = extracted_dir / safe_name
            if not extracted_csv.exists():
                with zf.open(member) as src, extracted_csv.open("wb") as dst:
                    dst.write(src.read())
            try:
                watt_df = pd.read_csv(extracted_csv, low_memory=False)
                if "appliance" in watt_df.columns and "power_watts_max" in watt_df.columns:
                    power_rows = watt_df[["appliance", "power_watts_max"]].copy()
                    power_rows["power_watts_max"] = pd.to_numeric(
                        power_rows["power_watts_max"], errors="coerce"
                    )
                    power_rows = power_rows.dropna(subset=["appliance", "power_watts_max"])
                    power_rows = power_rows[power_rows["power_watts_max"] > 0]
                    power_rows["_normalised"] = power_rows["appliance"].map(_normalise_appliance)
                    for feature, names in ALIASES.items():
                        match = power_rows[
                            power_rows["_normalised"].isin({_normalise_appliance(x) for x in names})
                        ]
                        if not match.empty:
                            APP_WATTS[feature] = float(match["power_watts_max"].median())
            except Exception:
                # The saved model can still serve predictions using fallback wattages.
                pass

def run_energy_model(payload):
    """Use the saved trained model for the primary monthly kWh forecast."""
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Please enter the household details and try again."}

        city = str(payload.get("city", ""))
        season = str(payload.get("season", ""))
        if city not in CITY_CHOICES or season not in SEASON_CHOICES:
            return {"ok": False, "message": "Choose a valid city and season."}

        values = {
            feature: float(payload.get(feature, 0) or 0)
            for feature in APP_FEATURES
        }
        hours_per_day = {
            feature: float(payload.get(f"{feature}_hours_per_day", 0) or 0)
            for feature in APP_FEATURES
        }

        if any(value < 0 for value in values.values()):
            return {"ok": False, "message": "Appliance values cannot be negative."}
        if any(value < 0 or value > 24 for value in hours_per_day.values()):
            return {"ok": False, "message": "Daily appliance hours must be between 0 and 24."}

        if any(value > 0 for value in values.values()):
            row = pd.DataFrame(
                [{"city": city, "season": season, **values}],
                columns=FEATURES,
            )
            model_reference_kwh = max(0.0, float(final_model.predict(row)[0]))
        else:
            model_reference_kwh = 0.0

        # The trained notebook model predicts monthly_total_kwh from the exact
        # saved feature set: city, season, and the five appliance quantities.
        # This prediction is the primary forecast; the bill is derived from it.
        kwh = model_reference_kwh
        tariff = float(TARIFF.get(city, 0.0))

        return {
            "ok": True,
            "kwh": kwh,
            "bill": kwh * tariff,
            "model_reference_kwh": model_reference_kwh,
            "tariff": tariff,
            "city": city,
            "season": season,
            "message": "Prediction ready.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Could not calculate the prediction: {exc}",
        }

APP_DATA = json.dumps({
    "cities": CITY_CHOICES,
    "seasons": SEASON_CHOICES,
    "appliances": APP_FEATURES,
    "watts": APP_WATTS,
    "tariffs": TARIFF,
    "maxReduction": MAX_REDUCTION,
}, ensure_ascii=False)
APP_DATA = APP_DATA.replace("</", "<\\/")

APP_HTML = r'''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--ink:#172b26;--muted:#71817a;--paper:#f6f5ef;--white:#fffefa;--green:#c8f169;--deep:#143d32;--line:#e6e8dc;--coral:#ff8067;--blue:#cbe9ee;--shadow:0 24px 65px rgba(23,43,38,.11)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:'DM Sans',sans-serif;font-size:15px}
button,input,select{font:inherit}button{cursor:pointer} .shell{max-width:1280px;margin:auto;padding:20px 26px 60px}
.nav{display:flex;align-items:center;justify-content:space-between;padding:8px 4px 22px}.brand{display:flex;align-items:center;gap:10px;font:700 20px 'Space Grotesk',sans-serif;letter-spacing:-.7px}.brand-mark{width:35px;height:35px;border-radius:12px;background:var(--deep);color:var(--green);display:grid;place-items:center;font-size:19px;box-shadow:0 6px 16px #143d3230}.nav-right{display:flex;align-items:center;gap:18px;color:var(--muted);font-size:13px}.live{display:flex;align-items:center;gap:8px}.live i{width:8px;height:8px;border-radius:50%;background:#60b878;box-shadow:0 0 0 5px #60b87822;animation:pulse 1.9s infinite}
.hero{display:grid;grid-template-columns:1.04fr .96fr;min-height:370px;background:var(--deep);color:#f6f8ef;border-radius:29px;overflow:hidden;position:relative;box-shadow:var(--shadow)}.hero:before{content:'';position:absolute;width:430px;height:430px;background:#c8f16917;border-radius:50%;filter:blur(3px);left:-145px;bottom:-275px}.hero-copy{padding:47px 34px 42px 48px;position:relative;z-index:2}.eyebrow{font:500 11px 'DM Mono',monospace;text-transform:uppercase;letter-spacing:1.8px;color:var(--green);display:flex;align-items:center;gap:9px}.eyebrow:before{content:'';width:18px;height:1px;background:var(--green)}h1{font:600 clamp(38px,5vw,66px)/.98 'Space Grotesk',sans-serif;letter-spacing:-3.8px;margin:23px 0 17px;max-width:570px}h1 span{color:var(--green)}.hero-copy>p{color:#bdcbc1;max-width:450px;line-height:1.65;margin:0}.hero-actions{display:flex;gap:10px;margin-top:28px;flex-wrap:wrap}.button{border:0;border-radius:12px;padding:13px 18px;font-weight:700;transition:transform .2s,box-shadow .2s,background .2s}.button:hover{transform:translateY(-2px);box-shadow:0 10px 25px #0002}.button-lime{background:var(--green);color:var(--deep)}.button-glass{background:#ffffff10;color:#f7f9f1;border:1px solid #ffffff27}.hero-art{position:relative;min-height:330px;display:flex;align-items:center;justify-content:center;overflow:hidden;background:radial-gradient(ellipse at 58% 45%,#34715a 0%,#194b3d 38%,#143d32 75%)}.hero-art:after{content:'';position:absolute;inset:0;background-image:radial-gradient(#d8f3d71b 1px,transparent 1px);background-size:20px 20px;mask-image:linear-gradient(90deg,transparent,black)}.orbit{position:absolute;width:320px;height:320px;border:1px solid #dbf4d828;border-radius:50%;animation:spin 36s linear infinite}.orbit:before,.orbit:after{content:'';position:absolute;border-radius:50%;background:var(--green);box-shadow:0 0 20px #c8f169}.orbit:before{width:8px;height:8px;top:32px;left:70px}.orbit:after{width:5px;height:5px;bottom:37px;right:36px;background:#ff896e;box-shadow:0 0 16px #ff896e}.sun{position:absolute;width:81px;height:81px;background:#d8fb91;border-radius:50%;top:38px;right:14%;filter:drop-shadow(0 0 28px #c8f16966);animation:bob 5s ease-in-out infinite}.house{position:relative;z-index:1;width:min(92%,440px);filter:drop-shadow(0 23px 20px #061b163c);animation:bob 6s ease-in-out infinite}.float-chip{position:absolute;z-index:3;padding:11px 14px;border:1px solid #ffffff35;border-radius:13px;background:#163e33c9;backdrop-filter:blur(9px);color:#e6f2df;font:11px 'DM Mono',monospace;box-shadow:0 12px 25px #061b1640}.chip-a{left:8%;top:23%}.chip-b{right:7%;bottom:15%}.chip-a b,.chip-b b{color:var(--green);font-size:15px;display:block;margin-top:4px}
.intro-row{display:flex;align-items:end;justify-content:space-between;gap:20px;margin:39px 2px 18px}.section-tag{color:#82918a;font:500 10px 'DM Mono',monospace;text-transform:uppercase;letter-spacing:1.5px}.intro-row h2,.panel-head h2{font:600 27px 'Space Grotesk',sans-serif;letter-spacing:-1px;margin:5px 0 0}.intro-row p{color:var(--muted);margin:0 0 3px;font-size:13px}.workspace{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}.panel{background:var(--white);border:1px solid #e9e9df;border-radius:22px;padding:23px;box-shadow:0 8px 25px #1c392309;transition:transform .25s,box-shadow .25s}.panel:hover{box-shadow:0 16px 40px #1c392310}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:10px;margin-bottom:20px}.panel-head h2{font-size:21px}.panel-head p{margin:6px 0 0;color:var(--muted);font-size:12px;line-height:1.5}.icon-box{width:42px;height:42px;border-radius:14px;display:grid;place-items:center;background:#e9f6d2;color:#35562a;font-size:20px;flex:none}.step{color:#a2ada3;font:500 10px 'DM Mono',monospace;letter-spacing:1px;margin-top:6px}.fields{display:grid;grid-template-columns:1fr 1fr;gap:12px}.field label{display:block;color:#63736a;font:500 11px 'DM Mono',monospace;margin-bottom:7px}.field input,.field select,.hours-input{width:100%;border:1px solid #e0e4d9;background:#fbfbf6;color:var(--ink);border-radius:10px;padding:12px;outline:none;transition:border .2s,box-shadow .2s}.field input:focus,.field select:focus,.hours-input:focus{border-color:#8db349;box-shadow:0 0 0 3px #c8f16940}.appliance-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}.appliance-input{border:1px solid #e9eae1;border-radius:12px;padding:11px;background:#fff;transition:border .18s,transform .18s}.appliance-input:focus-within{border-color:#accb6d;transform:translateY(-1px)}.appliance-input label{display:flex;align-items:center;gap:7px;color:#65756b;font-size:11px;font-weight:600;margin-bottom:7px}.appliance-input .emoji{font-size:15px}.appliance-input input{width:100%;border:0;outline:0;background:transparent;font:600 17px 'Space Grotesk',sans-serif;color:var(--ink)}.hint{font-size:10px;color:#9ba79e;margin-top:3px}.full{width:100%;margin-top:17px}.result{min-height:64px;margin-top:15px;border-radius:13px;background:#f4f6ee;padding:14px;color:#758279;font-size:12px;line-height:1.55}.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.result-box{padding:14px;border-radius:13px;background:#eef4e5}.result-box.dark{background:var(--deep);color:#fff}.result-label{font:10px 'DM Mono',monospace;text-transform:uppercase;letter-spacing:.6px;opacity:.7}.result-value{font:600 25px 'Space Grotesk',sans-serif;letter-spacing:-1px;margin-top:6px}.result-small{font-size:11px;opacity:.68;margin-top:2px}.divider{height:1px;background:var(--line);margin:17px 0}.target-row{display:grid;grid-template-columns:1fr 1fr;gap:11px}.callout{background:#f2f0e5;border:1px solid #e9e5d3;color:#737564;border-radius:11px;padding:11px 12px;font-size:11px;line-height:1.5;margin-top:12px}.callout strong{color:#46553b}.rows{display:grid;gap:8px;margin:13px 0}.usage-row{display:grid;grid-template-columns:minmax(115px,1.2fr) minmax(60px,.5fr) minmax(100px,1fr) 34px;gap:8px;align-items:center}.usage-row select,.usage-row input{width:100%;min-width:0;border:1px solid #e0e4d9;background:#fff;border-radius:10px;padding:11px;color:var(--ink)}.usage-row button{height:38px;border:1px solid #f0d9d3;background:#fff8f6;border-radius:9px;color:#c36d5c;transition:background .2s}.usage-row button:hover{background:#ffe9e2}.add-row{background:#f4f6ed;color:#50653d;border:1px dashed #c9d5b6;border-radius:10px;padding:10px 13px;font-size:12px;font-weight:700}.add-row:hover{background:#eaf3d9}.schedule{width:100%;border-collapse:collapse;margin-top:13px;font-size:11px}.schedule th{text-align:left;color:#849087;font:10px 'DM Mono',monospace;padding:8px 6px;border-bottom:1px solid var(--line)}.schedule td{padding:10px 6px;border-bottom:1px solid #eff0e9}.schedule td:last-child,.schedule th:last-child{text-align:right}.notice{padding:12px;border-left:3px solid var(--coral);background:#fff4ef;color:#86655d;border-radius:0 10px 10px 0;margin-top:12px;font-size:11px;line-height:1.5}.status{display:flex;gap:8px;align-items:center;margin-top:11px;font-size:11px;color:var(--muted)}.status.good{color:#327044}.status.bad{color:#b45747}.spinner{display:inline-block;width:14px;height:14px;border:2px solid #c8d2c7;border-top-color:#456e42;border-radius:50%;animation:spin .7s linear infinite}.footer{display:flex;justify-content:space-between;gap:18px;align-items:center;padding:20px 4px 0;color:#9aa49d;font-size:10px}.footer strong{color:#738277}.toast{position:fixed;right:25px;bottom:25px;z-index:50;background:#193e33;color:#f4f8ec;border:1px solid #ffffff20;padding:14px 17px;border-radius:13px;box-shadow:var(--shadow);transform:translateY(18px);opacity:0;pointer-events:none;transition:.3s}.toast.show{opacity:1;transform:translateY(0)}
@keyframes spin{to{transform:rotate(360deg)}}@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-9px)}}@keyframes pulse{50%{box-shadow:0 0 0 8px #60b87808;opacity:.65}}@media(prefers-reduced-motion:reduce){*,*:before,*:after{animation-duration:.01ms!important;animation-iteration-count:1!important;scroll-behavior:auto!important;transition-duration:.01ms!important}}
@media(max-width:850px){.hero{grid-template-columns:1fr}.hero-copy{padding:38px 30px 28px}.hero-art{min-height:265px}.house{width:360px}.workspace{grid-template-columns:1fr}.intro-row{align-items:start;flex-direction:column}.nav-right{gap:9px}.nav-right span:not(.live){display:none}}
@media(max-width:520px){.shell{padding:12px 13px 35px}.hero{border-radius:21px}.hero-copy{padding:30px 23px 24px}h1{font-size:43px;letter-spacing:-2.8px}.hero-art{min-height:220px}.house{width:290px}.sun{width:55px;height:55px;right:10%;top:28px}.orbit{width:240px;height:240px}.float-chip{font-size:9px;padding:8px 10px}.panel{padding:17px;border-radius:17px}.fields,.target-row{grid-template-columns:1fr}.appliance-grid{gap:7px}.appliance-input{padding:9px}.usage-row{grid-template-columns:minmax(0,1.2fr) 54px minmax(0,1fr) 28px;gap:4px}.usage-row select,.usage-row input{padding:9px 6px;font-size:11px}.intro-row h2{font-size:24px}.footer{align-items:flex-start;flex-direction:column}}
</style>
<style>
:root{--ink:#f2efff;--muted:#aaa2c4;--paper:#100d20;--white:#1a162b;--green:#83e9e1;--deep:#19152b;--line:#342c49;--coral:#ff9eae;--blue:#bd9dff;--shadow:0 24px 70px #05030c70}
body{background:#100d20;color:#f2efff}
.shell{max-width:1280px;color:#f2efff}
.nav{border-bottom:1px solid #332d4b}
.brand-mark{background:#bd9dff;color:#1b1430}
.nav-right{color:#b9b1d2}
.live i{background:#83e9e1;box-shadow:0 0 0 5px #83e9e122}
.hero{background:linear-gradient(115deg,#19152b 0%,#231b3d 58%,#19152b 100%);border:1px solid #3c3157;box-shadow:0 20px 60px #05030c88}
.hero:before{background:#8e72e922;filter:blur(18px)}
.eyebrow{color:#83e9e1}
.eyebrow:before{background:#83e9e1}
h1 span{color:#83e9e1}
.hero-copy>p{color:#c0b9d5}
.hero-art{background:radial-gradient(ellipse at 58% 45%,#493271 0%,#231b3d 42%,#100d20 84%)}
.hero-art:after{background-image:radial-gradient(#d7cfff2b 1px,transparent 1px);mask-image:linear-gradient(90deg,transparent,black)}
.orbit{border-color:#d7cfff30}
.orbit:before,.orbit:after{background:#83e9e1;box-shadow:0 0 22px #83e9e1}
.orbit:after{background:#bd9dff;box-shadow:0 0 18px #bd9dff}
.sun{width:142px;height:142px;top:calc(50% - 71px);right:calc(50% - 71px);background:radial-gradient(circle at 35% 33%,#a7f5ed 0 6%,#8275e7 19%,#40316a 48%,#151024 70%);box-shadow:0 0 0 11px #bd9dff12,0 0 42px #9380ff75;filter:none;animation:bob 5s ease-in-out infinite}
.house{display:none}
.float-chip{background:#1b1730dc;border-color:#8c7ebc55;color:#e9e3fa;box-shadow:0 14px 32px #08061175}
.chip-a b,.chip-b b{color:#83e9e1}
.button-lime{background:#83e9e1;color:#152127}
.button-glass{background:#ffffff0a;color:#f2efff;border-color:#ffffff2a}
.intro-row .section-tag{color:#83e9e1}
.intro-row p{color:#aaa2c4}
.panel{background:#1a162b;border-color:#342c49;box-shadow:0 10px 34px #05030c55}
.panel:hover{box-shadow:0 16px 42px #05030c88}
.panel-head p{color:#aaa2c4}
.step{color:#8d85aa}
.icon-box{background:#2b2443;color:#83e9e1}
.field label{color:#c5bdd9}
.field input,.field select,.hours-input,.quantity-input,.usage-row select,.usage-row input{background:#110e20;color:#eee9ff;border-color:#3c3453}
.field input:focus,.field select:focus,.hours-input:focus{border-color:#83e9e1;box-shadow:0 0 0 3px #83e9e122}
.appliance-input{background:#141125;border-color:#2c2640}
.appliance-input:focus-within{border-color:#83e9e1}
.appliance-input label{color:#c5bdd9}
.appliance-input .appliance-title{display:flex;align-items:center;gap:7px;margin-bottom:7px;color:#c5bdd9;font-size:11px;font-weight:600}
.appliance-input input{color:#f2efff}
.appliance-values{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.appliance-values .value-field label{display:block;margin:0 0 4px;font-size:9px;letter-spacing:.5px}
.appliance-values .value-field input{width:100%;padding:8px;border:1px solid #3c3453;border-radius:8px;background:#110e20;font:600 14px 'Space Grotesk',sans-serif}
.appliance-values .value-field input:focus{border-color:#83e9e1;box-shadow:0 0 0 3px #83e9e122}
.hint{color:#aaa2c4}
.result-grid{grid-template-columns:repeat(auto-fit,minmax(170px,1fr))}
.result{background:#141125;color:#aaa2c4}
.result-box{background:#28213e;color:#f2efff}
.result-box.dark{background:#83e9e1;color:#152127}
.divider{background:#342c49}
.callout{background:#201a35;border-color:#3d3357;color:#c4bbd9}
.callout strong{color:#83e9e1}
.usage-row button{background:#392335;color:#ff9eae;border-color:#593246}
.usage-row button:hover{background:#4a2a3b}
.add-row{background:#201a35;color:#83e9e1;border-color:#62558d}
.add-row:hover{background:#292142}
.schedule th{color:#aaa2c4;border-color:#342c49}
.schedule td{border-color:#2c2640}
.notice{background:#2a1b30;color:#d6b9ce;border-color:#ff9eae}
.status{color:#aaa2c4}
.status.good{color:#83e9e1}
.status.bad{color:#ff9eae}
.spinner{border-color:#564c72;border-top-color:#83e9e1}
.footer{color:#8d85aa}
.footer strong{color:#d7cfff}
.toast{background:#211a37;color:#f2efff;border-color:#65558a}
.hero-art .chip-a{left:7%;top:20%}
.hero-art .chip-b{right:7%;bottom:17%}
@media(max-width:850px){.hero-art{min-height:265px}.sun{width:116px;height:116px;top:calc(50% - 58px);right:calc(50% - 58px)}}
@media(max-width:520px){.sun{width:88px;height:88px;top:calc(50% - 44px);right:calc(50% - 44px)}.hero-art .chip-a{left:4%;top:12%}.hero-art .chip-b{right:4%;bottom:10%}}
@media(max-width:520px){.appliance-grid{grid-template-columns:1fr}}
</style>
<div class="shell">
  <nav class="nav"><div class="brand"><div class="brand-mark">✦</div>WattWise / Eclipse</div><div class="nav-right"><span class="live"><i></i> Connected</span><span>Monthly energy planner</span></div></nav>
  <header class="hero">
    <div class="hero-copy"><div class="eyebrow">Light up your next decision</div><h1>Forecast first.<br><span>Then fine-tune.</span></h1><p>Get your household prediction, then shape a realistic appliance schedule around your monthly bill target.</p><div class="hero-actions"><button class="button button-lime" onclick="document.querySelector('#forecast').scrollIntoView({behavior:'smooth'})">01&nbsp; Run forecast&nbsp; ↘</button><button class="button button-glass" onclick="document.querySelector('#reducer').scrollIntoView({behavior:'smooth'})">02&nbsp; Plan usage</button></div></div>
    <div class="hero-art" aria-label="Glowing solar eclipse illustration"><div class="orbit"></div><div class="sun"></div>
      <div class="float-chip chip-a">YOUR MONTHLY OUTLOOK<b>Forecast first</b></div><div class="float-chip chip-b">REALISTIC TARGETS<b>Adjust with confidence</b></div>
      <svg class="house" viewBox="0 0 500 360" role="img" aria-label="A modern home with solar panels">
        <defs><linearGradient id="wall" x1="0" x2="1" y1="0" y2="1"><stop stop-color="#f8f4df"/><stop offset="1" stop-color="#d8dfc3"/></linearGradient><linearGradient id="roof" x1="0" x2="1"><stop stop-color="#a8db69"/><stop offset="1" stop-color="#6dbb73"/></linearGradient><linearGradient id="glass" x1="0" x2="1"><stop stop-color="#a8d8d0"/><stop offset="1" stop-color="#579b8a"/></linearGradient></defs>
        <ellipse cx="253" cy="322" rx="180" ry="20" fill="#061b16" opacity=".25"/><path d="M102 172 250 66l148 106v139H102z" fill="url(#wall)"/><path d="m74 176 176-126 177 126-17 24L250 91 91 200z" fill="#183d32"/><path d="m118 168 129-91 131 91-11 12-120-83-119 83z" fill="url(#roof)"/><path d="M178 123 239 80l58 40-60 43z" fill="#234f49"/><path d="m187 123 52-37v13l-52 38zM217 143l54-38v12l-54 39z" fill="#9bd6d0" opacity=".9"/><path d="M175 202h67v109h-67z" fill="#b7d2c6"/><path d="M183 211h51v91h-51z" fill="url(#glass)"/><path d="M255 205h48v56h-48z" fill="#f3b970"/><path d="M263 213h32v40h-32z" fill="#f9da98"/><path d="M318 205h48v56h-48z" fill="#f3b970"/><path d="M326 213h32v40h-32z" fill="#f9da98"/><path d="M266 269h40v42h-40z" fill="#bd7955"/><path d="M287 269v42" stroke="#e7bd87" stroke-width="2"/><path d="M119 277c-28-21-30-59-8-82 3 31 20 45 36 50 5 15-7 29-28 32zM386 286c28-21 30-59 8-82-3 31-20 45-36 50-5 15 7 29 28 32z" fill="#78af68"/><path d="M91 310h324" stroke="#b5d879" stroke-width="5" stroke-linecap="round"/><circle cx="138" cy="278" r="10" fill="#ff9e73"/><circle cx="376" cy="287" r="8" fill="#f2d86e"/>
      </svg>
    </div>
  </header>
  <div class="intro-row"><div><div class="section-tag">WattWise / 02</div><h2>Your monthly outlook</h2></div><p>Predict household use, then fine-tune a practical schedule.</p></div>
  <main class="workspace">
    <section class="panel" id="forecast"><div class="panel-head"><div><div class="step">01 / FORECAST</div><h2>Household snapshot</h2><p>Enter appliance quantities and average daily hours. The bill uses rated watts, 30 days of use, and your city tariff.</p></div><div class="icon-box">⌂</div></div>
      <div class="fields"><div class="field"><label for="city">CITY</label><select id="city"></select></div><div class="field"><label for="season">SEASON</label><select id="season"></select></div></div>
      <div id="applianceInputs" class="appliance-grid"></div>
      <button id="predictBtn" class="button button-lime full">Show my energy forecast&nbsp; →</button>
      <div id="predictionResult" class="result" aria-live="polite">Your forecast will appear here. With no appliance values entered, the estimate is ₹0.</div>
      <div id="predictionStatus" class="status"></div>
    </section>
    <section class="panel" id="reducer"><div class="panel-head"><div><div class="step">02 / COST REDUCER</div><h2>Shape your monthly plan</h2><p>Starts with the quantities and daily hours from Household Snapshot. Adjust the quantity or monthly hours before calculating.</p></div><div class="icon-box">↘</div></div>
      <div class="target-row"><div class="field"><label for="targetBill">YOUR TARGET BILL (₹ / MONTH)</label><input id="targetBill" type="number" min="0" step="any" placeholder="Run forecast first"></div><div class="field"><label for="planCity">TARIFF CITY</label><select id="planCity"></select></div></div>
      <div class="callout">Your target can be up to <strong>40% lower than the predicted bill</strong>. Larger reductions are flagged as unrealistic for this planner.</div>
      <div class="divider"></div><div class="field"><label>APPLIANCE SCHEDULE · QUANTITY AND HOURS PER MONTH</label></div>
      <div id="usageRows" class="rows"></div><button id="addRow" class="add-row">＋ Add appliance row</button>
      <button id="planBtn" class="button button-lime full">Find my suggested schedule&nbsp; →</button>
      <div id="planResult" class="result" aria-live="polite">Run the forecast, set a target, and add appliance hours to see a suggested schedule.</div>
      <div id="planStatus" class="status"></div>
    </section>
  </main>
  <footer class="footer"><span><strong>WattWise / Eclipse</strong> · Household energy planning</span><span>Bill estimates use appliance quantities, daily hours, dataset wattage and city tariff. The trained-model reference is shown separately.</span></footer>
</div><div id="toast" class="toast" role="status"></div>
'''

APP_JS = r'''
(() => {
  const cfg = __APP_DATA__;
  const $ = (s) => document.querySelector(s);
  const money = (x) => new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0}).format(Number(x)||0);
  const number = (x,d=1) => new Intl.NumberFormat('en-IN',{maximumFractionDigits:d}).format(Number(x)||0);
  const safe = (x) => String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const setStatus = (id,text,kind='') => { const n=$('#'+id); n.className='status '+kind; n.innerHTML=text; };
  const toast = (message) => { const n=$('#toast'); n.textContent=message; n.classList.add('show'); setTimeout(()=>n.classList.remove('show'),2400); };
  const title = (name) => name==='AirConditioner' ? 'Air conditioner' : name;
  const icons = {Fan:'✳',Refrigerator:'▣',AirConditioner:'❄',Television:'▰',Monitor:'▣'};
  const city=$('#city'), season=$('#season'), planCity=$('#planCity');
  cfg.cities.forEach(x=>{city.add(new Option(x,x));planCity.add(new Option(x,x));});
  cfg.seasons.forEach(x=>season.add(new Option(x,x)));
  const applianceInputs=$('#applianceInputs');
  cfg.appliances.forEach((name,i)=>{
    const box=document.createElement('div'); box.className='appliance-input';
    box.innerHTML=`<div class="appliance-title"><span class="emoji">${icons[name]||'◉'}</span>${safe(title(name))}</div><div class="appliance-values"><div class="value-field"><label for="count-${i}">QUANTITY</label><input id="count-${i}" type="number" min="0" step="1" placeholder="0"></div><div class="value-field"><label for="hours-${i}">HOURS / DAY</label><input id="hours-${i}" type="number" min="0" max="24" step="0.5" placeholder="0"></div></div>`;
    applianceInputs.append(box);
  });

  const usageRows=$('#usageRows');
  function addUsageRow(name='',quantity=1,hours=''){
    const row=document.createElement('div'); row.className='usage-row';
    const options=cfg.appliances.map(x=>`<option value="${safe(x)}" ${x===name?'selected':''}>${safe(title(x))}</option>`).join('');
    row.innerHTML=`<select aria-label="Appliance"><option value="">Choose appliance</option>${options}</select><input class="quantity-input" type="number" min="0" step="1" value="${safe(quantity)}" aria-label="Appliance quantity" title="Quantity"><input class="hours-input" type="number" min="0" step="any" placeholder="Hours / month per appliance" value="${safe(hours)}" aria-label="Hours per month per appliance" title="Hours per month per appliance"><button type="button" aria-label="Remove row">×</button>`;
    row.querySelector('button').onclick=()=>{if(usageRows.children.length>1)row.remove();else{row.querySelector('select').value='';row.querySelector('.quantity-input').value='1';row.querySelector('.hours-input').value='';}};
    usageRows.append(row);
  }
  function loadSnapshotIntoReducer(snapshot){
    usageRows.replaceChildren();
    const selected=snapshot.filter(item=>item.quantity>0);
    if(!selected.length){addUsageRow();return;}
    selected.forEach(item=>addUsageRow(item.name,item.quantity,(item.hoursPerDay*30).toFixed(1)));
  }
  addUsageRow();
  $('#addRow').onclick=()=>addUsageRow();

  let predicted=null;
  $('#predictBtn').onclick=async()=>{
    const button=$('#predictBtn'); button.disabled=true; button.innerHTML='<span class="spinner"></span> Calculating your outlook…';
    setStatus('predictionStatus','Running the trained household energy model…');
    try{
      const payload={city:city.value,season:season.value};
      cfg.appliances.forEach((name,i)=>{
        payload[name]=Number($('#count-'+i).value)||0;
        payload[`${name}_hours_per_day`]=Number($('#hours-'+i).value)||0;
      });
      const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
      const result=await response.json();
      if(!result.ok)throw new Error(result.message||'Prediction could not be calculated.');
      const snapshot=cfg.appliances.map((name,i)=>({
        name,
        quantity:Number($('#count-'+i).value)||0,
        hoursPerDay:Number($('#hours-'+i).value)||0
      }));
      predicted=result;
      loadSnapshotIntoReducer(snapshot);
      $('#predictionResult').innerHTML=`<div class="result-grid"><div class="result-box"><div class="result-label">ML-PREDICTED MONTHLY USE</div><div class="result-value">${number(result.kwh)} <span style="font-size:13px">kWh</span></div><div class="result-small">Predicted by your trained model using city, season and appliance quantities</div></div><div class="result-box dark"><div class="result-label">ESTIMATED BILL</div><div class="result-value">${money(result.bill)}</div><div class="result-small">ML-predicted kWh × ${money(result.tariff)} per kWh · ${safe(result.city)}</div></div></div>`;
      if(!$('#targetBill').value)$('#targetBill').value=Number(result.bill).toFixed(0);
      planCity.value=result.city;
      setStatus('predictionStatus','Forecast ready. Set your target and build a schedule.','good');
      toast('Your monthly energy forecast is ready.');
    }catch(e){setStatus('predictionStatus',safe(e.message||String(e)),'bad');}
    finally{button.disabled=false;button.textContent='Show my energy forecast →';}
  };

  $('#planBtn').onclick=()=>{
    if(!predicted){setStatus('planStatus','Run your household forecast first so the target has a prediction to compare with.','bad');$('#forecast').scrollIntoView({behavior:'smooth',block:'center'});return;}
    const target=Number($('#targetBill').value);
    if(!Number.isFinite(target)||target<0){setStatus('planStatus','Enter a target bill of ₹0 or more.','bad');return;}
    const floor=predicted.bill*(1-cfg.maxReduction);
    if(target+0.005<floor){
      $('#planResult').innerHTML=`<div class="result-box" style="background:#3b2337"><div class="result-label">TARGET TOO LOW FOR THIS FORECAST</div><div class="result-value" style="color:#ff9eae">${money(target)}</div><div class="result-small">The lowest realistic target in this planner is ${money(floor)} (40% below the ${money(predicted.bill)} prediction).</div></div>`;
      setStatus('planStatus','This reduction is outside the planner’s realistic range. Raise the target to continue.','bad');return;
    }
    const tariff=Number(cfgTariff(planCity.value));
    const items=[...usageRows.children].map(row=>({
      name:row.querySelector('select').value,
      quantity:Math.max(0,Number(row.querySelector('.quantity-input').value)||0),
      hours:Math.max(0,Number(row.querySelector('.hours-input').value)||0)
    })).filter(x=>x.name&&x.quantity>0&&x.hours>0);

    if(!items.length){
      $('#planResult').innerHTML=`<div class="result-grid"><div class="result-box"><div class="result-label">ESTIMATED USE</div><div class="result-value">0 <span style="font-size:13px">kWh</span></div></div><div class="result-box dark"><div class="result-label">APPLIANCE COST</div><div class="result-value">${money(0)}</div></div></div><div class="callout">No appliance hours were entered, so the appliance schedule cost is <strong>₹0</strong>. Add hours to receive a suggested schedule.</div>`;
      setStatus('planStatus','Your empty schedule costs ₹0. Add any of the five listed appliances to plan hours.','good');return;
    }

    const rawCost=items.reduce((sum,x)=>sum+(Number(cfg.watts[x.name]||0)*x.quantity*x.hours/1000*tariff),0);
    const factor=rawCost>0?Math.min(1,target/rawCost):1;
    const finalCost=rawCost*factor;
    const finalKwh=items.reduce((sum,x)=>sum+(Number(cfg.watts[x.name]||0)*x.quantity*x.hours*factor/1000),0);
    const body=items.map(x=>{
      const suggestedHours=x.hours*factor;
      const kwh=Number(cfg.watts[x.name]||0)*x.quantity*suggestedHours/1000;
      const cost=kwh*tariff;
      return `<tr><td>${safe(title(x.name))}</td><td>${number(x.quantity,0)}</td><td>${number(suggestedHours)}</td><td>${number(kwh,2)} kWh</td><td>${money(cost)}</td></tr>`;
    }).join('');
    const within=rawCost<=target+0.005;
    $('#planResult').innerHTML=`<div class="result-grid"><div class="result-box"><div class="result-label">SCHEDULE ENERGY</div><div class="result-value">${number(finalKwh)} <span style="font-size:13px">kWh</span></div><div class="result-small">At your suggested monthly hours</div></div><div class="result-box dark"><div class="result-label">SUGGESTED COST</div><div class="result-value">${money(finalCost)}</div><div class="result-small">Entered schedule: ${money(rawCost)}</div></div></div><div style="overflow:auto"><table class="schedule"><thead><tr><th>APPLIANCE</th><th>QUANTITY</th><th>HOURS / MO. EACH</th><th>ENERGY</th><th>COST</th></tr></thead><tbody>${body}</tbody></table></div>`;
    setStatus('planStatus',within?'Your entered schedule is within the target bill.':'Hours have been adjusted proportionally to fit your target bill.',within?'good':'');
  };

  function cfgTariff(cityName){
    return cfg.tariffs[cityName]||0;
  }
  $('#city').addEventListener('change',()=>{if(!predicted)planCity.value=city.value;});
  city.addEventListener('change',()=>{if(!predicted)planCity.value=city.value;});
})();
'''

APP_DATA = APP_DATA.replace("</", "<\\/")
APP_JS = APP_JS.replace("__APP_DATA__", APP_DATA)

flask_app = Flask("wattwise_energy_planner")
FULL_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WattWise — Solar Eclipse energy planner</title></head><body>""" + APP_HTML + "<script>" + APP_JS + "</script></body></html>"

@flask_app.get("/")
def home_page():
    return make_response(FULL_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"})

@flask_app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    return jsonify(run_energy_model(payload))
