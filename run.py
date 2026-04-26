from inference import predict_from_questionnaire

# Fill in 7 days of data
days = []
for i in range(7):
    days.append({
        "sleep_duration_hrs": 5.5,
        "sleep_efficiency": 0.70,
        "heart_rate_resting": 70,
        "rmssd": 30, "sdnn": 35,
        "steps": 6000,
        "light_exposure_lux": 2000,
        "bedtime_hour": 1.5,   # 1:30 AM
        "isi_score": 16,
        "phq_score": 10,
        "gad_score": 9,
        "meq_score": 30,
    })

# v2: include user context for personalised results
result = predict_from_questionnaire(
    days,
    age=24,
    gender="female",
    has_caffeine=True,
    has_alcohol=False,
    acute_stress=False,
    on_medication=False,
)

pred = result["prediction"]
print(f"\n🧠 Sleep Duration Predicted : {pred['sleep_duration_hrs']} hrs ({pred['sleep_duration_label']})")
print(f"⚠️  Insomnia Risk             : {pred['insomnia_risk_level']} ({int(pred['insomnia_probability']*100)}%)")
print(f"🕐  Circadian Bedtime Est.    : {pred['circadian_bedtime_str']}")
print(f"🎯  Sleep Pattern Type        : {pred.get('sleep_pattern_type','—')}")
print(f"📏  Age-adjusted Target       : {pred.get('age_group_sleep_target','—')}")

if result.get("context_notes"):
    print("\n⚠️  Context Notes:")
    for note in result["context_notes"]:
        print(f"  {note}")

print("\n💡 Insights:")
for insight in result["insights"]:
    print(f"  - {insight}")

print("\n🩺 Recommendations:")
for rec in result["recommendations"]:
    print(f"  → {rec['title']}: {rec['desc']}")

print(f"\n{result['clinical_disclaimer']}")