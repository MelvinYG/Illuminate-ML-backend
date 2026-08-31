# Illuminate ML Service

## Project documentation

- [Project context for humans, AI agents, and dependent services](PROJECT_CONTEXT.md)
- [ML theory, calculations, metrics, assumptions, and SDE-2 interview guide](ML_SYSTEM_AND_INTERVIEW_GUIDE.md)
- [ML improvement roadmap, data sources, effort, validation, and expected results](improvementplan.md)

## Health

```bash
curl -X GET http://localhost:8000/health
```

## Predict

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Bangalore,IN",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "current_battery_level_pct": 60,
    "settings": {
      "mode": "tou_savings",
      "battery_capacity_kwh": 15,
      "solar_capacity_kw": 5
    },
    "devices": [
      {
        "name": "washing_machine",
        "power_rating_kw": 0.8,
        "usage_hours": 2,
        "priority": "normal",
        "earliest_hour": 9,
        "latest_hour": 18,
        "contiguous": true,
        "category": "washing_machine"
      }
    ]
  }'
```

## Login

```bash
curl -X POST http://localhost:8000/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email":"<admin-email>","password":"<admin-password>"}'
```

Response includes `access_token` (HS256 JWT, 12h validity).

## Retrain (admin only)

```bash
curl -X POST http://localhost:8000/retrain \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"months_back":1,"notes":"manual retrain"}'
```

## Prediction logs (admin only)

```bash
curl -H "Authorization: Bearer <access_token>" \
  http://localhost:8000/prediction-logs
```

## Protected endpoints

- `POST /retrain` *(admin)*
- `GET /prediction-logs` *(admin)*
- `GET /admin/me` *(admin)*

## Public endpoints

- `GET /health`
- `POST /predict`
- `GET /notifications/{user_id}`
- `GET /history/{user_id}`
- `GET /model/versions`
