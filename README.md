# Illuminate ML Service

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
    "battery_capacity_kwh": 15,
    "current_battery_level_pct": 60,
    "solar_capacity_kw": 100,
    "appliances": ["washing_machine", "dishwasher", "pump"]
  }'
```

## Login

```bash
curl -X POST http://localhost:8000/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email":"email1@illuminate.com","password":"email1@#123"}'
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


