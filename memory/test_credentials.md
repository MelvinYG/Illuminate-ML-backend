# Test Credentials — Illuminate ML Service

## Admin (seeded from `.env`)
- **Email**: `admin@illuminate.com`
- **Password**: `admin123`
- **Role**: `admin`

## Login

```bash
curl -X POST http://localhost:8001/admin/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@illuminate.com","password":"admin123"}'
```

Response includes `access_token` (HS256 JWT, 12h validity).

## Use the JWT

```bash
curl -X POST http://localhost:8001/retrain \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"months_back":1,"notes":"manual retrain"}'

curl -H "Authorization: Bearer <access_token>" \
  http://localhost:8001/prediction-logs
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

## DB (PostgreSQL — local dev)
- Host: `localhost:5432`
- DB: `illuminate`
- User: `illuminate`
- Password: `illuminate`

## Env file: `/app/.env`
```
DATABASE_URL=postgresql://illuminate:illuminate@localhost:5432/illuminate
JWT_SECRET=<64-char hex>
ADMIN_EMAIL=admin@illuminate.com
ADMIN_PASSWORD=admin123
WEATHER_API_KEY=demo_key  (synthetic mock used while this is "demo_key")
```
