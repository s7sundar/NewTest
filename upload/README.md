# Revulog Intelligence Engine - Node.js v5.0

Express.js backend converted from FastAPI Python implementation.

## Requirements

- Node.js v24+

## Setup

```bash
cd nodejs
npm install
```

## Run

```bash
# Production
npm start

# Development (with auto-reload)
npm run dev
```

Server runs on `http://localhost:8000`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root - version info |
| POST | `/api/generate-data` | Generate mock data |
| GET | `/api/locations` | List all locations |
| GET | `/api/locations/:id` | Get location detail |
| GET | `/api/locations/:id/focus-areas/:area/timeseries` | Get timeseries data |
| GET | `/api/locations/:id/focus-areas/:area/reviews` | Get reviews |
| POST | `/api/locations/:id/focus-areas/:area/simulate-reviews` | Simulate reviews |
| GET | `/api/locations/:id/focus-areas/:area/analyze` | Analyze with custom window |
| POST | `/api/locations/:id/focus-areas/:area/refresh` | Refresh state |
| GET | `/api/locations/:id/alerts` | Get location alerts |

## Project Structure

```
nodejs/
├── package.json
├── README.md
├── revulog_demo.db          # SQLite database (auto-created)
└── src/
    ├── app.js               # Entry point
    ├── models/
    │   └── index.js         # Sequelize models
    ├── engine/
    │   ├── classifier.js    # Classification engine
    │   ├── engineRunner.js  # Engine runner utilities
    │   └── dataGenerator.js # Mock data generation
    └── routes/
        └── api.js           # API routes
```

## States

- **CRITICAL** - Unexpected spike after stable history
- **RECURRING** - Periodic pattern with consistent spikes
- **WORSENING** - Rising baseline trend
- **EMERGING** - New issue appearing
- **IMPROVING** - Declining trend from elevated levels
- **RESOLVED** - Previously problematic, now stable
- **STABLE** - No significant issues
