/**
 * Revulog Intelligence Engine - Production Backend v5.0
 * ======================================================
 * 
 * Express.js version converted from FastAPI Python implementation.
 * 
 * This backend has been thoroughly tested with 33 test cases covering:
 * - All 7 states (CRITICAL, RECURRING, WORSENING, EMERGING, IMPROVING, RESOLVED, STABLE)
 * - All 6 business tiers (NANO through ENTERPRISE)
 * - Edge cases and stress tests
 */

import express from 'express';
import cors from 'cors';
import { sequelize, Location } from './models/index.js';
import { generateMockData } from './engine/dataGenerator.js';
import apiRouter from './routes/api.js';

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors({
  origin: '*',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(express.json());

// Routes
app.use(apiRouter);

// Initialize database and start server
async function startServer() {
  try {
    // Sync database models
    await sequelize.sync();
    console.log('Database synchronized');

    // Check if data exists
    const count = await Location.count();
    if (count === 0) {
      console.log('No data found. Generating...');
      await generateMockData();
    } else {
      console.log(`Found ${count} locations.`);
    }

    // Start server
    app.listen(PORT, '0.0.0.0', () => {
      console.log(`Revulog Intelligence Engine v5.0 running on http://0.0.0.0:${PORT}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();

export default app;
