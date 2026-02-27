/**
 * Engine Runner Module
 * ====================
 * Runs the classification engine for focus areas.
 */

import { DailyAggregate } from '../models/index.js';
import { classify } from './classifier.js';
import { Op } from 'sequelize';

/**
 * Run classification engine for a specific focus area.
 */
async function runEngineForFocusArea(locationId, focusArea, tier, transaction = null) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  
  const lookback = ['NANO', 'ENTERPRISE'].includes(tier) ? 180 : 90;
  const startDate = new Date(today);
  startDate.setDate(startDate.getDate() - lookback);

  const queryOptions = {
    where: {
      location_id: locationId,
      focus_area: focusArea,
      date: { [Op.gte]: startDate.toISOString().split('T')[0] }
    },
    order: [['date', 'ASC']]
  };

  if (transaction) {
    queryOptions.transaction = transaction;
  }

  const records = await DailyAggregate.findAll(queryOptions);

  const dateLookup = {};
  for (const r of records) {
    dateLookup[r.date] = r;
  }

  const N = [];
  const C = [];

  for (let i = 0; i <= lookback; i++) {
    const d = new Date(startDate);
    d.setDate(d.getDate() + i);
    const dateStr = d.toISOString().split('T')[0];

    if (dateLookup[dateStr]) {
      N.push(dateLookup[dateStr].total_reviews);
      C.push(dateLookup[dateStr].issue_count);
    } else {
      N.push(0);
      C.push(0);
    }
  }

  // For small tiers, aggregate to weekly BUT use issue RATE not raw counts
  if (['NANO', 'MICRO'].includes(tier)) {
    const nWeeks = Math.floor(N.length / 7);
    const ratesWeekly = [];

    for (let i = 0; i < nWeeks; i++) {
      let weekTotal = 0;
      let weekIssues = 0;
      for (let j = 0; j < 7; j++) {
        weekTotal += N[i * 7 + j];
        weekIssues += C[i * 7 + j];
      }
      if (weekTotal >= 1) {
        ratesWeekly.push((weekIssues / weekTotal) * 100);
      } else {
        ratesWeekly.push(0);
      }
    }

    // Filter out zero weeks for better signal
    const activeWeeks = ratesWeekly.filter(r => r > 0);
    if (activeWeeks.length < 3) {
      return {
        state: 'STABLE',
        confidence: 'Very Low',
        confidence_score: 0.3,
        spike_count: 0,
        is_recurring: false,
        period_days: null,
        trend_direction: null,
        evidence: { note: 'Insufficient data for analysis' }
      };
    }

    const result = classify(ratesWeekly, tier);
    return {
      state: result.state,
      confidence: result.confidence,
      confidence_score: result.confidence_score,
      spike_count: result.spike_count,
      is_recurring: result.is_recurring,
      period_days: result.period_days,
      trend_direction: result.trend_direction,
      evidence: result.evidence
    };
  } else {
    // Compute daily rates
    const rates = N.map((n, i) => (n >= 1 ? (C[i] / n) * 100 : 0));
    const result = classify(rates, tier);

    return {
      state: result.state,
      confidence: result.confidence,
      confidence_score: result.confidence_score,
      spike_count: result.spike_count,
      is_recurring: result.is_recurring,
      period_days: result.period_days,
      trend_direction: result.trend_direction,
      evidence: result.evidence
    };
  }
}

/**
 * Run classification engine for a specific date range.
 */
async function runEngineForPeriod(locationId, focusArea, tier, startDate, endDate, transaction = null) {
  const queryOptions = {
    where: {
      location_id: locationId,
      focus_area: focusArea,
      date: {
        [Op.gte]: startDate.toISOString().split('T')[0],
        [Op.lte]: endDate.toISOString().split('T')[0]
      }
    },
    order: [['date', 'ASC']]
  };

  if (transaction) {
    queryOptions.transaction = transaction;
  }

  const records = await DailyAggregate.findAll(queryOptions);

  const dateLookup = {};
  for (const r of records) {
    dateLookup[r.date] = r;
  }

  const N = [];
  const C = [];

  const current = new Date(startDate);
  while (current <= endDate) {
    const dateStr = current.toISOString().split('T')[0];
    if (dateLookup[dateStr]) {
      N.push(dateLookup[dateStr].total_reviews);
      C.push(dateLookup[dateStr].issue_count);
    } else {
      N.push(0);
      C.push(0);
    }
    current.setDate(current.getDate() + 1);
  }

  const daysCount = N.length;

  // For small tiers or short periods, aggregate to weekly
  if (['NANO', 'MICRO'].includes(tier) || daysCount <= 30) {
    const nWeeks = Math.floor(daysCount / 7);
    if (nWeeks < 2) {
      return {
        state: 'STABLE',
        confidence: 'Very Low',
        confidence_score: 0.3,
        spike_count: 0,
        is_recurring: false,
        period_days: null,
        trend_direction: null,
        evidence: { note: 'Insufficient data for analysis' }
      };
    }

    const ratesWeekly = [];
    for (let i = 0; i < nWeeks; i++) {
      let weekTotal = 0;
      let weekIssues = 0;
      for (let j = 0; j < 7; j++) {
        weekTotal += N[i * 7 + j];
        weekIssues += C[i * 7 + j];
      }
      if (weekTotal >= 1) {
        ratesWeekly.push((weekIssues / weekTotal) * 100);
      } else {
        ratesWeekly.push(0);
      }
    }

    const result = classify(ratesWeekly, tier);
    return {
      state: result.state,
      confidence: result.confidence,
      confidence_score: result.confidence_score,
      spike_count: result.spike_count,
      is_recurring: result.is_recurring,
      period_days: result.period_days,
      trend_direction: result.trend_direction,
      evidence: result.evidence
    };
  } else {
    const rates = N.map((n, i) => (n >= 1 ? (C[i] / n) * 100 : 0));
    const result = classify(rates, tier);

    return {
      state: result.state,
      confidence: result.confidence,
      confidence_score: result.confidence_score,
      spike_count: result.spike_count,
      is_recurring: result.is_recurring,
      period_days: result.period_days,
      trend_direction: result.trend_direction,
      evidence: result.evidence
    };
  }
}

export { runEngineForFocusArea, runEngineForPeriod };
