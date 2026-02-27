/**
 * API Routes Module
 * =================
 * Express.js routes converted from FastAPI endpoints.
 */

import express from 'express';
import { Op } from 'sequelize';
import { Location, Review, DailyAggregate, FocusAreaState } from '../models/index.js';
import { generateMockData } from '../engine/dataGenerator.js';
import { runEngineForFocusArea, runEngineForPeriod } from '../engine/engineRunner.js';
import { computeRobustStatistics, mean } from '../engine/classifier.js';

const router = express.Router();

// Review volume by tier (daily average)
const TIER_REVIEW_VOLUMES = {
  NANO: 0.25,
  MICRO: 0.7,
  SMALL: 2.5,
  MEDIUM: 7.0,
  LARGE: 18.0,
  ENTERPRISE: 45.0
};

/**
 * GET /
 * Root endpoint
 */
router.get('/', (req, res) => {
  res.json({
    message: 'Revulog Intelligence Engine',
    version: '5.0 (Production)',
    tests_passed: '33/33'
  });
});

/**
 * POST /api/generate-data
 * Generate mock data
 */
router.post('/api/generate-data', async (req, res) => {
  try {
    await generateMockData();
    res.json({ success: true });
  } catch (error) {
    console.error('Error generating data:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * GET /api/locations
 * Get all locations
 */
router.get('/api/locations', async (req, res) => {
  try {
    const locations = await Location.findAll();
    const result = [];

    for (const loc of locations) {
      const states = await FocusAreaState.findAll({
        where: { location_id: loc.id }
      });

      const critical = states.filter(s => s.current_state === 'CRITICAL').length;
      const warnings = states.filter(s =>
        ['RECURRING', 'WORSENING', 'EMERGING'].includes(s.current_state)
      ).length;

      result.push({
        id: loc.id,
        name: loc.name,
        business_type: loc.business_type,
        address: loc.address,
        engine_tier: loc.engine_tier,
        monthly_volume: Math.round(loc.monthly_volume),
        overall_rating: loc.overall_rating,
        total_reviews: loc.total_reviews,
        critical_count: critical,
        warning_count: warnings
      });
    }

    res.json(result);
  } catch (error) {
    console.error('Error fetching locations:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * GET /api/locations/:locationId
 * Get location detail
 */
router.get('/api/locations/:locationId', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);
    const location = await Location.findByPk(locationId);

    if (!location) {
      return res.status(404).json({ detail: 'Location not found' });
    }

    const states = await FocusAreaState.findAll({
      where: { location_id: locationId }
    });

    const focusAreas = states.map(s => {
      let evidence = {};
      try {
        evidence = s.evidence_summary ? JSON.parse(s.evidence_summary) : {};
      } catch (e) {
        evidence = {};
      }

      return {
        name: s.focus_area,
        state: s.current_state,
        confidence: s.confidence,
        confidence_score: Math.round(s.confidence_score * 100) / 100,
        is_recurring: s.is_recurring,
        period_days: s.period_days,
        spike_count: s.spike_count,
        trend_direction: s.trend_direction,
        evidence
      };
    });

    const priority = {
      CRITICAL: 0, RECURRING: 1, WORSENING: 2, EMERGING: 3,
      IMPROVING: 4, RESOLVED: 5, STABLE: 6
    };
    focusAreas.sort((a, b) => (priority[a.state] || 99) - (priority[b.state] || 99));

    res.json({
      id: location.id,
      name: location.name,
      business_type: location.business_type,
      address: location.address,
      engine_tier: location.engine_tier,
      monthly_volume: Math.round(location.monthly_volume),
      overall_rating: location.overall_rating,
      total_reviews: location.total_reviews,
      focus_areas: focusAreas
    });
  } catch (error) {
    console.error('Error fetching location detail:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * GET /api/locations/:locationId/focus-areas/:focusArea/timeseries
 * Get timeseries data for a focus area
 */
router.get('/api/locations/:locationId/focus-areas/:focusArea/timeseries', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);
    const focusArea = req.params.focusArea;
    const daysParam = req.query.days ? parseInt(req.query.days) : null;

    const location = await Location.findByPk(locationId);
    if (!location) {
      return res.status(404).json({ detail: 'Location not found' });
    }

    const tier = location.engine_tier;

    let lookback;
    if (daysParam !== null) {
      lookback = Math.max(7, Math.min(daysParam, 365));
    } else {
      lookback = ['NANO', 'ENTERPRISE'].includes(tier) ? 180 : 90;
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const startDate = new Date(today);
    startDate.setDate(startDate.getDate() - lookback);

    const records = await DailyAggregate.findAll({
      where: {
        location_id: locationId,
        focus_area: focusArea,
        date: { [Op.gte]: startDate.toISOString().split('T')[0] }
      },
      order: [['date', 'ASC']]
    });

    const byDate = {};
    for (const r of records) {
      byDate[r.date] = r;
    }

    // Use weekly aggregation for small tiers OR short time periods
    const useWeekly = ['NANO', 'MICRO'].includes(tier) || lookback <= 30;

    if (useWeekly) {
      const weeks = [];
      const nWeeks = Math.floor(lookback / 7);

      for (let w = 0; w < nWeeks; w++) {
        const weekStart = new Date(startDate);
        weekStart.setDate(weekStart.getDate() + w * 7);
        let total = 0, issues = 0;

        for (let d = 0; d < 7; d++) {
          const day = new Date(weekStart);
          day.setDate(day.getDate() + d);
          const dayStr = day.toISOString().split('T')[0];
          if (byDate[dayStr]) {
            total += byDate[dayStr].total_reviews;
            issues += byDate[dayStr].issue_count;
          }
        }

        const rate = (issues / Math.max(total, 1)) * 100;
        weeks.push({
          week_start: weekStart.toISOString().split('T')[0],
          total_reviews: total,
          issues,
          rate: Math.round(rate * 10) / 10
        });
      }

      const rates = weeks.map(w => w.rate);
      const stats = computeRobustStatistics(rates);

      res.json({
        tier,
        chart_type: 'weekly_bars',
        data: weeks,
        baseline: Math.round(stats.baseline * 10) / 10,
        threshold: Math.round(stats.threshold * 10) / 10
      });
    } else {
      const data = [];
      const rates = [];

      for (let i = 0; i <= lookback; i++) {
        const d = new Date(startDate);
        d.setDate(d.getDate() + i);
        const dateStr = d.toISOString().split('T')[0];

        if (byDate[dateStr]) {
          const r = byDate[dateStr];
          const rate = (r.issue_count / Math.max(r.total_reviews, 1)) * 100;
          data.push({
            date: dateStr,
            total_reviews: r.total_reviews,
            issues: r.issue_count,
            rate: Math.round(rate * 10) / 10
          });
          rates.push(rate);
        } else {
          data.push({
            date: dateStr,
            total_reviews: 0,
            issues: 0,
            rate: 0
          });
          rates.push(0);
        }
      }

      const stats = computeRobustStatistics(rates);

      res.json({
        tier,
        chart_type: 'rate_chart',
        data,
        baseline: Math.round(stats.baseline * 10) / 10,
        upper_band: Math.round(stats.threshold * 10) / 10,
        lower_band: Math.round(Math.max(0, stats.baseline - (stats.threshold - stats.baseline)) * 10) / 10
      });
    }
  } catch (error) {
    console.error('Error fetching timeseries:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * GET /api/locations/:locationId/focus-areas/:focusArea/reviews
 * Get reviews for a focus area
 */
router.get('/api/locations/:locationId/focus-areas/:focusArea/reviews', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);
    const focusArea = req.params.focusArea;
    const limit = req.query.limit ? parseInt(req.query.limit) : 20;

    const reviews = await Review.findAll({
      where: {
        location_id: locationId,
        focus_areas: { [Op.like]: `%${focusArea}%` }
      },
      order: [['date', 'DESC']],
      limit
    });

    res.json(reviews.map(r => ({
      id: r.id,
      reviewer: r.reviewer_name,
      rating: r.rating,
      text: r.text,
      date: r.date,
      escalation: r.has_escalation
    })));
  } catch (error) {
    console.error('Error fetching reviews:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * POST /api/locations/:locationId/focus-areas/:focusArea/simulate-reviews
 * Simulate new reviews coming in for a focus area
 */
router.post('/api/locations/:locationId/focus-areas/:focusArea/simulate-reviews', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);
    const focusArea = req.params.focusArea;
    const days = req.query.days ? parseInt(req.query.days) : 7;
    const spike = req.query.spike === 'true';

    const location = await Location.findByPk(locationId);
    if (!location) {
      return res.status(404).json({ detail: 'Location not found' });
    }

    const tier = location.engine_tier;
    const dailyVolume = TIER_REVIEW_VOLUMES[tier] || 2.5;

    const today = new Date();
    today.setHours(0, 0, 0, 0);
    let reviewsAdded = 0;

    const state = await FocusAreaState.findOne({
      where: { location_id: locationId, focus_area: focusArea }
    });

    const currentState = state ? state.current_state : 'STABLE';

    let baseIssueProb;
    if (spike) {
      baseIssueProb = 0.6;
    } else if (currentState === 'CRITICAL') {
      baseIssueProb = 0.5;
    } else if (currentState === 'WORSENING') {
      baseIssueProb = 0.35;
    } else if (currentState === 'RECURRING') {
      baseIssueProb = 0.25;
    } else if (currentState === 'IMPROVING') {
      baseIssueProb = 0.1;
    } else {
      baseIssueProb = 0.15;
    }

    const reviewTemplates = [
      `The ${focusArea.toLowerCase()} was concerning during my visit.`,
      `Had issues with ${focusArea.toLowerCase()} this time.`,
      `Not happy with the ${focusArea.toLowerCase()} situation.`,
      `${focusArea} needs improvement.`,
      `Disappointed with ${focusArea.toLowerCase()} today.`
    ];

    const positiveTemplates = [
      `Great ${focusArea.toLowerCase()}! Very satisfied.`,
      `No issues with ${focusArea.toLowerCase()} at all.`,
      `${focusArea} was excellent as always.`,
      `Happy with the ${focusArea.toLowerCase()}.`,
      `Everything was fine regarding ${focusArea.toLowerCase()}.`
    ];

    const names = ['Alex', 'Jordan', 'Sam', 'Casey', 'Morgan', 'Riley', 'Taylor', 'Jamie', 'Quinn', 'Avery'];

    function poisson(lambda) {
      let L = Math.exp(-lambda);
      let k = 0;
      let p = 1;
      do {
        k++;
        p *= Math.random();
      } while (p > L);
      return k - 1;
    }

    for (let dayOffset = 0; dayOffset < days; dayOffset++) {
      const reviewDate = new Date(today);
      reviewDate.setDate(reviewDate.getDate() - (days - dayOffset - 1));
      const dateStr = reviewDate.toISOString().split('T')[0];

      const numReviews = Math.max(1, poisson(dailyVolume));

      for (let i = 0; i < numReviews; i++) {
        const hasIssue = Math.random() < baseIssueProb;

        await Review.create({
          location_id: locationId,
          reviewer_name: names[Math.floor(Math.random() * names.length)] + ' ' +
            String.fromCharCode(65 + Math.floor(Math.random() * 26)) + '.',
          rating: hasIssue ? Math.floor(Math.random() * 2) + 1 : Math.floor(Math.random() * 3) + 3,
          text: hasIssue
            ? reviewTemplates[Math.floor(Math.random() * reviewTemplates.length)]
            : positiveTemplates[Math.floor(Math.random() * positiveTemplates.length)],
          date: dateStr,
          focus_areas: hasIssue ? JSON.stringify([focusArea]) : JSON.stringify([]),
          has_escalation: hasIssue && Math.random() < 0.1
        });
        reviewsAdded++;
      }

      // Update daily aggregate
      const existingAgg = await DailyAggregate.findOne({
        where: { location_id: locationId, focus_area: focusArea, date: dateStr }
      });

      const dayReviews = await Review.findAll({
        where: { location_id: locationId, date: dateStr }
      });

      const totalReviews = dayReviews.length;
      const issues = dayReviews.filter(r => r.focus_areas && r.focus_areas.includes(focusArea)).length;

      if (existingAgg) {
        await existingAgg.update({
          total_reviews: totalReviews,
          issue_count: issues,
          issue_rate: issues / Math.max(totalReviews, 1),
          negative_count: issues
        });
      } else {
        await DailyAggregate.create({
          location_id: locationId,
          focus_area: focusArea,
          date: dateStr,
          total_reviews: totalReviews,
          issue_count: issues,
          issue_rate: issues / Math.max(totalReviews, 1),
          negative_count: issues
        });
      }
    }

    // Update location review count
    const totalCount = await Review.count({ where: { location_id: locationId } });
    await location.update({ total_reviews: totalCount });

    res.json({
      success: true,
      reviews_added: reviewsAdded,
      days_simulated: days,
      spike_mode: spike,
      tier,
      issue_probability: baseIssueProb
    });
  } catch (error) {
    console.error('Error simulating reviews:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * GET /api/locations/:locationId/focus-areas/:focusArea/analyze
 * Analyze a focus area with a custom time window
 */
router.get('/api/locations/:locationId/focus-areas/:focusArea/analyze', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);
    const focusArea = req.params.focusArea;
    let days = req.query.days ? parseInt(req.query.days) : 90;
    const comparePrevious = req.query.compare_previous === 'true';

    const location = await Location.findByPk(locationId);
    if (!location) {
      return res.status(404).json({ detail: 'Location not found' });
    }

    const tier = location.engine_tier;
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    days = Math.max(7, Math.min(days, 365));

    const currentStart = new Date(today);
    currentStart.setDate(currentStart.getDate() - days);
    const currentResult = await runEngineForPeriod(locationId, focusArea, tier, currentStart, today);

    const storedState = await FocusAreaState.findOne({
      where: { location_id: locationId, focus_area: focusArea }
    });

    const response = {
      current_period: {
        start_date: currentStart.toISOString().split('T')[0],
        end_date: today.toISOString().split('T')[0],
        days,
        classification: currentResult
      },
      stored_state: storedState ? storedState.current_state : null,
      state_changed: storedState ? storedState.current_state !== currentResult.state : null
    };

    if (comparePrevious) {
      const prevEnd = new Date(currentStart);
      prevEnd.setDate(prevEnd.getDate() - 1);
      const prevStart = new Date(prevEnd);
      prevStart.setDate(prevStart.getDate() - days);
      const prevResult = await runEngineForPeriod(locationId, focusArea, tier, prevStart, prevEnd);

      response.previous_period = {
        start_date: prevStart.toISOString().split('T')[0],
        end_date: prevEnd.toISOString().split('T')[0],
        days,
        classification: prevResult
      };

      const statePriority = {
        CRITICAL: 0, WORSENING: 1, RECURRING: 2, EMERGING: 3,
        STABLE: 4, IMPROVING: 5, RESOLVED: 6
      };

      const currentPriority = statePriority[currentResult.state] ?? 4;
      const prevPriority = statePriority[prevResult.state] ?? 4;

      if (currentPriority < prevPriority) {
        response.trend_summary = 'DETERIORATING';
      } else if (currentPriority > prevPriority) {
        response.trend_summary = 'IMPROVING';
      } else {
        response.trend_summary = 'UNCHANGED';
      }
    }

    res.json(response);
  } catch (error) {
    console.error('Error analyzing focus area:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * POST /api/locations/:locationId/focus-areas/:focusArea/refresh
 * Refresh the stored state for a focus area
 */
router.post('/api/locations/:locationId/focus-areas/:focusArea/refresh', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);
    const focusArea = req.params.focusArea;

    const location = await Location.findByPk(locationId);
    if (!location) {
      return res.status(404).json({ detail: 'Location not found' });
    }

    const tier = location.engine_tier;

    const oldState = await FocusAreaState.findOne({
      where: { location_id: locationId, focus_area: focusArea }
    });

    const oldStateValue = oldState ? oldState.current_state : null;

    const result = await runEngineForFocusArea(locationId, focusArea, tier);

    if (oldState) {
      await oldState.update({
        previous_state: oldState.current_state,
        current_state: result.state,
        confidence: result.confidence,
        confidence_score: result.confidence_score,
        is_recurring: result.is_recurring,
        period_days: result.period_days,
        spike_count: result.spike_count,
        trend_direction: result.trend_direction,
        evidence_summary: JSON.stringify(result.evidence),
        last_updated: new Date()
      });
    } else {
      await FocusAreaState.create({
        location_id: locationId,
        focus_area: focusArea,
        engine_tier: tier,
        current_state: result.state,
        confidence: result.confidence,
        confidence_score: result.confidence_score,
        is_recurring: result.is_recurring,
        period_days: result.period_days,
        spike_count: result.spike_count,
        trend_direction: result.trend_direction,
        evidence_summary: JSON.stringify(result.evidence)
      });
    }

    const stateChanged = oldStateValue !== result.state;

    res.json({
      success: true,
      previous_state: oldStateValue,
      current_state: result.state,
      state_changed: stateChanged,
      classification: result,
      alert: stateChanged ? generateStateChangeAlert(oldStateValue, result.state) : null
    });
  } catch (error) {
    console.error('Error refreshing focus area state:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * GET /api/locations/:locationId/alerts
 * Get all focus areas that need attention for a location
 */
router.get('/api/locations/:locationId/alerts', async (req, res) => {
  try {
    const locationId = parseInt(req.params.locationId);

    const states = await FocusAreaState.findAll({
      where: { location_id: locationId }
    });

    const critical = [];
    const warning = [];
    const improving = [];
    const stable = [];

    for (const s of states) {
      const item = {
        focus_area: s.focus_area,
        state: s.current_state,
        confidence: s.confidence,
        previous_state: s.previous_state,
        spike_count: s.spike_count,
        last_updated: s.last_updated ? s.last_updated.toISOString() : null
      };

      if (s.current_state === 'CRITICAL') {
        critical.push(item);
      } else if (['WORSENING', 'RECURRING', 'EMERGING'].includes(s.current_state)) {
        warning.push(item);
      } else if (['IMPROVING', 'RESOLVED'].includes(s.current_state)) {
        improving.push(item);
      } else {
        stable.push(item);
      }
    }

    res.json({
      critical,
      warning,
      improving,
      stable,
      summary: {
        total_issues: critical.length + warning.length,
        needs_attention: critical.length,
        monitoring: warning.length,
        improving: improving.length
      }
    });
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ detail: error.message });
  }
});

/**
 * Generate an alert message for state transitions.
 */
function generateStateChangeAlert(oldState, newState) {
  const statePriority = {
    CRITICAL: 0, WORSENING: 1, RECURRING: 2, EMERGING: 3,
    STABLE: 4, IMPROVING: 5, RESOLVED: 6
  };

  const oldPriority = oldState ? (statePriority[oldState] ?? 4) : 4;
  const newPriority = statePriority[newState] ?? 4;

  if (newPriority < oldPriority) {
    if (newState === 'CRITICAL') {
      return {
        type: 'URGENT',
        message: `⚠️ CRITICAL ALERT: Issue escalated from ${oldState} to CRITICAL`,
        action: 'Immediate investigation required'
      };
    } else {
      return {
        type: 'WARNING',
        message: `📈 Issue worsened: Changed from ${oldState} to ${newState}`,
        action: 'Monitor closely and consider intervention'
      };
    }
  } else {
    return {
      type: 'POSITIVE',
      message: `✅ Issue improved: Changed from ${oldState} to ${newState}`,
      action: 'Continue current measures'
    };
  }
}

export default router;
