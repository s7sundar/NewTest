/**
 * Mock Data Generation Module
 * ===========================
 * Generates comprehensive mock data with realistic patterns.
 */

import { Location, Review, DailyAggregate, FocusAreaState, sequelize } from '../models/index.js';
import { runEngineForFocusArea } from './engineRunner.js';

const REVIEW_TEMPLATES = {
  'Food Quality': {
    negative: [
      'The food was cold when it arrived.',
      'Found something strange in my meal.',
      'Food was undercooked and tasteless.',
      'The dish tasted like it was reheated.',
      'Portions are tiny for the price.'
    ],
    positive: ['The food was absolutely delicious!', 'Fresh ingredients and great flavors.']
  },
  'Service': {
    negative: [
      'Waited forever just to get our order taken.',
      'The waiter was incredibly rude.',
      'Staff completely ignored us.'
    ],
    positive: ['Our server was fantastic!', 'Quick service and friendly staff.']
  },
  'Wait Time': {
    negative: [
      'Waited over an hour for our food.',
      '30 minute wait despite having a reservation.'
    ],
    positive: ['Food came out surprisingly fast.']
  },
  'Hygiene': {
    negative: ['The bathroom was filthy.', 'Tables were sticky and dirty.'],
    positive: ['Spotlessly clean restaurant.'],
    escalation: ['I got food poisoning after eating here.', 'Found a bug in my food.']
  },
  'Value': {
    negative: ['Overpriced for what you get.', 'Not worth the money at all.'],
    positive: ['Great value for money.']
  },
  'Ambiance': {
    negative: ['Way too loud in there.', 'Lighting is terrible.'],
    positive: ['Love the atmosphere here.']
  },
  'Delivery': {
    negative: ['Delivery took 2 hours.', 'Order was completely wrong.'],
    positive: ['Super fast delivery!']
  }
};

const FIRST_NAMES = [
  'James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael', 'Linda',
  'David', 'Barbara', 'Raj', 'Priya', 'Wei', 'Mei', 'Carlos', 'Maria'
];

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function generateReviewerName() {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  return `${randomChoice(FIRST_NAMES)} ${randomChoice(letters.split(''))}.`;
}

function generateReviewText(focusArea, rating) {
  const templates = REVIEW_TEMPLATES[focusArea] || REVIEW_TEMPLATES['Service'];

  if (rating <= 2) {
    if (templates.escalation && Math.random() < 0.1) {
      return [randomChoice(templates.escalation), true];
    }
    return [randomChoice(templates.negative), false];
  } else {
    return [randomChoice(templates.positive || ['Good experience.']), false];
  }
}

function calculateIssueProbability(pattern, daysAgo, dow) {
  const ptype = pattern.type;

  if (ptype === 'stable') {
    return pattern.base_prob || 0.1;
  }

  if (ptype === 'critical') {
    if (daysAgo <= (pattern.spike_day || 7)) {
      return pattern.spike_prob || 0.6;
    }
    return pattern.base_prob || 0.1;
  }

  if (ptype === 'recurring') {
    const period = pattern.period || 7;
    const phase = daysAgo % period;
    if (phase < period * 0.25) {
      return pattern.spike_prob || 0.5;
    }
    return pattern.base_prob || 0.1;
  }

  if (ptype === 'worsening') {
    const progress = (180 - daysAgo) / 180;
    const start = pattern.start_prob || 0.1;
    const end = pattern.end_prob || 0.4;
    return start + progress * (end - start);
  }

  if (ptype === 'improving') {
    const progress = (180 - daysAgo) / 180;
    const start = pattern.start_prob || 0.4;
    const end = pattern.end_prob || 0.1;
    return start + progress * (end - start);
  }

  if (ptype === 'emerging') {
    const emergeDay = pattern.emerge_day || 45;
    if (daysAgo > emergeDay) {
      return pattern.base_prob || 0.02;
    }
    return pattern.emerge_prob || 0.3;
  }

  if (ptype === 'resolved') {
    const probStart = pattern.problem_start || 60;
    const probEnd = pattern.problem_end || 120;
    if (daysAgo >= probStart && daysAgo <= probEnd) {
      return pattern.problem_prob || 0.4;
    }
    return pattern.base_prob || 0.05;
  }

  return 0.1;
}

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

function weightedChoice(items, weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < items.length; i++) {
    r -= weights[i];
    if (r <= 0) return items[i];
  }
  return items[items.length - 1];
}

async function generateMockData() {
  const transaction = await sequelize.transaction();

  try {
    // Delete existing data
    await FocusAreaState.destroy({ where: {}, transaction });
    await DailyAggregate.destroy({ where: {}, transaction });
    await Review.destroy({ where: {}, transaction });
    await Location.destroy({ where: {}, transaction });

    const locationsConfig = [
      {
        name: "Grandma's Kitchen",
        business_type: 'Family Restaurant',
        address: '123 Quiet Lane',
        tier: 'NANO',
        daily_reviews: 0.25,
        patterns: {
          'Hygiene': { type: 'recurring', period: 28, spike_prob: 0.6, base_prob: 0.05 },
          'Food Quality': { type: 'stable', base_prob: 0.1 },
          'Service': { type: 'emerging', emerge_day: 45, base_prob: 0.02, emerge_prob: 0.3 }
        }
      },
      {
        name: 'Corner Cafe Delights',
        business_type: 'Cafe',
        address: '456 Main Street',
        tier: 'MICRO',
        daily_reviews: 0.7,
        patterns: {
          'Wait Time': { type: 'recurring', period: 7, spike_prob: 0.55, base_prob: 0.15 },
          'Food Quality': { type: 'worsening', start_prob: 0.1, end_prob: 0.35 },
          'Service': { type: 'stable', base_prob: 0.08 },
          'Ambiance': { type: 'improving', start_prob: 0.3, end_prob: 0.08 }
        }
      },
      {
        name: 'The Urban Grill',
        business_type: 'Casual Dining',
        address: '789 Food Street',
        tier: 'SMALL',
        daily_reviews: 2.5,
        patterns: {
          'Service': { type: 'critical', spike_day: 5, spike_prob: 0.7, base_prob: 0.12 },
          'Food Quality': { type: 'recurring', period: 14, spike_prob: 0.5, base_prob: 0.15 },
          'Wait Time': { type: 'worsening', start_prob: 0.12, end_prob: 0.4 },
          'Hygiene': { type: 'stable', base_prob: 0.05 },
          'Value': { type: 'stable', base_prob: 0.1 }
        }
      },
      {
        name: 'Spice Garden Central',
        business_type: 'Indian Restaurant',
        address: '321 Curry Road',
        tier: 'MEDIUM',
        daily_reviews: 7.0,
        patterns: {
          'Food Quality': { type: 'stable', base_prob: 0.08 },
          'Service': { type: 'recurring', period: 7, spike_prob: 0.45, base_prob: 0.12 },
          'Wait Time': { type: 'critical', spike_day: 3, spike_prob: 0.65, base_prob: 0.15 },
          'Delivery': { type: 'worsening', start_prob: 0.1, end_prob: 0.35 },
          'Hygiene': { type: 'resolved', problem_start: 60, problem_end: 120, problem_prob: 0.4, base_prob: 0.03 },
          'Value': { type: 'improving', start_prob: 0.25, end_prob: 0.05 }
        }
      },
      {
        name: 'Burger Nation Mall',
        business_type: 'Fast Food Chain',
        address: '555 Shopping Mall',
        tier: 'LARGE',
        daily_reviews: 18.0,
        patterns: {
          'Food Quality': { type: 'recurring', period: 7, spike_prob: 0.5, base_prob: 0.12 },
          'Service': { type: 'stable', base_prob: 0.1 },
          'Wait Time': { type: 'worsening', start_prob: 0.08, end_prob: 0.35 },
          'Hygiene': { type: 'critical', spike_day: 4, spike_prob: 0.7, base_prob: 0.08 },
          'Value': { type: 'stable', base_prob: 0.12 }
        }
      },
      {
        name: 'FreshBites Airport',
        business_type: 'QSR Enterprise',
        address: 'Terminal 2, Airport',
        tier: 'ENTERPRISE',
        daily_reviews: 45.0,
        patterns: {
          'Wait Time': { type: 'recurring', period: 7, spike_prob: 0.55, base_prob: 0.15 },
          'Food Quality': { type: 'recurring', period: 14, spike_prob: 0.4, base_prob: 0.1 },
          'Service': { type: 'worsening', start_prob: 0.08, end_prob: 0.28 },
          'Hygiene': { type: 'stable', base_prob: 0.05 },
          'Value': { type: 'critical', spike_day: 2, spike_prob: 0.6, base_prob: 0.15 },
          'Ambiance': { type: 'improving', start_prob: 0.2, end_prob: 0.05 }
        }
      }
    ];

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    for (const locConfig of locationsConfig) {
      const location = await Location.create({
        name: locConfig.name,
        business_type: locConfig.business_type,
        address: locConfig.address,
        engine_tier: locConfig.tier,
        monthly_volume: locConfig.daily_reviews * 30
      }, { transaction });

      console.log(`Creating data for ${location.name} (${location.engine_tier})...`);

      const allReviews = [];
      const dailyData = {};
      for (const fa of Object.keys(locConfig.patterns)) {
        dailyData[fa] = {};
      }

      for (let daysAgo = 180; daysAgo >= 0; daysAgo--) {
        const reviewDate = new Date(today);
        reviewDate.setDate(reviewDate.getDate() - daysAgo);
        const dateStr = reviewDate.toISOString().split('T')[0];
        const dow = reviewDate.getDay();

        const dowMult = dow === 0 || dow === 6 ? 1.3 : 1.0;
        const numReviews = Math.max(0, poisson(locConfig.daily_reviews * dowMult));

        for (const fa of Object.keys(locConfig.patterns)) {
          if (!dailyData[fa][dateStr]) {
            dailyData[fa][dateStr] = { total: 0, issues: 0 };
          }
          dailyData[fa][dateStr].total = numReviews;
        }

        for (let i = 0; i < numReviews; i++) {
          const mentionedAreas = [];

          for (const [fa, pattern] of Object.entries(locConfig.patterns)) {
            const prob = calculateIssueProbability(pattern, daysAgo, dow);
            if (Math.random() < prob) {
              mentionedAreas.push(fa);
            }
          }

          let rating, text, escalation;
          if (mentionedAreas.length > 0) {
            const primary = mentionedAreas[0];
            rating = weightedChoice([1, 2, 3], [0.3, 0.5, 0.2]);
            [text, escalation] = generateReviewText(primary, rating);
          } else {
            rating = weightedChoice([3, 4, 5], [0.15, 0.35, 0.5]);
            text = 'Good experience overall.';
            escalation = false;
          }

          allReviews.push({
            location_id: location.id,
            reviewer_name: generateReviewerName(),
            rating,
            text,
            date: dateStr,
            focus_areas: JSON.stringify(mentionedAreas),
            issue_weight: rating === 1 ? 4.0 : (rating === 2 ? 3.0 : (rating === 3 ? 1.0 : 0)),
            has_escalation: escalation
          });

          for (const fa of mentionedAreas) {
            dailyData[fa][dateStr].issues += 1;
          }
        }
      }

      if (allReviews.length > 0) {
        await Review.bulkCreate(allReviews, { transaction });
      }

      const aggregates = [];
      for (const [fa, dates] of Object.entries(dailyData)) {
        for (const [dateStr, data] of Object.entries(dates)) {
          const total = data.total;
          const issues = data.issues;
          aggregates.push({
            location_id: location.id,
            focus_area: fa,
            date: dateStr,
            total_reviews: total,
            issue_count: issues,
            issue_rate: issues / Math.max(total, 1),
            negative_count: issues
          });
        }
      }

      if (aggregates.length > 0) {
        await DailyAggregate.bulkCreate(aggregates, { transaction });
      }

      await location.update({
        total_reviews: allReviews.length,
        overall_rating: allReviews.length > 0
          ? Math.round((allReviews.reduce((sum, r) => sum + r.rating, 0) / allReviews.length) * 10) / 10
          : 4.0
      }, { transaction });

      for (const fa of Object.keys(locConfig.patterns)) {
        const result = await runEngineForFocusArea(location.id, fa, locConfig.tier, transaction);

        await FocusAreaState.create({
          location_id: location.id,
          focus_area: fa,
          engine_tier: locConfig.tier,
          current_state: result.state,
          confidence: result.confidence,
          confidence_score: result.confidence_score,
          is_recurring: result.is_recurring,
          period_days: result.period_days,
          spike_count: result.spike_count,
          trend_direction: result.trend_direction,
          evidence_summary: JSON.stringify(result.evidence)
        }, { transaction });
      }

      console.log(`  ✓ ${location.name}: ${allReviews.length} reviews`);
    }

    await transaction.commit();
    console.log('✅ Mock data generation complete!');
    return true;
  } catch (error) {
    await transaction.rollback();
    console.error('Error:', error);
    throw error;
  }
}

export { generateMockData, REVIEW_TEMPLATES, FIRST_NAMES };
