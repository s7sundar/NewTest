/**
 * Revulog Intelligence Engine - Classification Engine
 * ====================================================
 * 
 * This module has been converted from Python to JavaScript.
 * Key improvements over v3/v4:
 * 1. RECURRING vs WORSENING: Distinguished by baseline stability (25th percentile)
 * 2. CRITICAL only fires for unexpected spikes (not recurring patterns)
 * 3. Spike magnitude consistency prevents noise from triggering false recurring
 * 4. FFT-based periodicity with higher SNR threshold (6.0) for robustness
 * 5. Short data handling with adjusted max_gap for spike grouping
 */

const State = {
  CRITICAL: 'CRITICAL',
  RECURRING: 'RECURRING',
  WORSENING: 'WORSENING',
  EMERGING: 'EMERGING',
  IMPROVING: 'IMPROVING',
  RESOLVED: 'RESOLVED',
  STABLE: 'STABLE'
};

function median(arr) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function mean(arr) {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((acc, val) => acc + Math.pow(val - m, 2), 0) / arr.length);
}

function percentile(arr, p) {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] * (upper - index) + sorted[upper] * (index - lower);
}

function polyfit(x, y) {
  const n = x.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
  }
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  return [slope, intercept];
}

function corrcoef(x, y) {
  const n = x.length;
  const meanX = mean(x);
  const meanY = mean(y);
  let num = 0, denX = 0, denY = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    num += dx * dy;
    denX += dx * dx;
    denY += dy * dy;
  }
  if (denX === 0 || denY === 0) return 0;
  return num / Math.sqrt(denX * denY);
}

function fft(values) {
  const n = values.length;
  if (n <= 1) return values.map(v => ({ re: v, im: 0 }));
  
  const result = [];
  for (let k = 0; k < n; k++) {
    let re = 0, im = 0;
    for (let t = 0; t < n; t++) {
      const angle = (2 * Math.PI * t * k) / n;
      re += values[t] * Math.cos(angle);
      im -= values[t] * Math.sin(angle);
    }
    result.push({ re, im });
  }
  return result;
}

function rfft(values) {
  const full = fft(values);
  return full.slice(0, Math.floor(values.length / 2) + 1);
}

function rfftfreq(n, d = 1.0) {
  const freqs = [];
  for (let i = 0; i <= Math.floor(n / 2); i++) {
    freqs.push(i / (n * d));
  }
  return freqs;
}

function hanning(n) {
  const window = [];
  for (let i = 0; i < n; i++) {
    window.push(0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1))));
  }
  return window;
}

/**
 * Compute robust statistics using median and MAD.
 */
function computeRobustStatistics(values) {
  if (values.length === 0) {
    return { median: 0, mad: 0, robust_std: 0, threshold: 0, baseline: 0 };
  }

  const zeroRatio = values.filter(v => v === 0).length / values.length;
  const nonZero = values.filter(v => v > 0);

  if (zeroRatio > 0.7) {
    const med = 0;
    let threshold = 5;
    if (nonZero.length > 0) {
      threshold = Math.max(Math.min(...nonZero) * 0.5, 5);
    }
    return { median: med, mad: 0, robust_std: 0, threshold, baseline: med };
  }

  let med, mad;
  if (zeroRatio > 0.5) {
    if (nonZero.length >= 3) {
      med = median(nonZero);
      mad = median(nonZero.map(v => Math.abs(v - med)));
    } else {
      med = median(values);
      mad = median(values.map(v => Math.abs(v - med)));
    }
  } else {
    med = median(values);
    mad = median(values.map(v => Math.abs(v - med)));
  }

  const robustStd = 1.4826 * mad;
  const threshold = Math.max(med + 2.5 * robustStd, med * 1.3, med + 5);

  return { median: med, mad, robust_std: robustStd, threshold, baseline: med };
}

/**
 * Detect indices where values exceed threshold.
 */
function detectSpikes(values, threshold, minAbsolute = 5.0) {
  return values.map((v, i) => (v > threshold && v >= minAbsolute ? i : -1)).filter(i => i !== -1);
}

/**
 * Group consecutive/nearby spikes into events.
 */
function groupSpikesIntoEvents(spikeIndices, maxGap = 3, nTotal = null) {
  if (spikeIndices.length === 0) return [];

  let adjustedMaxGap = maxGap;
  if (nTotal !== null) {
    if (nTotal < 20) {
      adjustedMaxGap = 1;
    } else if (nTotal < 30) {
      adjustedMaxGap = Math.min(maxGap, 2);
    }
  }

  const events = [];
  let current = { start: spikeIndices[0], end: spikeIndices[0] };

  for (let i = 1; i < spikeIndices.length; i++) {
    const idx = spikeIndices[i];
    if (idx - current.end <= adjustedMaxGap) {
      current.end = idx;
    } else {
      current.duration = current.end - current.start + 1;
      events.push(current);
      current = { start: idx, end: idx };
    }
  }

  current.duration = current.end - current.start + 1;
  events.push(current);

  return events;
}

/**
 * Compute gaps between consecutive events.
 */
function computeEventGaps(events) {
  if (events.length < 2) return [];
  return events.slice(1).map((e, i) => e.start - events[i].end);
}

/**
 * Comprehensive trend analysis using multiple methods.
 */
function analyzeTrend(values, windowSize = null) {
  const n = values.length;
  if (n < 14) {
    return { direction: 'none', strength: 0, evidence: { reason: 'insufficient_data' } };
  }

  if (windowSize === null) {
    windowSize = Math.max(7, Math.floor(n / 6));
  }

  const evidence = {};
  let upVotes = 0;
  let downVotes = 0;

  // Method 1: Linear regression
  const x = Array.from({ length: n }, (_, i) => i);
  const [slope, intercept] = polyfit(x, values);

  const predicted = x.map(xi => slope * xi + intercept);
  const ssRes = values.reduce((acc, v, i) => acc + Math.pow(v - predicted[i], 2), 0);
  const meanVal = mean(values);
  const ssTot = values.reduce((acc, v) => acc + Math.pow(v - meanVal, 2), 0);
  let rSquared = ssTot > 0 ? 1 - ssRes / ssTot : 0;
  if (isNaN(rSquared)) rSquared = 0;

  const slopePct = meanVal > 0 ? (slope / meanVal) * 100 : 0;

  evidence.slope_pct_per_day = Math.round(slopePct * 10000) / 10000;
  evidence.r_squared = Math.round(rSquared * 1000) / 1000;

  if (slopePct > 0.2 && rSquared > 0.15) {
    upVotes += 1;
    if (rSquared > 0.3) upVotes += 1;
  } else if (slopePct < -0.2 && rSquared > 0.15) {
    downVotes += 1;
    if (rSquared > 0.3) downVotes += 1;
  }

  // Method 2: Higher highs / higher lows
  const quarter = Math.floor(n / 4);
  if (quarter >= 5) {
    const q1 = values.slice(0, quarter);
    const q4 = values.slice(-quarter);

    const q1Max = Math.max(...q1), q4Max = Math.max(...q4);
    const q1Min = Math.min(...q1), q4Min = Math.min(...q4);
    const q1Median = median(q1), q4Median = median(q4);

    evidence.q1_max = Math.round(q1Max * 10) / 10;
    evidence.q4_max = Math.round(q4Max * 10) / 10;
    evidence.q1_min = Math.round(q1Min * 10) / 10;
    evidence.q4_min = Math.round(q4Min * 10) / 10;

    const higherHighs = q4Max > q1Max * 1.15;
    const higherLows = q4Min > q1Min * 1.15;
    const higherMedian = q4Median > q1Median * 1.2;

    const lowerHighs = q4Max < q1Max * 0.85;
    const lowerLows = q4Min < q1Min * 0.85;
    const lowerMedian = q4Median < q1Median * 0.8;

    evidence.higher_highs = higherHighs;
    evidence.higher_lows = higherLows;
    evidence.lower_highs = lowerHighs;
    evidence.lower_lows = lowerLows;

    if (higherHighs && higherLows) {
      upVotes += 2;
    } else if (higherHighs || higherMedian) {
      upVotes += 1;
    }

    if (lowerHighs && lowerLows) {
      downVotes += 2;
    } else if (lowerHighs || lowerMedian) {
      downVotes += 1;
    }
  }

  // Method 3: Segment means comparison
  const third = Math.floor(n / 3);
  if (third >= 5) {
    const firstThird = values.slice(0, third);
    const lastThird = values.slice(-third);

    const firstMean = mean(firstThird);
    const lastMean = mean(lastThird);

    evidence.first_third_mean = Math.round(firstMean * 10) / 10;
    evidence.last_third_mean = Math.round(lastMean * 10) / 10;

    if (lastMean > firstMean * 1.25) {
      upVotes += 1;
    } else if (lastMean < firstMean * 0.75) {
      downVotes += 1;
    }
  }

  // Method 4: Rolling window comparison
  if (n >= 28) {
    const firstWindow = values.slice(0, windowSize);
    const lastWindow = values.slice(-windowSize);

    const firstWMean = mean(firstWindow);
    const lastWMean = mean(lastWindow);

    const changeRatio = lastWMean / (firstWMean + 0.01);
    evidence.window_change_ratio = Math.round(changeRatio * 100) / 100;

    if (changeRatio > 1.4) {
      upVotes += 1;
    } else if (changeRatio < 0.7) {
      downVotes += 1;
    }
  }

  evidence.up_votes = upVotes;
  evidence.down_votes = downVotes;

  let direction, strength;
  if (upVotes >= 3 && upVotes > downVotes + 1) {
    direction = 'up';
    strength = Math.min(1.0, upVotes / 5);
  } else if (downVotes >= 3 && downVotes > upVotes + 1) {
    direction = 'down';
    strength = Math.min(1.0, downVotes / 5);
  } else {
    direction = 'none';
    strength = 0;
  }

  return { direction, strength, evidence };
}

/**
 * Analyze periodicity using gap analysis, FFT, and autocorrelation.
 */
function analyzePeriodicity(values, events, minPeriod = 4, maxPeriod = 45) {
  const n = values.length;
  const results = { detected: false, period: null, confidence: 'Low', evidence: {} };

  // Method 1: Gap analysis
  if (events.length >= 3) {
    const gaps = computeEventGaps(events);
    if (gaps.length >= 2) {
      const gapMean = mean(gaps);
      const gapStd = std(gaps);
      const gapCv = gapStd / (gapMean + 0.01);

      results.evidence.gap_mean = Math.round(gapMean * 10) / 10;
      results.evidence.gap_std = Math.round(gapStd * 10) / 10;
      results.evidence.gap_cv = Math.round(gapCv * 100) / 100;
      results.evidence.num_events = events.length;
      results.evidence.num_gaps = gaps.length;

      if (gapMean >= minPeriod && gapMean <= maxPeriod) {
        if (gapCv < 0.35) {
          results.detected = true;
          results.period = gapMean;
          results.confidence = 'High';
          results.method = 'gap_analysis';
          return results;
        } else if (gapCv < 0.5 && events.length >= 4) {
          results.detected = true;
          results.period = gapMean;
          results.confidence = 'Medium';
          results.method = 'gap_analysis';
          return results;
        }
      }
    }
  }

  // Method 2: FFT-based detection
  if (n >= minPeriod * 4) {
    try {
      const meanVal = mean(values);
      const centered = values.map(v => v - meanVal);
      const window = hanning(n);
      const windowed = centered.map((v, i) => v * window[i]);

      const fftVals = rfft(windowed);
      const freqs = rfftfreq(n, 1.0);
      const power = fftVals.map(c => c.re * c.re + c.im * c.im);

      const periods = freqs.map(f => (f > 0 ? 1.0 / f : Infinity));

      const validIndices = [];
      for (let i = 0; i < periods.length; i++) {
        if (periods[i] >= minPeriod && periods[i] <= maxPeriod && periods[i] < n / 3) {
          validIndices.push(i);
        }
      }

      if (validIndices.length > 0) {
        const validPower = validIndices.map(i => power[i]);
        const validPeriods = validIndices.map(i => periods[i]);

        let maxIdx = 0;
        for (let i = 1; i < validPower.length; i++) {
          if (validPower[i] > validPower[maxIdx]) maxIdx = i;
        }

        const bestPeriod = validPeriods[maxIdx];
        const bestPower = validPower[maxIdx];

        const otherPower = validPower.filter((_, i) => i !== maxIdx);
        let snr = 1.0;
        if (otherPower.length > 2) {
          const noiseLevel = median(otherPower);
          snr = noiseLevel > 0 ? bestPower / noiseLevel : 1.0;
        }

        results.evidence.fft_period = Math.round(bestPeriod * 10) / 10;
        results.evidence.fft_snr = Math.round(snr * 100) / 100;

        if (snr > 6.0) {
          results.detected = true;
          results.period = bestPeriod;
          results.confidence = snr > 10.0 ? 'High' : 'Medium';
          results.method = 'fft';
          return results;
        }
      }
    } catch (e) {
      results.evidence.fft_error = e.message;
    }
  }

  // Method 3: Autocorrelation
  if (n >= 35 && events.length >= 2) {
    let bestAutocorr = 0;
    let bestPeriod = null;

    for (const testPeriod of [7, 14, 21, 28]) {
      if (testPeriod < Math.floor(n / 3)) {
        try {
          const x1 = values.slice(0, -testPeriod);
          const x2 = values.slice(testPeriod);

          if (std(x1) > 0.01 && std(x2) > 0.01) {
            const autocorr = corrcoef(x1, x2);

            if (!isNaN(autocorr)) {
              results.evidence[`autocorr_${testPeriod}d`] = Math.round(autocorr * 1000) / 1000;

              if (autocorr > bestAutocorr) {
                bestAutocorr = autocorr;
                bestPeriod = testPeriod;
              }
            }
          }
        } catch (e) {
          // ignore
        }
      }
    }

    if (bestAutocorr > 0.5 && events.length >= 3) {
      results.detected = true;
      results.period = bestPeriod;
      results.confidence = bestAutocorr > 0.65 ? 'Medium' : 'Low';
      results.method = 'autocorrelation';
      return results;
    }
  }

  return results;
}

/**
 * Determine if the time series is truly stable.
 */
function analyzeStability(values, threshold, spikeIndices) {
  const n = values.length;

  if (n < 14) {
    return { is_stable: false, score: 0.3, reason: 'insufficient_data' };
  }

  const evidence = {};
  let stabilityScore = 1.0;

  const spikeRate = spikeIndices.length / n;
  evidence.spike_rate = Math.round(spikeRate * 1000) / 1000;

  if (spikeRate > 0.15) {
    stabilityScore -= 0.4;
    evidence.spike_issue = 'high';
  } else if (spikeRate > 0.08) {
    stabilityScore -= 0.2;
    evidence.spike_issue = 'moderate';
  }

  const meanVal = mean(values);
  const stdVal = std(values);
  const cv = stdVal / (meanVal + 0.01);
  evidence.cv = Math.round(cv * 1000) / 1000;

  if (cv > 0.6) {
    stabilityScore -= 0.3;
    evidence.cv_issue = 'high';
  } else if (cv > 0.4) {
    stabilityScore -= 0.15;
    evidence.cv_issue = 'moderate';
  }

  if (n >= 28) {
    const firstHalf = mean(values.slice(0, Math.floor(n / 2)));
    const secondHalf = mean(values.slice(Math.floor(n / 2)));
    const changeRatio = secondHalf / (firstHalf + 0.01);
    evidence.half_change_ratio = Math.round(changeRatio * 100) / 100;

    if (changeRatio > 1.3 || changeRatio < 0.7) {
      stabilityScore -= 0.25;
      evidence.trend_issue = true;
    }
  }

  const isStable = stabilityScore >= 0.5 && spikeRate < 0.1;

  return { is_stable: isStable, score: Math.max(0, stabilityScore), evidence };
}

/**
 * Master classification function - PRODUCTION VERSION.
 */
function classify(values, tier = 'MEDIUM') {
  const n = values.length;

  if (n < 7) {
    return {
      state: State.STABLE,
      confidence: 'Very Low',
      confidence_score: 0.3,
      spike_count: 0,
      is_recurring: false,
      period_days: null,
      trend_direction: null,
      evidence: { note: 'Insufficient data (< 7 days)' }
    };
  }

  // Compute metrics
  const stats = computeRobustStatistics(values);
  const baseline = stats.baseline;
  const threshold = stats.threshold;

  const spikeIndices = detectSpikes(values, threshold);
  const spikeEvents = groupSpikesIntoEvents(spikeIndices, 3, n);
  const numSpikes = spikeIndices.length;
  const numEvents = spikeEvents.length;

  const trend = analyzeTrend(values);

  let periodicity;
  if (numEvents >= 3) {
    periodicity = analyzePeriodicity(values, spikeEvents);
  } else {
    periodicity = { detected: false, period: null, confidence: 'Low', evidence: {} };
  }

  const stability = analyzeStability(values, threshold, spikeIndices);

  const recentWindow = Math.min(7, Math.max(3, Math.floor(n / 10)));
  const recentIndices = spikeIndices.filter(i => i >= n - recentWindow);
  const hasRecentSpike = recentIndices.length > 0;

  let recentMax = 0;
  let spikeSeverity = 0;
  if (hasRecentSpike) {
    recentMax = Math.max(...recentIndices.map(i => values[i]));
    spikeSeverity = (recentMax - baseline) / Math.max(baseline, 1);
  }

  const baseEvidence = {
    baseline: Math.round(baseline * 10) / 10,
    threshold: Math.round(threshold * 10) / 10,
    total_spikes: numSpikes,
    spike_events: numEvents
  };

  // Early recurring check
  const minEventsForRecurring = n < 20 ? 2 : 3;

  if (numEvents >= minEventsForRecurring) {
    const gaps = computeEventGaps(spikeEvents);
    if (gaps.length >= 1) {
      const gapMean = mean(gaps);
      const gapStd = gaps.length > 1 ? std(gaps) : 0;
      const gapCv = gaps.length > 1 ? gapStd / (gapMean + 0.01) : 0;

      const spikeValues = spikeIndices.map(i => values[i]);
      const avgSpikeHeight = mean(spikeValues);
      const spikeLift = avgSpikeHeight / Math.max(baseline, 1);
      const spikeValueCv = std(spikeValues) / (mean(spikeValues) + 0.01);
      const spikesConsistent = spikeValueCv < 0.5;

      const avgSpikeAbsoluteDiff = avgSpikeHeight - baseline;
      const hasSignificantSpikes = spikeLift >= 2.2 || avgSpikeAbsoluteDiff >= 25;
      const hasVerySignificantSpikes = spikeLift >= 2.8 || avgSpikeAbsoluteDiff >= 35;

      let isValidRecurring;
      if (gaps.length === 1) {
        if (numSpikes >= 3) {
          isValidRecurring = gapMean >= 2 && gapMean <= n / 2 && hasSignificantSpikes && spikesConsistent;
        } else {
          isValidRecurring = gapMean >= 2 && gapMean <= n / 2 && hasVerySignificantSpikes && spikesConsistent;
        }
      } else {
        isValidRecurring = gapCv < 0.5 && gapMean >= 2 && hasSignificantSpikes && spikesConsistent;
      }

      if (isValidRecurring) {
        const quarter = Math.max(Math.floor(n / 4), 2);
        let baselineRising = false;

        if (n >= 30 && quarter >= 5) {
          const nonSpikeMask = values.map((_, i) => !spikeIndices.includes(i));
          const firstQVals = values.slice(0, quarter).filter((_, i) => nonSpikeMask[i]);
          const lastQVals = values.slice(-quarter).filter((_, i) => nonSpikeMask[n - quarter + i]);

          if (firstQVals.length >= 2 && lastQVals.length >= 2) {
            const firstQP25 = percentile(firstQVals, 25);
            const lastQP25 = percentile(lastQVals, 25);
            baselineRising = lastQP25 > firstQP25 * 1.3;
          }
        }

        if (!baselineRising) {
          const confidence = gaps.length === 1 ? 'Medium' : (gapCv < 0.3 ? 'High' : 'Medium');
          return {
            state: State.RECURRING,
            confidence,
            confidence_score: Math.min(0.9, 0.55 + numEvents * 0.07),
            spike_count: numSpikes,
            is_recurring: true,
            period_days: gapMean,
            trend_direction: null,
            evidence: {
              ...baseEvidence,
              gap_mean: Math.round(gapMean * 10) / 10,
              gap_cv: Math.round(gapCv * 100) / 100,
              spike_lift: Math.round(spikeLift * 100) / 100,
              detection_method: 'consistent_gaps'
            }
          };
        }
      }
    }
  }

  // Critical check (after early recurring)
  if (hasRecentSpike && spikeSeverity >= 1.5) {
    if (n > recentWindow + 14) {
      const history = values.slice(0, n - recentWindow);
      const historySpikes = spikeIndices.filter(i => i < n - recentWindow);
      const historySpikeRate = historySpikes.length / history.length;
      const historyCv = std(history) / (mean(history) + 0.01);

      const historyWasStable = historySpikeRate < 0.08 && historyCv < 0.5;

      if (historyWasStable) {
        return {
          state: State.CRITICAL,
          confidence: spikeSeverity >= 2.5 ? 'High' : 'Medium',
          confidence_score: Math.min(0.95, 0.6 + spikeSeverity * 0.12),
          spike_count: numSpikes,
          is_recurring: false,
          period_days: null,
          trend_direction: null,
          evidence: {
            ...baseEvidence,
            recent_spike_value: Math.round(recentMax * 10) / 10,
            spike_severity: `${spikeSeverity.toFixed(1)}x baseline`,
            history_spike_rate: `${(historySpikeRate * 100).toFixed(1)}%`,
            history_cv: Math.round(historyCv * 100) / 100,
            note: 'Sudden spike after stable history'
          }
        };
      }
    }
  }

  // Emerging check
  if (n >= 30) {
    const cutoff = Math.floor(n * 2 / 3);
    const firstPortion = values.slice(0, cutoff);
    const lastPortion = values.slice(cutoff);

    const firstMean = mean(firstPortion);
    const lastMean = mean(lastPortion);
    const firstMax = Math.max(...firstPortion);

    if (firstMean < 8 && lastMean > 20 && lastMean > firstMean * 3 && firstMax < threshold * 0.8) {
      return {
        state: State.EMERGING,
        confidence: lastMean > firstMean * 5 ? 'High' : 'Medium',
        confidence_score: 0.8,
        spike_count: numSpikes,
        is_recurring: false,
        period_days: null,
        trend_direction: 'up',
        evidence: {
          ...baseEvidence,
          first_portion_mean: Math.round(firstMean * 10) / 10,
          last_portion_mean: Math.round(lastMean * 10) / 10,
          ratio: Math.round((lastMean / Math.max(firstMean, 0.1)) * 10) / 10
        }
      };
    }
  }

  // Baseline stability analysis
  let baselineStable = true;
  let baselineChange = 0;

  const quarter = Math.floor(n / 4);
  if (quarter >= 5) {
    const firstQP25 = percentile(values.slice(0, quarter), 25);
    const lastQP25 = percentile(values.slice(-quarter), 25);
    if (firstQP25 > 0.5) {
      baselineChange = (lastQP25 - firstQP25) / firstQP25;
    } else {
      baselineChange = lastQP25 - firstQP25;
    }
    baselineStable = baselineChange < 0.25;
  }

  // Second recurring check (with periodicity detection)
  const baselineCv = std(values) / (mean(values) + 0.01);

  let spikeLift = 1.0;
  let spikesConsistent = false;
  let hasSignificantSpikes = false;

  if (spikeIndices.length > 0) {
    const spikeValues = spikeIndices.map(i => values[i]);
    const avgSpikeHeight = mean(spikeValues);
    spikeLift = avgSpikeHeight / Math.max(baseline, 1);
    const spikeValueCv = std(spikeValues) / (mean(spikeValues) + 0.01);
    spikesConsistent = spikeValueCv < 0.5;
    const avgSpikeAbsoluteDiff = avgSpikeHeight - baseline;
    hasSignificantSpikes = spikeLift >= 2.2 || avgSpikeAbsoluteDiff >= 25;
  }

  if (periodicity.detected && baselineStable && baselineCv < 0.45 && hasSignificantSpikes && spikesConsistent) {
    return {
      state: State.RECURRING,
      confidence: periodicity.confidence || 'Medium',
      confidence_score: Math.min(0.92, 0.55 + numEvents * 0.06),
      spike_count: numSpikes,
      is_recurring: true,
      period_days: periodicity.period,
      trend_direction: null,
      evidence: {
        ...baseEvidence,
        periodicity: periodicity.evidence || {},
        baseline_stable: true,
        baseline_change_pct: Math.round(baselineChange * 1000) / 10,
        spike_lift: Math.round(spikeLift * 100) / 100
      }
    };
  }

  // Worsening check
  if (!baselineStable && (periodicity.detected || numEvents >= 2)) {
    return {
      state: State.WORSENING,
      confidence: baselineChange > 0.4 ? 'High' : 'Medium',
      confidence_score: Math.min(0.9, 0.55 + baselineChange * 0.5),
      spike_count: numSpikes,
      is_recurring: periodicity.detected,
      period_days: periodicity.period,
      trend_direction: 'up',
      evidence: {
        ...baseEvidence,
        baseline_change_pct: Math.round(baselineChange * 1000) / 10,
        note: periodicity.detected ? 'Rising baseline with periodic spikes' : 'Rising baseline with spikes'
      }
    };
  }

  if (trend.direction === 'up' && trend.strength >= 0.4) {
    let lowsRising;
    if (quarter >= 5) {
      const firstQP25 = percentile(values.slice(0, quarter), 25);
      const lastQP25 = percentile(values.slice(-quarter), 25);
      lowsRising = lastQP25 > firstQP25 * 1.15;
    } else {
      lowsRising = trend.evidence.higher_lows || false;
    }

    if (lowsRising || trend.strength >= 0.6) {
      return {
        state: State.WORSENING,
        confidence: trend.strength >= 0.6 ? 'High' : 'Medium',
        confidence_score: Math.min(0.92, 0.5 + trend.strength * 0.4),
        spike_count: numSpikes,
        is_recurring: false,
        period_days: null,
        trend_direction: 'up',
        evidence: { ...baseEvidence, trend: trend.evidence }
      };
    }
  }

  // Critical check (no pattern)
  if (hasRecentSpike && spikeSeverity >= 1.0) {
    const historicalSpikes = spikeIndices.filter(i => i < n - recentWindow);
    const historicalSpikeRate = historicalSpikes.length / Math.max(n - recentWindow, 1);

    const isTrulyUnexpected = historicalSpikeRate < 0.15 && spikeSeverity >= 1.5;

    if (isTrulyUnexpected) {
      return {
        state: State.CRITICAL,
        confidence: spikeSeverity >= 2.5 ? 'High' : 'Medium',
        confidence_score: Math.min(0.95, 0.6 + spikeSeverity * 0.12),
        spike_count: numSpikes,
        is_recurring: false,
        period_days: null,
        trend_direction: null,
        evidence: {
          ...baseEvidence,
          recent_spike_value: Math.round(recentMax * 10) / 10,
          spike_severity: `${spikeSeverity.toFixed(1)}x baseline`,
          historical_spike_rate: `${(historicalSpikeRate * 100).toFixed(1)}%`,
          note: 'Unexpected spike - history was stable'
        }
      };
    }
  }

  // Improving check
  if (trend.direction === 'down' && trend.strength >= 0.4) {
    const quarterSize = Math.max(Math.floor(n / 4), 7);
    const firstPortionMean = mean(values.slice(0, quarterSize));
    const lastPortionMean = mean(values.slice(-quarterSize));

    const wasElevated = firstPortionMean > Math.max(threshold * 0.6, lastPortionMean * 1.5);

    if (wasElevated && lastPortionMean < firstPortionMean * 0.65) {
      return {
        state: State.IMPROVING,
        confidence: trend.strength >= 0.6 ? 'High' : 'Medium',
        confidence_score: Math.min(0.88, 0.5 + trend.strength * 0.4),
        spike_count: numSpikes,
        is_recurring: false,
        period_days: null,
        trend_direction: 'down',
        evidence: {
          ...baseEvidence,
          trend: trend.evidence,
          first_portion_mean: Math.round(firstPortionMean * 10) / 10,
          last_portion_mean: Math.round(lastPortionMean * 10) / 10
        }
      };
    }
  }

  // Resolved check
  if (n >= 45) {
    const thirdSize = Math.floor(n / 3);
    const middle = values.slice(thirdSize, 2 * thirdSize);
    const recent = values.slice(2 * thirdSize);

    const middleMean = mean(middle);
    const recentMean = mean(recent);
    const recentMaxVal = Math.max(...recent);

    if (middleMean > threshold && recentMean < baseline * 0.8 && recentMaxVal < threshold) {
      return {
        state: State.RESOLVED,
        confidence: recentMaxVal < baseline * 0.7 ? 'High' : 'Medium',
        confidence_score: 0.82,
        spike_count: numSpikes,
        is_recurring: false,
        period_days: null,
        trend_direction: null,
        evidence: {
          ...baseEvidence,
          middle_period_mean: Math.round(middleMean * 10) / 10,
          recent_mean: Math.round(recentMean * 10) / 10
        }
      };
    }
  }

  // Stable check
  if (stability.is_stable) {
    return {
      state: State.STABLE,
      confidence: stability.score > 0.7 ? 'High' : 'Medium',
      confidence_score: Math.min(0.85, 0.5 + stability.score * 0.4),
      spike_count: numSpikes,
      is_recurring: false,
      period_days: null,
      trend_direction: null,
      evidence: { ...baseEvidence, stability: stability.evidence || {} }
    };
  }

  // Fallback
  if (numSpikes > 0 || numEvents > 0) {
    return {
      state: State.STABLE,
      confidence: 'Low',
      confidence_score: 0.45,
      spike_count: numSpikes,
      is_recurring: false,
      period_days: null,
      trend_direction: null,
      evidence: {
        ...baseEvidence,
        note: 'Some activity detected but no clear pattern',
        stability: stability.evidence || {}
      }
    };
  }

  return {
    state: State.STABLE,
    confidence: 'Medium',
    confidence_score: 0.7,
    spike_count: 0,
    is_recurring: false,
    period_days: null,
    trend_direction: null,
    evidence: { ...baseEvidence, status: 'No significant issues detected' }
  };
}

export {
  State,
  classify,
  computeRobustStatistics,
  detectSpikes,
  groupSpikesIntoEvents,
  computeEventGaps,
  analyzeTrend,
  analyzePeriodicity,
  analyzeStability,
  mean,
  median,
  std,
  percentile
};
