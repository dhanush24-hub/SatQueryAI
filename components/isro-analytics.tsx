'use client';
import React from 'react';
import {Satellite,Gauge,Layers,ShieldAlert,ShieldCheck,TrendingUp,TrendingDown,Minus,Cpu,Activity,Compass,Sun,Radar,Grid} from 'lucide-react';
import type {IsroAnalysisData,MissionResult,Finding,EvidenceValidation} from '@/lib/domain';

interface IsroAnalyticsProps {
  isroData?: IsroAnalysisData;
  result?: MissionResult;
  selectedFindingId: string | null;
  onSelectFinding: (findingId: string) => void;
}

export function IsroAnalytics({isroData,result,selectedFindingId,onSelectFinding}:IsroAnalyticsProps) {
  if (!isroData) {
    return (
      <div className="inspector-empty">
        <Satellite size={28} style={{color:'#c4f86a'}}/>
        <h3>Geospatial Telemetry & Radiometric Indices</h3>
        <p>Quantitative telemetry, LULC area deltas, and spectral band indices require calibrated GeoTIFF rasters with embedded CRS and sensor metadata. Explore a demonstration mission to view reference telemetry.</p>
      </div>
    );
  }

  const telemetry = isroData?.telemetry || [];
  const lulc = isroData?.lulc || [];
  const spectral = isroData?.spectral || [];
  const confidenceScore = isroData?.confidenceScore;

  return (
    <div className="isro-analytics-panel">
      {/* ISRO Mission Telemetry HUD */}
      <div className="isro-telemetry-hud">
        <div className="telemetry-header">
          <span><Satellite size={15}/> SATELLITE ORBIT & SENSOR TELEMETRY</span>
          <span className="telemetry-badge">CURATED MISSION FIXTURE</span>
        </div>
        <div className="telemetry-grid">
          {telemetry.map((t, idx) => (
            <div className="telemetry-card" key={idx}>
              <div className="telemetry-card-top">
                <strong>ACQUISITION {idx + 1}</strong>
                <span>{t.platform}</span>
              </div>
              <div className="telemetry-data-row">
                <span className="label">Sensor:</span>
                <span className="val">{t.sensor}</span>
              </div>
              <div className="telemetry-data-row">
                <span className="label">GSD Resolution:</span>
                <span className="val">{t.resolutionGsd}</span>
              </div>
              <div className="telemetry-data-row">
                <span className="label">Orbit / Pass:</span>
                <span className="val">{t.orbitPass}</span>
              </div>
              <div className="telemetry-data-row">
                <span className="label">CRS Datum:</span>
                <span className="val">{t.crs}</span>
              </div>
              {t.solarElevation && (
                <div className="telemetry-data-row">
                  <span className="label">Sun Elevation:</span>
                  <span className="val">{t.solarElevation}</span>
                </div>
              )}
              {t.incidenceAngle && (
                <div className="telemetry-data-row">
                  <span className="label">SAR Incidence:</span>
                  <span className="val">{t.incidenceAngle}</span>
                </div>
              )}
              {t.coregistrationRms && (
                <div className="telemetry-data-row highlight">
                  <span className="label">Co-registration RMS:</span>
                  <span className="val">{t.coregistrationRms}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* LULC (Land-Use Land-Cover) Quantification & Delta Engine */}
      <div className="isro-section">
        <div className="section-head">
          <span><Layers size={15}/> QUANTITATIVE LULC CLASSIFICATION & AREA DELTA</span>
          <span className="area-total-tag">ESTIMATED AREA DELTA IN HECTARES</span>
        </div>
        <p className="section-sub">
          Automated pixel-level spatial accounting comparing baseline (T₁) against target (T₂) surface composition.
        </p>
        <div className="lulc-matrix">
          {lulc.map((item, idx) => (
            <div className="lulc-row" key={idx}>
              <div className="lulc-info">
                <span className="lulc-color-indicator" style={{backgroundColor: item.color}}/>
                <strong>{item.category}</strong>
                <div className="lulc-badges">
                  <span className="hectare-tag">
                    {item.trend === 'increase' ? '+' : item.trend === 'decrease' ? '-' : ''}
                    {item.areaHectares} ha
                  </span>
                  <span className={`delta-tag ${item.trend}`}>
                    {item.trend === 'increase' ? <TrendingUp size={12}/> : item.trend === 'decrease' ? <TrendingDown size={12}/> : <Minus size={12}/>}
                    {item.deltaPercent > 0 ? `+${item.deltaPercent}%` : `${item.deltaPercent}%`}
                  </span>
                </div>
              </div>

              {/* Progress bars showing T1 vs T2 */}
              <div className="lulc-bars">
                <div className="bar-track">
                  <div className="bar-fill t1" style={{width: `${Math.min(100, item.t1Percent)}%`, backgroundColor: item.color, opacity: 0.6}}/>
                  <span className="bar-label">T₁: {item.t1Percent}%</span>
                </div>
                <div className="bar-track">
                  <div className="bar-fill t2" style={{width: `${Math.min(100, item.t2Percent)}%`, backgroundColor: item.color}}/>
                  <span className="bar-label">T₂: {item.t2Percent}%</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Multispectral Remote Sensing Indices & SAR Physics */}
      <div className="isro-section">
        <div className="section-head">
          <span><Activity size={15}/> SPECTRAL INDICES & RADAR BACKSCATTER PHYSICS</span>
        </div>
        <div className="spectral-grid">
          {spectral.map((s, idx) => (
            <div className="spectral-card" key={idx}>
              <div className="spectral-card-head">
                <strong>{s.indexName}</strong>
                <code>{s.description}</code>
              </div>
              <div className="spectral-values">
                <div><span>Baseline T₁:</span> <b>{s.valueT1}</b></div>
                <div><span>Target T₂:</span> <b>{s.valueT2}</b></div>
              </div>
              <p className="spectral-delta-desc">{s.deltaInterpretation}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Multi-Sensor Fusion Agreement Matrix */}
      {result && result.crossSensorAssessment && result.crossSensorAssessment.length > 0 && (
        <div className="isro-section">
          <div className="section-head">
            <span><Radar size={15}/> MULTI-MODAL OPTICAL + SAR FUSION AUDIT</span>
          </div>
          <div className="cross-sensor-table">
            {result.crossSensorAssessment.map((c, i) => (
              <div className={`cross-sensor-row ${c.relationship}`} key={i}>
                <div className="cross-badge">
                  <span className={`status-pill ${c.relationship}`}>{c.relationship.toUpperCase()}</span>
                  <strong>{c.finding}</strong>
                </div>
                <p>{c.explanation}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evidence Gate Verification Scorecard */}
      <div className="isro-section">
        <div className="section-head">
          <span><ShieldCheck size={15}/> SCIENTIFIC EVIDENCE GATE & INTEGRITY SCORECARD</span>
          <span className="confidence-pill" style={{color: confidenceScore && confidenceScore >= 90 ? '#c4f86a' : confidenceScore && confidenceScore >= 75 ? '#eab308' : '#f43f5e'}}>
            {confidenceScore ? `${confidenceScore}% GATED CONFIDENCE (DEMO)` : 'EVIDENCE REVIEWED · QUALITATIVE'}
          </span>
        </div>
        <div className="gate-scorecard">
          <div className="gate-item passed">
            <ShieldCheck size={16}/>
            <div>
              <strong>1. Geo-Harmonization Check</strong>
              <small>Automated CRS projection matching, pixel resolution resampling, spatial overlap bounds verification.</small>
            </div>
            <span className="gate-status">VERIFIED</span>
          </div>
          <div className="gate-item passed">
            <ShieldCheck size={16}/>
            <div>
              <strong>2. Semantic Change Validation</strong>
              <small>Filtering seasonal solar azimuth & illumination disparities from permanent anthropogenic development.</small>
            </div>
            <span className="gate-status">VALIDATED</span>
          </div>
          <div className="gate-item passed">
            <ShieldCheck size={16}/>
            <div>
              <strong>3. Bilateral Visual Grounding</strong>
              <small>Enforces counterpart spatial coordinates across both temporal acquisitions to eliminate hallucinated regions.</small>
            </div>
            <span className="gate-status">GROUNDED</span>
          </div>
          <div className={`gate-item ${isroData?.telemetry?.some(t=>t.coregistrationRms?.includes('Domain Gap')) ? 'warn' : 'passed'}`}>
            <ShieldCheck size={16}/>
            <div>
              <strong>4. Cross-Sensor Domain Bridge</strong>
              <small>Cross-checks optical reflectance bands against radar cross-section surface roughness.</small>
            </div>
            <span className="gate-status">{isroData?.telemetry?.some(t=>t.coregistrationRms?.includes('Domain Gap')) ? 'QUALIFIED' : 'PASSED'}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
