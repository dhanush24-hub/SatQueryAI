'use client';

import React from 'react';
import {
  ShieldCheck,
  AlertTriangle,
  FileText,
  Download,
  Clock,
  Layers,
  CheckCircle2,
  XCircle,
  ExternalLink,
  ChevronRight,
  Info,
  Scale,
  Cpu,
  Compass,
  FileCode
} from 'lucide-react';
import type { AnalysisResponse, FindingItemDTO } from '@/lib/api/analysis';

interface EvidenceDossierProps {
  analysis: AnalysisResponse;
  selectedFindingId: string | null;
  onSelectFinding: (findingId: string) => void;
}

export function EvidenceDossier({
  analysis,
  selectedFindingId,
  onSelectFinding,
}: EvidenceDossierProps) {
  const isGeo = analysis.overlays?.georeferenced ?? false;
  const artifacts = analysis.downloadable_artifacts;

  // Determine badge styling based on explicit evidence state
  const getBadgeStyle = (state?: string) => {
    switch (state) {
      case 'SUPPORTED':
        return { bg: '#10b98118', border: '#10b98150', text: '#34d399', label: 'SUPPORTED' };
      case 'SUPPORTED_WITH_WARNINGS':
        return { bg: '#f59e0b18', border: '#f59e0b50', text: '#fbbf24', label: 'SUPPORTED WITH WARNINGS' };
      case 'WEAK':
        return { bg: '#eab30818', border: '#eab30850', text: '#fde047', label: 'WEAK EVIDENCE' };
      case 'INSUFFICIENT_EVIDENCE':
        return { bg: '#f43f5e18', border: '#f43f5e50', text: '#fb7185', label: 'INSUFFICIENT EVIDENCE' };
      case 'OUT_OF_DOMAIN':
        return { bg: '#a855f718', border: '#a855f750', text: '#c084fc', label: 'OUT OF DOMAIN' };
      default:
        return { bg: '#64748b18', border: '#64748b50', text: '#94a3b8', label: state || 'UNAVAILABLE' };
    }
  };

  const badge = getBadgeStyle(analysis.evidence_state);

  return (
    <div className="evidence-dossier" style={{ display: 'flex', flexDirection: 'column', gap: '16px', padding: '12px' }}>
      {/* 1. Header: State Badge & Analysis ID */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              padding: '4px 10px',
              borderRadius: '4px',
              fontSize: '11px',
              fontWeight: 700,
              letterSpacing: '0.05em',
              background: badge.bg,
              border: `1px solid ${badge.border}`,
              color: badge.text,
              display: 'inline-flex',
              alignItems: 'center',
              gap: '5px'
            }}
          >
            <ShieldCheck size={13} /> {badge.label}
          </span>
          <span style={{ fontSize: '11px', color: '#64748b', fontFamily: 'monospace' }}>
            {analysis.id}
          </span>
        </div>
        <span style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {analysis.task}
        </span>
      </div>

      {/* 2. Direct Answer Block */}
      <div
        style={{
          background: 'rgba(15, 23, 42, 0.65)',
          border: '1px solid rgba(196, 248, 106, 0.25)',
          borderRadius: '8px',
          padding: '14px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
        }}
      >
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#c4f86a', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Compass size={13} /> Analytical Synthesis
        </div>
        <p style={{ margin: 0, fontSize: '13px', lineHeight: '1.55', color: '#f1f5f9' }}>
          {analysis.answer}
        </p>
      </div>

      {/* 3. Download & Export Action Bar */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
          gap: '8px',
          padding: '10px',
          background: 'rgba(255, 255, 255, 0.03)',
          border: '1px solid rgba(255, 255, 255, 0.06)',
          borderRadius: '8px'
        }}
      >
        {artifacts?.report_pdf_url && (
          <a
            href={artifacts.report_pdf_url}
            target="_blank"
            rel="noreferrer"
            download
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              background: '#c4f86a',
              color: '#091319',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: 700,
              textDecoration: 'none',
              cursor: 'pointer'
            }}
          >
            <FileText size={13} /> PDF Report
          </a>
        )}

        {artifacts?.result_json_url && (
          <a
            href={artifacts.result_json_url}
            target="_blank"
            rel="noreferrer"
            download
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              background: 'rgba(255,255,255,0.08)',
              color: '#e2e8f0',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: 600,
              textDecoration: 'none',
              cursor: 'pointer'
            }}
          >
            <FileCode size={13} /> Export JSON
          </a>
        )}

        {artifacts?.geojson_url ? (
          <a
            href={artifacts.geojson_url}
            target="_blank"
            rel="noreferrer"
            download
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              background: 'rgba(56, 189, 248, 0.15)',
              color: '#38bdf8',
              border: '1px solid rgba(56, 189, 248, 0.35)',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: 600,
              textDecoration: 'none',
              cursor: 'pointer'
            }}
          >
            <Download size={13} /> GeoJSON
          </a>
        ) : (
          <button
            disabled
            title="GeoJSON export strictly requires authentic georeferencing (zero-fabrication policy)"
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              background: 'rgba(255,255,255,0.03)',
              color: '#64748b',
              border: '1px solid rgba(255,255,255,0.05)',
              borderRadius: '6px',
              fontSize: '11px',
              cursor: 'not-allowed'
            }}
          >
            <Download size={13} /> GeoJSON (No CRS)
          </button>
        )}

        {artifacts?.mask_png_url && (
          <a
            href={artifacts.mask_png_url}
            target="_blank"
            rel="noreferrer"
            download
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '6px',
              padding: '8px 10px',
              background: 'rgba(255,255,255,0.08)',
              color: '#cbd5e1',
              border: '1px solid rgba(255,255,255,0.12)',
              borderRadius: '6px',
              fontSize: '11px',
              fontWeight: 600,
              textDecoration: 'none',
              cursor: 'pointer'
            }}
          >
            <Layers size={13} /> Mask PNG
          </a>
        )}
      </div>

      {/* 4. Quantitative & Spatial Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
          <span style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase' }}>Change Ratio</span>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc', marginTop: '2px' }}>
            {analysis.overlays?.change_ratio_pct !== undefined && analysis.overlays?.change_ratio_pct !== null
              ? `${analysis.overlays.change_ratio_pct.toFixed(2)}%`
              : 'N/A'}
          </div>
        </div>

        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
          <span style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase' }}>Surface Area</span>
          <div style={{ fontSize: '14px', fontWeight: 700, color: isGeo ? '#38bdf8' : '#94a3b8', marginTop: '2px' }}>
            {isGeo && analysis.findings?.[0]?.area_m2
              ? `${(analysis.findings[0].area_m2 / 10000).toFixed(2)} ha`
              : 'Omitted (No CRS)'}
          </div>
        </div>

        <div style={{ background: 'rgba(255,255,255,0.03)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
          <span style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase' }}>Changed Pixels</span>
          <div style={{ fontSize: '16px', fontWeight: 700, color: '#f8fafc', marginTop: '2px' }}>
            {analysis.overlays?.total_changed_pixels !== undefined && analysis.overlays?.total_changed_pixels !== null
              ? analysis.overlays.total_changed_pixels.toLocaleString()
              : '0'}
          </div>
        </div>
      </div>

      {/* 5. Key Findings List */}
      <div>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <Scale size={13} /> Delineated Region Findings ({analysis.findings?.length || 0})
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {analysis.findings && analysis.findings.length > 0 ? (
            analysis.findings.map((f: FindingItemDTO) => {
              const isSelected = selectedFindingId === f.finding_id;
              return (
                <div
                  key={f.finding_id}
                  onClick={() => onSelectFinding(f.finding_id)}
                  style={{
                    padding: '10px',
                    borderRadius: '6px',
                    background: isSelected ? 'rgba(196, 248, 106, 0.08)' : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${isSelected ? '#c4f86a' : 'rgba(255,255,255,0.07)'}`,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontSize: '12px', fontWeight: 600, color: '#f8fafc' }}>
                      {f.summary}
                    </span>
                    <span
                      style={{
                        fontSize: '9px',
                        padding: '2px 6px',
                        borderRadius: '3px',
                        fontWeight: 600,
                        background: f.evidence_quality === 'SUPPORTED' ? '#10b98120' : '#f59e0b20',
                        color: f.evidence_quality === 'SUPPORTED' ? '#34d399' : '#fbbf24'
                      }}
                    >
                      {f.evidence_quality}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: '12px', marginTop: '6px', fontSize: '11px', color: '#94a3b8' }}>
                    {f.bbox_px && (
                      <span>Box: [{f.bbox_px.join(', ')}] px</span>
                    )}
                    {f.area_m2 !== null && f.area_m2 !== undefined && (
                      <span style={{ color: '#38bdf8' }}>{(f.area_m2 / 10000).toFixed(2)} ha ({f.area_m2.toFixed(0)} m²)</span>
                    )}
                    <span style={{ fontStyle: 'italic' }}>{f.source_model}</span>
                  </div>

                  {f.limitations && f.limitations.length > 0 && (
                    <div style={{ fontSize: '10px', color: '#fbbf24', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <AlertTriangle size={10} /> {f.limitations[0]}
                    </div>
                  )}
                </div>
              );
            })
          ) : (
            <div style={{ padding: '12px', textAlign: 'center', color: '#64748b', fontSize: '12px', background: 'rgba(255,255,255,0.02)', borderRadius: '6px' }}>
              No discrete spatial change clusters identified exceeding the calibrated detection threshold.
            </div>
          )}
        </div>
      </div>

      {/* 6. Registration & Domain Suitability Audit Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
        {analysis.registration && (
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '4px' }}>
              Co-Registration
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: analysis.registration.status === 'HIGH' ? '#34d399' : '#fbbf24' }}>
              {analysis.registration.status === 'HIGH' ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
              {analysis.registration.status} ({analysis.registration.translation_px.toFixed(1)} px shift)
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px' }}>
              ZNCC Correlation: {analysis.registration.correlation.toFixed(3)}
            </div>
          </div>
        )}

        {analysis.domain_suitability && (
          <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '4px' }}>
              Domain Suitability
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', fontWeight: 600, color: analysis.domain_suitability.is_suitable ? '#34d399' : '#c084fc' }}>
              {analysis.domain_suitability.is_suitable ? <CheckCircle2 size={13} /> : <XCircle size={13} />}
              {analysis.domain_suitability.status}
            </div>
            <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px' }}>
              {analysis.domain_suitability.reasons[0] || 'Domain evaluated'}
            </div>
          </div>
        )}
      </div>

      {/* 7. Specialist Model Provenance */}
      {analysis.model_provenance && (
        <div style={{ background: 'rgba(255,255,255,0.02)', padding: '10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ fontSize: '10px', color: '#94a3b8', textTransform: 'uppercase', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <Cpu size={12} /> Model Provenance & Verification
          </div>
          <div style={{ fontSize: '12px', fontWeight: 600, color: '#e2e8f0' }}>
            {analysis.model_provenance.model_name}
          </div>
          <div style={{ fontSize: '10px', color: '#64748b', marginTop: '2px', fontFamily: 'monospace' }}>
            {analysis.model_provenance.architecture} · {analysis.model_provenance.parameters.toLocaleString()} params · τ = {analysis.model_provenance.threshold.toFixed(2)}
          </div>
          {analysis.model_provenance.benchmark_f1 !== null && (
            <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '4px' }}>
              Documented Benchmark: Val F1 = {analysis.model_provenance.benchmark_f1?.toFixed(4)} · IoU = {analysis.model_provenance.benchmark_iou?.toFixed(4)}
            </div>
          )}
        </div>
      )}

      {/* 8. Observable Execution Trace */}
      <div>
        <div style={{ fontSize: '11px', fontWeight: 600, color: '#94a3b8', textTransform: 'uppercase', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '5px' }}>
          <Clock size={13} /> Observable Execution Trace ({analysis.execution_summary?.length || 0} Steps)
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {analysis.execution_summary?.map((step, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '8px',
                padding: '6px 8px',
                background: 'rgba(255,255,255,0.02)',
                borderRadius: '4px',
                fontSize: '11px'
              }}
            >
              <span style={{ color: '#c4f86a', fontWeight: 700, fontSize: '10px', minWidth: '16px' }}>
                {idx + 1}.
              </span>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <strong style={{ color: '#e2e8f0' }}>{step.step}</strong>
                  {step.duration_ms !== null && step.duration_ms !== undefined && (
                    <span style={{ fontSize: '10px', color: '#64748b' }}>{step.duration_ms}ms</span>
                  )}
                </div>
                <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>
                  Tool: <code style={{ color: '#cbd5e1' }}>{step.tool_name}</code> · {step.summary}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 9. Scientific Limitations */}
      {analysis.limitations && analysis.limitations.length > 0 && (
        <div style={{ background: 'rgba(245, 158, 11, 0.04)', border: '1px solid rgba(245, 158, 11, 0.2)', padding: '10px', borderRadius: '6px' }}>
          <div style={{ fontSize: '10px', color: '#fbbf24', textTransform: 'uppercase', fontWeight: 700, marginBottom: '4px', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <AlertTriangle size={12} /> Scientific Boundaries & Limitations
          </div>
          <ul style={{ margin: 0, paddingLeft: '16px', fontSize: '11px', color: '#cbd5e1', lineHeight: '1.5' }}>
            {analysis.limitations.map((lim, i) => (
              <li key={i}>{lim}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
