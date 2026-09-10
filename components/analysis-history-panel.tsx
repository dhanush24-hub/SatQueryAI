'use client';

import React, { useEffect, useState } from 'react';
import { History, Clock, ShieldCheck, ChevronRight, RefreshCw, X } from 'lucide-react';
import { listAnalyses, getAnalysis, type AnalysisHistorySummaryDTO, type AnalysisResponse } from '@/lib/api/analysis';

interface AnalysisHistoryPanelProps {
  onSelectAnalysis: (analysis: AnalysisResponse) => void;
  onClose?: () => void;
  activeAnalysisId?: string;
}

export function AnalysisHistoryPanel({
  onSelectAnalysis,
  onClose,
  activeAnalysisId,
}: AnalysisHistoryPanelProps) {
  const [history, setHistory] = useState<AnalysisHistorySummaryDTO[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>('');

  const fetchHistory = async () => {
    setLoading(true);
    setError('');
    try {
      const items = await listAnalyses();
      setHistory(items);
    } catch (e: any) {
      setError(e.message || 'Failed to load analysis history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleSelect = async (id: string) => {
    try {
      const anl = await getAnalysis(id);
      onSelectAnalysis(anl);
      if (onClose) onClose();
    } catch (e: any) {
      setError(e.message || 'Failed to retrieve persisted analysis');
    }
  };

  const getBadgeColor = (state: string) => {
    switch (state) {
      case 'SUPPORTED':
        return '#34d399';
      case 'SUPPORTED_WITH_WARNINGS':
        return '#fbbf24';
      case 'INSUFFICIENT_EVIDENCE':
        return '#fb7185';
      case 'OUT_OF_DOMAIN':
        return '#c084fc';
      default:
        return '#94a3b8';
    }
  };

  return (
    <div className="analysis-history-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%', padding: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '13px', fontWeight: 600, color: '#f8fafc' }}>
          <History size={16} /> Analysis History (Persistent)
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <button
            onClick={fetchHistory}
            disabled={loading}
            style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: '4px' }}
            title="Refresh History"
          >
            <RefreshCw size={13} className={loading ? 'spin' : ''} />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: '4px' }}
            >
              <X size={15} />
            </button>
          )}
        </div>
      </div>

      {loading && history.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '24px', color: '#64748b', fontSize: '12px' }}>
          Loading saved analyses...
        </div>
      ) : error ? (
        <div style={{ color: '#fb7185', fontSize: '12px', padding: '8px' }}>
          {error}
        </div>
      ) : history.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '24px', color: '#64748b', fontSize: '12px' }}>
          No persistent analyses stored yet. Completed missions are automatically persisted to SQLite.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>
          {history.map((item) => {
            const isActive = item.id === activeAnalysisId;
            const badgeColor = getBadgeColor(item.evidence_state);
            return (
              <div
                key={item.id}
                onClick={() => handleSelect(item.id)}
                style={{
                  padding: '10px',
                  borderRadius: '6px',
                  background: isActive ? 'rgba(196,248,106,0.08)' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${isActive ? '#c4f86a' : 'rgba(255,255,255,0.06)'}`,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '11px', fontFamily: 'monospace', color: '#94a3b8' }}>
                    {item.id}
                  </span>
                  <span
                    style={{
                      fontSize: '9px',
                      padding: '1px 5px',
                      borderRadius: '3px',
                      fontWeight: 700,
                      color: badgeColor,
                      border: `1px solid ${badgeColor}40`,
                      background: `${badgeColor}15`
                    }}
                  >
                    {item.evidence_state}
                  </span>
                </div>

                <div style={{ fontSize: '12px', fontWeight: 600, color: '#f1f5f9', marginTop: '4px', lineHeight: '1.4' }}>
                  {item.query}
                </div>

                <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {item.summary}
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '6px', fontSize: '10px', color: '#475569' }}>
                  <span>{item.task_family}</span>
                  <span>{item.created_at ? new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
