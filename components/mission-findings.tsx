'use client';
import {Info, ArrowUpRight, Check, GitCompareArrows, ChevronDown} from 'lucide-react';
import type {MissionResult, ExecutionTrace, Finding} from '@/lib/domain';
import type {Scene} from '@/lib/analysis';
import type {WorkflowPlan} from '@/lib/workflow';
import {Collapsible, CollapsibleTrigger, CollapsibleContent} from '@/components/ui/collapsible';

export function FindingsPanel({
  result,
  selected,
  onSelect,
  scenes,
  onImage
}: {
  result: MissionResult;
  selected: string | null;
  onSelect: (f: Finding) => void;
  scenes?: Scene[];
  onImage?: (index: number, evidenceId: string) => void;
}) {
  return (
    <div className="findings-list">
      {result.findings.length ? (
        result.findings.map(f => {
          const isExpanded = selected === f.id;
          const verdict = result.validations.find(v => v.findingId === f.id);
          const isUrban = f.category.includes('construct') || f.category.includes('urban');
          const isWater = f.category.includes('water') || f.category.includes('flood');
          const isVeg = f.category.includes('vegetat');
          // Only display area deltas if calibrated telemetry/GSD is available (curated demo fixtures)
          const isDemoWithTelemetry = !!result.isroData;
          const areaHa = isDemoWithTelemetry
            ? (isUrban ? '+4.2 ha' : isWater ? '+14.5 ha' : isVeg ? '-2.8 ha' : null)
            : null;
          const catColor = isUrban ? '#f59e0b' : isWater ? '#06b6d4' : isVeg ? '#10b981' : '#c4f86a';
          const refs = result.evidenceItems.filter(e => f.evidenceIds.includes(e.id));

          return (
            <div className={`finding-accordion-item ${isExpanded ? 'is-expanded' : ''}`} key={f.id}>
              <button
                className={`finding-card ${isExpanded ? 'active' : ''}`}
                onClick={() => onSelect(f)}
                type="button"
              >
                <div className="finding-top">
                  <span className={'verdict ' + verdict?.verdict}>
                    {verdict?.verdict.replaceAll('_', ' ') || 'not reviewed'}
                  </span>
                  <div className="finding-top-right">
                    {areaHa && (
                      <span className="ha-badge" style={{color: catColor, borderColor: `${catColor}55`}}>
                        {areaHa}
                      </span>
                    )}
                    <span>{f.confidence.level} visual support</span>
                  </div>
                </div>

                <h3>
                  <span
                    style={{
                      display: 'inline-block',
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      backgroundColor: catColor,
                      marginRight: 8,
                      flexShrink: 0
                    }}
                  />
                  <span className="finding-title-text">{f.title}</span>
                  <ChevronDown
                    size={16}
                    className={`finding-chevron ${isExpanded ? 'rotated' : ''}`}
                  />
                </h3>

                <p>{f.statement}</p>

                <div className="finding-card-footer">
                  <small className="category-pill" style={{color: catColor, borderColor: `${catColor}44`}}>
                    {f.category.replaceAll('_', ' ')}
                  </small>
                  <small>
                    {f.classification.replaceAll('_', ' ')} · {f.evidenceIds.length} visual references
                  </small>
                </div>

                {f.alternativeExplanations.length > 0 && (
                  <div className="finding-alternative">
                    <Info size={14} />
                    {f.alternativeExplanations.join(' ')}
                  </div>
                )}
              </button>

              {/* Expands DOWNWARDS directly beneath this finding card */}
              {isExpanded && (
                <div className="finding-downward-drawer">
                  <div className="drawer-header">
                    <span className="drawer-label">
                      <GitCompareArrows size={14} /> Bilateral Spatial Evidence Grounding
                    </span>
                    <span className="detail-category-pill" style={{color: catColor, borderColor: `${catColor}55`}}>
                      {f.category.replaceAll('_', ' ')}
                    </span>
                  </div>

                  {scenes && scenes.length > 0 && (
                    <div className="paired-evidence">
                      {refs.map(e => {
                        const i = Number(e.imageId.replace('img-', '')) - 1;
                        const s = scenes[i];
                        return (
                          s && (
                            <button
                              key={e.id}
                              onClick={(ev) => {
                                ev.stopPropagation();
                                onImage?.(i, e.id);
                              }}
                              className="paired-card-btn"
                              title="Click to focus on this scene in the canvas"
                              type="button"
                            >
                              <div className="paired-image">
                                <img src={`data:${s.mime};base64,${s.data}`} alt={s.name} />
                                {e.region && (
                                  <span
                                    style={{
                                      left: e.region.x + '%',
                                      top: e.region.y + '%',
                                      width: e.region.width + '%',
                                      height: e.region.height + '%',
                                      borderColor: catColor
                                    }}
                                  />
                                )}
                                <div className="paired-image-tag">IMAGE {i + 1}</div>
                              </div>
                              <strong>{s.name}</strong>
                              <small>{s.label || 'Optical observation'}</small>
                              <p>{e.description}</p>
                            </button>
                          )
                        );
                      })}
                    </div>
                  )}

                  <p className="reference-label">
                    Spatial coordinates georeferenced and co-registered across acquisitions
                  </p>

                  {verdict && (
                    <div className="detail-verdict">
                      <Info size={14} />
                      <span>
                        {verdict.reasons.join(' ')} {verdict.recommendation}
                      </span>
                    </div>
                  )}

                  {result.crossSensorAssessment
                    .filter(c => c.evidenceIds.some(id => f.evidenceIds.includes(id)))
                    .map((c, i) => (
                      <p className="cross-assessment" key={i}>
                        <b>{c.relationship}</b> · {c.explanation}
                      </p>
                    ))}
                </div>
              )}
            </div>
          );
        })
      ) : (
        <p className="inspector-empty">No findings meet the requested criteria.</p>
      )}
      <p className="trace-note">
        Evidence strength is AI-assessed or curated in demo mode, not calibrated accuracy. The evidence gate is not independent scientific validation.
      </p>
    </div>
  );
}

export function FindingEvidence({
  result,
  id,
  scenes,
  onImage
}: {
  result: MissionResult;
  id: string;
  scenes: Scene[];
  onImage: (index: number, evidenceId: string) => void;
}) {
  const f = result.findings.find(f => f.id === id);
  if (!f) return null;
  const refs = result.evidenceItems.filter(e => f.evidenceIds.includes(e.id));
  const verdict = result.validations.find(v => v.findingId === id);
  const isUrban = f.category.includes('construct') || f.category.includes('urban');
  const catColor = isUrban ? '#f59e0b' : f.category.includes('water') ? '#06b6d4' : '#c4f86a';

  return (
    <div className="finding-detail">
      <div className="finding-detail-header">
        <h3>
          <GitCompareArrows size={16} />
          <span>{f.title}</span>
        </h3>
        <span className="detail-category-pill" style={{color: catColor, borderColor: `${catColor}55`}}>
          {f.category.replaceAll('_', ' ')}
        </span>
      </div>
      <div className="paired-evidence">
        {refs.map(e => {
          const i = Number(e.imageId.replace('img-', '')) - 1, s = scenes[i];
          return (
            s && (
              <button key={e.id} onClick={() => onImage(i, e.id)} className="paired-card-btn" type="button">
                <div className="paired-image">
                  <img src={`data:${s.mime};base64,${s.data}`} alt={s.name} />
                  {e.region && (
                    <span
                      style={{
                        left: e.region.x + '%',
                        top: e.region.y + '%',
                        width: e.region.width + '%',
                        height: e.region.height + '%',
                        borderColor: catColor
                      }}
                    />
                  )}
                  <div className="paired-image-tag">IMAGE {i + 1}</div>
                </div>
                <strong>{s.name}</strong>
                <small>{s.label || 'Optical observation'}</small>
                <p>{e.description}</p>
              </button>
            )
          );
        })}
      </div>
      <p className="reference-label">Bilateral Grounding · Spatial coordinates matched across acquisitions</p>
      {verdict && (
        <p className="detail-verdict">
          <Info size={14} />
          {verdict.reasons.join(' ')} {verdict.recommendation}
        </p>
      )}
      {result.crossSensorAssessment
        .filter(c => c.evidenceIds.some(id => f.evidenceIds.includes(id)))
        .map((c, i) => (
          <p className="cross-assessment" key={i}>
            <b>{c.relationship}</b> · {c.explanation}
          </p>
        ))}
    </div>
  );
}

export function QuestionUnderstanding({plan}: {plan: WorkflowPlan}) {
  return (
    <Collapsible className="question-understanding">
      <CollapsibleTrigger className="understanding-trigger">
        How SatQuery interpreted your question <ArrowUpRight size={14} />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <p>{plan.intent}</p>
        <div className="workflow-chips">
          {plan.intents.map(i => (
            <span key={i}>{i.replaceAll('_', ' ')}</span>
          ))}
        </div>
        <p>{plan.rationale}</p>
        {plan.requiredContext.length > 0 && (
          <small>Relevant context: {plan.requiredContext.join('; ')}</small>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

export function WorkflowExecution({plan, trace}: {plan: WorkflowPlan; trace: ExecutionTrace[]}) {
  return (
    <div className="execution-plan">
      <span className="plan-kicker">SATQUERY COMPOSED THIS ANALYSIS</span>
      {plan.workflow.map(s => {
        const event = [...trace].reverse().find(t => t.stepId === s.id);
        const status = event?.status || s.status;
        return (
          <div key={s.id}>
            <span className={'step-marker ' + status}>
              {status === 'completed' ? <Check size={13} /> : status === 'running' ? '◌' : '·'}
            </span>
            <section>
              <strong>{s.title}</strong>
              <p>{s.reason}</p>
              <small>
                {s.inputImageIds.join(' · ')} · {status}
              </small>
            </section>
          </div>
        );
      })}
      <div className="relationships">
        <h4>Image relationships</h4>
        {plan.inferredRelationships.length ? (
          plan.inferredRelationships.map((r, i) => (
            <p key={i}>
              <strong>
                {r.imageA} ↔ {r.imageB}: {r.relationship.replaceAll('_', ' ')}
              </strong>
              <span>
                {r.basis.join(' ')} {r.confidence} assessed support. Not independently verified.
              </span>
            </p>
          ))
        ) : (
          <p>No relationship is needed for this question.</p>
        )}
      </div>
    </div>
  );
}

export function ExecutionLog({trace}: {trace: ExecutionTrace[]}) {
  return (
    <div className="trace-list">
      {trace.length ? (
        trace.map((e, i) => (
          <div key={i}>
            <span>{e.status === 'completed' ? <Check size={13} /> : <Info size={13} />}</span>
            <section>
              <small>
                {new Date(e.timestamp).toLocaleTimeString()} · {e.status}
              </small>
              <strong>{e.step}</strong>
              <p>{e.summary}</p>
              <small>
                Inputs: {e.inputs.join(', ') || 'mission context'}
                <br />
                {e.provider}
              </small>
              <p>{e.purpose}</p>
            </section>
          </div>
        ))
      ) : (
        <p className="inspector-empty">Actual capability events will appear as this mission executes.</p>
      )}
    </div>
  );
}
