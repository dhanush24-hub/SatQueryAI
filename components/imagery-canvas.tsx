'use client';
import React, {useState, useRef, useEffect, useMemo} from 'react';
import {
  ZoomIn, ZoomOut, RotateCcw, Eye, Split, Columns2, Sparkles,
  Layers, Compass, Grid, Sliders, Activity, Info, Check, Play, Pause,
  ChevronRight, ArrowRightLeft, ShieldCheck, MapPin, Maximize2, Crosshair,
  Zap, Calendar, Satellite
} from 'lucide-react';
import type {Scene, Evidence} from '@/lib/analysis';
import type {IsroAnalysisData} from '@/lib/domain';

export type ViewMode = 'side-by-side' | 'swipe' | 'flicker' | 'diff' | 'single';
export type FilterLens = 'normal' | 'cir' | 'ndvi' | 'ndwi' | 'sar';

interface ImageryCanvasProps {
  scenes: Scene[];
  imageIndex: number;
  setImageIndex: (index: number) => void;
  evidenceList: Evidence[];
  selectedEvidenceIndex: number | null;
  onSelectEvidence: (index: number | null) => void;
  selectedFindingId: string | null;
  isroData?: IsroAnalysisData;
  busy?: boolean;
  changeMaskUrl?: string | null;
  onUploadClick?: () => void;
  onRunAnalysis?: () => void;
}

export function ImageryCanvas({
  scenes,
  imageIndex,
  setImageIndex,
  evidenceList,
  selectedEvidenceIndex,
  onSelectEvidence,
  selectedFindingId,
  isroData,
  busy,
  changeMaskUrl,
  onUploadClick,
  onRunAnalysis
}: ImageryCanvasProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');
  const [filterLens, setFilterLens] = useState<FilterLens>('normal');
  const [showMarkings, setShowMarkings] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showGraticule, setShowGraticule] = useState(true);
  const [showCrosshair, setShowCrosshair] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [maskOpacity, setMaskOpacity] = useState(0.70);
  const [zoom, setZoom] = useState(1);
  const [swipePos, setSwipePos] = useState(50); // percentage (0 to 100)
  const [isDraggingSwipe, setIsDraggingSwipe] = useState(false);
  const [flickerActive, setFlickerActive] = useState(true);
  const [flickerIndex, setFlickerIndex] = useState(0);
  const [crosshairPos, setCrosshairPos] = useState<{pctX: number; pctY: number; lat: string; lon: string} | null>(null);
  const [hoveredEvidenceId, setHoveredEvidenceId] = useState<string | null>(null);

  const containerRef = useRef<HTMLDivElement>(null);

  const activeScene = scenes[imageIndex] || scenes[0];
  const sceneA = scenes[0];
  const sceneB = scenes[1] || scenes[0];
  const isMultiScene = scenes.length > 1;

  // Auto ensure appropriate view mode when scenes change
  useEffect(() => {
    if (!isMultiScene) {
      setViewMode('single');
    } else if (viewMode === 'single' && scenes.length > 1) {
      setViewMode('side-by-side');
    }
  }, [isMultiScene, scenes.length]);

  // Handle flicker animation loop
  useEffect(() => {
    if (!flickerActive || !isMultiScene || viewMode !== 'flicker') return;
    const interval = setInterval(() => {
      setFlickerIndex(prev => (prev === 0 ? 1 : 0));
    }, 900);
    return () => clearInterval(interval);
  }, [flickerActive, isMultiScene, viewMode]);

  // Handle swipe dragging
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingSwipe || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const newPos = Math.max(0, Math.min(100, (x / rect.width) * 100));
      setSwipePos(newPos);
    };
    const handleMouseUp = () => setIsDraggingSwipe(false);

    if (isDraggingSwipe) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingSwipe]);

  // Calculate synchronized crosshair relative to the active target image
  const handleImageMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top));
    const pctX = (x / rect.width) * 100;
    const pctY = (y / rect.height) * 100;

    // Approximate geodetic coordinates for display HUD (Cairo ~30.04°N, 31.23°E)
    const baseLat = 30.0444 + (0.5 - pctY / 100) * 0.16;
    const baseLon = 31.2357 + (pctX / 100 - 0.5) * 0.20;

    const latDeg = Math.floor(Math.abs(baseLat));
    const latMin = Math.floor((Math.abs(baseLat) - latDeg) * 60);
    const latSec = (((Math.abs(baseLat) - latDeg) * 60 - latMin) * 60).toFixed(1);
    const latStr = `${latDeg}°${latMin}'${latSec}"${baseLat >= 0 ? 'N' : 'S'}`;

    const lonDeg = Math.floor(Math.abs(baseLon));
    const lonMin = Math.floor((Math.abs(baseLon) - lonDeg) * 60);
    const lonSec = (((Math.abs(baseLon) - lonDeg) * 60 - lonMin) * 60).toFixed(1);
    const lonStr = `${lonDeg}°${lonMin}'${lonSec}"${baseLon >= 0 ? 'E' : 'W'}`;

    setCrosshairPos({pctX, pctY, lat: latStr, lon: lonStr});
  };

  const getCategoryColor = (cat?: string) => {
    if (!cat) return '#c4f86a';
    const c = cat.toLowerCase();
    if (c.includes('construct') || c.includes('urban') || c.includes('building')) return '#f59e0b'; // Amber
    if (c.includes('water') || c.includes('flood') || c.includes('river')) return '#06b6d4'; // Cyan
    if (c.includes('vegetat') || c.includes('forest') || c.includes('crop')) return '#10b981'; // Emerald
    if (c.includes('sar') || c.includes('radar') || c.includes('proxy')) return '#a855f7'; // Purple
    if (c.includes('loss') || c.includes('conflict') || c.includes('risk')) return '#f43f5e'; // Red
    return '#c4f86a'; // Neon Lime
  };

  const filterStyle = useMemo(() => {
    switch (filterLens) {
      case 'cir':
        return {filter: 'contrast(140%) saturate(175%) hue-rotate(-55deg)'};
      case 'ndvi':
        return {filter: 'contrast(190%) brightness(95%) sepia(85%) hue-rotate(65deg) saturate(230%)'};
      case 'ndwi':
        return {filter: 'contrast(210%) brightness(85%) hue-rotate(175deg) saturate(260%)'};
      case 'sar':
        return {filter: 'grayscale(100%) contrast(220%) brightness(105%)'};
      default:
        return {};
    }
  }, [filterLens]);

  // Find counterpart evidence in other image for cross-highlighting
  const hoveredFindingId = useMemo(() => {
    if (!hoveredEvidenceId) return null;
    return evidenceList.find(e => e.id === hoveredEvidenceId)?.findingId || null;
  }, [hoveredEvidenceId, evidenceList]);

  // Render bounding box overlays for an image
  const renderVisualMarkings = (imgIdx: number) => {
    if (!showMarkings) return null;

    const activeList = evidenceList.filter(e => e.image === imgIdx);

    return activeList.map((ev, i) => {
      const isSelected = selectedEvidenceIndex !== null && evidenceList[selectedEvidenceIndex]?.id === ev.id;
      const isHovered = hoveredEvidenceId === ev.id || (hoveredFindingId && ev.findingId === hoveredFindingId);
      const isRelatedToSelectedFinding = selectedFindingId && ev.findingId === selectedFindingId;
      const color = getCategoryColor(ev.category);

      const left = ev.box[0];
      const top = ev.box[1];
      const width = ev.box[2] - ev.box[0];
      const height = ev.box[3] - ev.box[1];

      return (
        <div
          key={ev.id || i}
          className={`isro-roi-box ${isSelected || isRelatedToSelectedFinding ? 'selected' : ''} ${isHovered ? 'hovered' : ''}`}
          style={{
            left: `${left}%`,
            top: `${top}%`,
            width: `${width}%`,
            height: `${height}%`,
            borderColor: color,
            boxShadow: isSelected || isRelatedToSelectedFinding
              ? `0 0 0 2px ${color}, 0 0 18px ${color}aa`
              : isHovered
              ? `0 0 0 1.5px ${color}, 0 0 12px ${color}88`
              : `0 0 0 1px ${color}55`
          }}
          onClick={(e) => {
            e.stopPropagation();
            const realIdx = evidenceList.findIndex(x => x.id === ev.id);
            onSelectEvidence(realIdx >= 0 ? realIdx : null);
          }}
          onMouseEnter={() => setHoveredEvidenceId(ev.id || null)}
          onMouseLeave={() => setHoveredEvidenceId(null)}
        >
          {/* High-tech cartographic corner brackets */}
          <span className="roi-corner tl" style={{borderColor: color}}/>
          <span className="roi-corner tr" style={{borderColor: color}}/>
          <span className="roi-corner bl" style={{borderColor: color}}/>
          <span className="roi-corner br" style={{borderColor: color}}/>

          {/* Header pill badge */}
          <div className="roi-badge" style={{backgroundColor: color, color: '#091014'}}>
            <span className="roi-num">E{evidenceList.findIndex(x => x.id === ev.id) + 1}</span>
            <span className="roi-title">{ev.title}</span>
            {ev.areaHa && <span className="roi-ha">+{ev.areaHa} ha</span>}
          </div>

          {/* Area & confidence tags */}
          <div className="roi-footer" style={{backgroundColor: `${color}30`, borderColor: color}}>
            <span className="roi-coord">[{left.toFixed(0)}%, {top.toFixed(0)}%]</span>
            <span className="roi-conf">{ev.confidence || 'verified'}</span>
          </div>
        </div>
      );
    });
  };

  // Render synchronized crosshair reticle over image
  const renderCrosshair = () => {
    if (!showCrosshair || !crosshairPos) return null;
    return (
      <div className="sync-crosshair-reticle" style={{left: `${crosshairPos.pctX}%`, top: `${crosshairPos.pctY}%`}}>
        <div className="crosshair-h" />
        <div className="crosshair-v" />
        <div className="crosshair-circle" />
        <div className="crosshair-tag">
          {crosshairPos.pctX.toFixed(1)}%, {crosshairPos.pctY.toFixed(1)}%
        </div>
      </div>
    );
  };

  // Render change detection heatmap overlay
  const renderHeatmap = () => {
    if (!showHeatmap || !isroData?.changeDetectionMask) return null;
    return isroData.changeDetectionMask.map((mask, idx) => (
      <div
        key={idx}
        className="isro-heatmap-zone"
        style={{
          left: `${mask.x}%`,
          top: `${mask.y}%`,
          width: `${mask.width}%`,
          height: `${mask.height}%`,
          opacity: mask.intensity
        }}
      >
        <div className="heatmap-pulse" />
      </div>
    ));
  };

  if (!activeScene) {
    return (
      <div className="canvas-empty-wrap">
        <button className="canvas-empty drop-canvas" onClick={onUploadClick} disabled={busy}>
          <div className="drop-orbit">
            <Layers size={32} />
          </div>
          <span className="eyebrow">ISRO GEOSPATIAL OBSERVATION CANVAS</span>
          <h3>Drop satellite imagery to begin.</h3>
          <p>Multi-temporal T₁ ↔ T₂ pairs or Optical + SAR acquisitions.<br/>Automated georeferencing, change detection, and evidence grounding.</p>
          <span className="drop-browse">Upload Imagery <ChevronRight size={16}/></span>
        </button>
      </div>
    );
  }

  return (
    <div className="isro-canvas-container" ref={containerRef}>
      {/* Canvas Master Toolbar */}
      <div className="canvas-master-toolbar">
        {/* Left: View Mode Switches */}
        <div className="canvas-toolbar-group">
          {isMultiScene ? (
            <>
              <button
                className={`toolbar-mode-btn ${viewMode === 'side-by-side' ? 'active' : ''}`}
                onClick={() => setViewMode('side-by-side')}
                title="Synchronized Side-by-Side Dual View"
              >
                <Columns2 size={14} /> Side-by-Side
              </button>
              <button
                className={`toolbar-mode-btn ${viewMode === 'swipe' ? 'active' : ''}`}
                onClick={() => setViewMode('swipe')}
                title="Interactive Swipe Curtain Comparison Slider"
              >
                <Split size={14} /> Curtain Swipe
              </button>
              <button
                className={`toolbar-mode-btn ${viewMode === 'flicker' ? 'active' : ''}`}
                onClick={() => {
                  setViewMode('flicker');
                  setFlickerActive(true);
                }}
                title="1 Hz Rapid Flicker Comparator"
              >
                <ArrowRightLeft size={14} /> Flicker
              </button>
              <button
                className={`toolbar-mode-btn ${viewMode === 'diff' ? 'active' : ''}`}
                onClick={() => setViewMode('diff')}
                title="Pixel Difference Heat-Map Overlay"
              >
                <Sparkles size={14} /> Δ Difference
              </button>
              <button
                className={`toolbar-mode-btn ${viewMode === 'single' ? 'active' : ''}`}
                onClick={() => setViewMode('single')}
                title="Single Scene Focus"
              >
                <Eye size={14} /> Single Focus
              </button>
            </>
          ) : (
            <div className="single-scene-tag">
              <Eye size={14} /> SINGLE OBSERVATION VIEWPORT
            </div>
          )}
        </div>

        {/* Center: Multispectral Lens Filter */}
        <div className="canvas-toolbar-group center-group">
          <span className="group-label"><Activity size={13}/> LENS:</span>
          <button
            className={`lens-chip ${filterLens === 'normal' ? 'active' : ''}`}
            onClick={() => setFilterLens('normal')}
          >
            RGB True Color
          </button>
          <button
            className={`lens-chip ${filterLens === 'cir' ? 'active' : ''}`}
            onClick={() => setFilterLens('cir')}
            title="False Color Infrared (RGB Display Simulation)"
          >
            CIR False-Color
          </button>
          <button
            className={`lens-chip ${filterLens === 'ndvi' ? 'active' : ''}`}
            onClick={() => setFilterLens('ndvi')}
            title="NDVI Vegetation Enhancement (RGB Display Simulation)"
          >
            NDVI (Sim)
          </button>
          <button
            className={`lens-chip ${filterLens === 'ndwi' ? 'active' : ''}`}
            onClick={() => setFilterLens('ndwi')}
            title="NDWI Water Feature Enhancement (RGB Display Simulation)"
          >
            NDWI (Sim)
          </button>
          <button
            className={`lens-chip ${filterLens === 'sar' ? 'active' : ''}`}
            onClick={() => setFilterLens('sar')}
            title="Radar Backscatter Texture Simulation (RGB Display Simulation)"
          >
            SAR Texture (Sim)
          </button>
        </div>

        {/* Right: Overlays & Zoom Controls */}
        <div className="canvas-toolbar-group right-group">
          {changeMaskUrl && (
            <div className="mask-control-group" style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: '4px', border: '1px solid rgba(196,248,106,0.3)' }}>
              <button
                className={`toggle-icon-btn ${showMask ? 'active' : ''}`}
                onClick={() => setShowMask(!showMask)}
                title="Toggle Predicted Change Mask"
                style={{ background: 'transparent', border: 'none', color: showMask ? '#c4f86a' : '#94a3b8', display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer', fontSize: '11px', fontWeight: 600 }}
              >
                <Layers size={13} /> Change Mask
              </button>
              {showMask && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '10px', color: '#94a3b8' }}>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={maskOpacity}
                    onChange={e => setMaskOpacity(parseFloat(e.target.value))}
                    style={{ width: '60px', height: '4px', cursor: 'pointer', accentColor: '#c4f86a' }}
                    title={`Mask Opacity: ${Math.round(maskOpacity * 100)}%`}
                  />
                  <span>{Math.round(maskOpacity * 100)}%</span>
                </div>
              )}
            </div>
          )}
          <button
            className={`toggle-icon-btn ${showMarkings ? 'active' : ''}`}
            onClick={() => setShowMarkings(!showMarkings)}
            title="Toggle Visual Markings / ROIs"
          >
            <ShieldCheck size={14} /> ROIs ({evidenceList.length})
          </button>
          <button
            className={`toggle-icon-btn ${showCrosshair ? 'active' : ''}`}
            onClick={() => setShowCrosshair(!showCrosshair)}
            title="Toggle Synchronized Dual Crosshair"
          >
            <Crosshair size={14} /> Crosshair
          </button>
          <button
            className={`toggle-icon-btn ${showGraticule ? 'active' : ''}`}
            onClick={() => setShowGraticule(!showGraticule)}
            title="Toggle Cartographic Graticule"
          >
            <Grid size={14} /> Graticule
          </button>

          <div className="zoom-controls">
            <button onClick={() => setZoom(z => Math.max(1, z - 0.25))} disabled={zoom <= 1} title="Zoom Out">
              <ZoomOut size={14} />
            </button>
            <span className="zoom-val">{Math.round(zoom * 100)}%</span>
            <button onClick={() => setZoom(z => Math.min(3, z + 0.25))} disabled={zoom >= 3} title="Zoom In">
              <ZoomIn size={14} />
            </button>
            <button onClick={() => setZoom(1)} title="Reset 100%">
              <RotateCcw size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* Main Imagery Display Viewport Area */}
      <div className="isro-viewport-area">
        {/* Unanalyzed Banner Prompt if images are loaded but evidence is not yet run */}
        {evidenceList.length === 0 && (
          <div className="canvas-unanalyzed-banner">
            <div className="banner-text">
              <Zap size={16} className="zap-icon"/>
              <span>
                <strong>{scenes.length} ACQUISITIONS READY FOR COMPARISON:</strong> Run remote sensing analysis to generate visual markings, LULC area deltas & ISRO telemetry.
              </span>
            </div>
            {onRunAnalysis && (
              <button className="run-analysis-btn" onClick={onRunAnalysis} disabled={busy}>
                <Zap size={14} /> {busy ? 'Analyzing...' : 'Execute Analysis Plan'}
              </button>
            )}
          </div>
        )}

        {/* 1. SYNCHRONIZED SIDE-BY-SIDE DUAL VIEW */}
        {viewMode === 'side-by-side' && isMultiScene && (
          <div className="side-by-side-layout">
            {/* Left Pane (Image 1 - T1 Baseline) */}
            <div className="side-pane pane-left">
              <div className="pane-header-tag">
                <span className="acq-tag">T₁ BASELINE</span>
                <strong>{sceneA.name}</strong>
                <span className="sensor-tag">{sceneA.label || 'Optical'}</span>
              </div>
              <div className="pane-scroll-container">
                <div
                  className="pane-image-wrap"
                  style={{transform: `scale(${zoom})`, transformOrigin: 'center center', ...filterStyle}}
                  onMouseMove={handleImageMouseMove}
                  onMouseLeave={() => setCrosshairPos(null)}
                >
                  <img src={`data:${sceneA.mime};base64,${sceneA.data}`} alt={sceneA.name} />
                  {renderVisualMarkings(0)}
                  {renderHeatmap()}
                  {renderCrosshair()}
                </div>
              </div>
            </div>

            {/* Central Divider with VS Tag */}
            <div className="side-by-side-divider">
              <span className="divider-label">VS</span>
            </div>

            {/* Right Pane (Image 2 - T2 Target) */}
            <div className="side-pane pane-right">
              <div className="pane-header-tag">
                <span className="acq-tag">T₂ TARGET</span>
                <strong>{sceneB.name}</strong>
                <span className="sensor-tag">{sceneB.label || 'Target'}</span>
              </div>
              <div className="pane-scroll-container">
                <div
                  className="pane-image-wrap"
                  style={{transform: `scale(${zoom})`, transformOrigin: 'center center', ...filterStyle}}
                  onMouseMove={handleImageMouseMove}
                  onMouseLeave={() => setCrosshairPos(null)}
                >
                  <img src={`data:${sceneB.mime};base64,${sceneB.data}`} alt={sceneB.name} />
                  {changeMaskUrl && showMask && (
                    <img
                      src={changeMaskUrl}
                      alt="Predicted Change Mask"
                      className="predicted-change-mask-layer"
                      style={{
                        position: 'absolute',
                        inset: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        opacity: maskOpacity,
                        pointerEvents: 'none',
                        zIndex: 6
                      }}
                    />
                  )}
                  {renderVisualMarkings(1)}
                  {renderHeatmap()}
                  {renderCrosshair()}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 2. INTERACTIVE CURTAIN SWIPE SLIDER */}
        {viewMode === 'swipe' && isMultiScene && (
          <div className="swipe-slider-layout" style={{userSelect: isDraggingSwipe ? 'none' : 'auto'}}>
            <div
              className="swipe-image-container"
              style={{transform: `scale(${zoom})`, transformOrigin: 'center center', ...filterStyle}}
              onMouseMove={handleImageMouseMove}
              onMouseLeave={() => setCrosshairPos(null)}
            >
              {/* Underneath image (T2 / Target) */}
              <div className="swipe-img-layer target-layer">
                <img src={`data:${sceneB.mime};base64,${sceneB.data}`} alt={sceneB.name} />
                {changeMaskUrl && showMask && (
                  <img
                    src={changeMaskUrl}
                    alt="Predicted Change Mask"
                    className="predicted-change-mask-layer"
                    style={{
                      position: 'absolute',
                      inset: 0,
                      width: '100%',
                      height: '100%',
                      objectFit: 'contain',
                      opacity: maskOpacity,
                      pointerEvents: 'none',
                      zIndex: 6
                    }}
                  />
                )}
                {renderVisualMarkings(1)}
                {renderHeatmap()}
                <div className="swipe-watermark right">T₂: {sceneB.name}</div>
              </div>

              {/* Clipped top image (T1 / Baseline) */}
              <div
                className="swipe-img-layer baseline-layer"
                style={{clipPath: `polygon(0 0, ${swipePos}% 0, ${swipePos}% 100%, 0 100%)`}}
              >
                <img src={`data:${sceneA.mime};base64,${sceneA.data}`} alt={sceneA.name} />
                {renderVisualMarkings(0)}
                <div className="swipe-watermark left">T₁: {sceneA.name}</div>
              </div>

              {/* Synchronized Crosshair */}
              {renderCrosshair()}

              {/* Draggable Divider Handle */}
              <div
                className="swipe-divider-handle"
                style={{left: `${swipePos}%`}}
                onMouseDown={() => setIsDraggingSwipe(true)}
              >
                <div className="handle-bar" />
                <div className="handle-knob">
                  <ArrowRightLeft size={13} />
                </div>
              </div>
            </div>

            {/* Swipe Presets Controls */}
            <div className="swipe-presets-bar">
              <span className="preset-label">SWIPE:</span>
              <button onClick={() => setSwipePos(25)}>25%</button>
              <button onClick={() => setSwipePos(50)}>50% (Split)</button>
              <button onClick={() => setSwipePos(75)}>75%</button>
            </div>
          </div>
        )}

        {/* 3. 1 Hz FLICKER COMPARATOR */}
        {viewMode === 'flicker' && isMultiScene && (
          <div className="flicker-layout">
            {(() => {
              const currentScene = flickerIndex === 0 ? sceneA : sceneB;
              return (
                <div
                  className="flicker-image-wrap"
                  style={{transform: `scale(${zoom})`, transformOrigin: 'center center', ...filterStyle}}
                  onMouseMove={handleImageMouseMove}
                  onMouseLeave={() => setCrosshairPos(null)}
                >
                  <img src={`data:${currentScene.mime};base64,${currentScene.data}`} alt={currentScene.name} />
                  {renderVisualMarkings(flickerIndex)}
                  {renderHeatmap()}
                  {renderCrosshair()}
                </div>
              );
            })()}

            <div className="flicker-hud-tag">
              <span className="flicker-dot" />
              <strong>
                FLICKER: {flickerIndex === 0 ? `T₁ [${sceneA.name}]` : `T₂ [${sceneB.name}]`}
              </strong>
              <button
                className="flicker-toggle-btn"
                onClick={() => setFlickerActive(!flickerActive)}
              >
                {flickerActive ? <Pause size={12}/> : <Play size={12}/>}
                {flickerActive ? 'Pause' : 'Resume'}
              </button>
            </div>
          </div>
        )}

        {/* 4. DIFFERENCE HEATMAP LENS VIEW */}
        {viewMode === 'diff' && isMultiScene && (
          <div className="diff-layout">
            <div
              className="diff-image-wrap"
              style={{transform: `scale(${zoom})`, transformOrigin: 'center center'}}
              onMouseMove={handleImageMouseMove}
              onMouseLeave={() => setCrosshairPos(null)}
            >
              {/* Baseline under layer */}
              <img src={`data:${sceneA.mime};base64,${sceneA.data}`} alt={sceneA.name} className="diff-base" />
              {/* Target difference blending */}
              <img
                src={`data:${sceneB.mime};base64,${sceneB.data}`}
                alt={sceneB.name}
                className="diff-overlay"
                style={{mixBlendMode: 'difference', filter: 'contrast(300%) invert(100%) saturate(200%)'}}
              />
              {renderVisualMarkings(1)}
              {renderCrosshair()}
            </div>
            <div className="diff-hud-tag">
              <Sparkles size={14} className="diff-icon" />
              <span>TEMPORAL PIXEL DIFFERENCE MATRIX: High-intensity highlights represent structural & biomass displacement</span>
            </div>
          </div>
        )}

        {/* 5. SINGLE SCENE FOCUS VIEW */}
        {viewMode === 'single' && (
          <div className="single-scene-layout">
            {isMultiScene && (
              <div className="single-switcher-bar">
                <button
                  className={`scene-select-chip ${imageIndex === 0 ? 'active' : ''}`}
                  onClick={() => setImageIndex(0)}
                >
                  <Calendar size={12}/> T₁: {sceneA.name}
                </button>
                <button
                  className={`scene-select-chip ${imageIndex === 1 ? 'active' : ''}`}
                  onClick={() => setImageIndex(1)}
                >
                  <Calendar size={12}/> T₂: {sceneB.name}
                </button>
              </div>
            )}
            <div
              className="single-image-wrap"
              style={{transform: `scale(${zoom})`, transformOrigin: 'center center', ...filterStyle}}
              onMouseMove={handleImageMouseMove}
              onMouseLeave={() => setCrosshairPos(null)}
            >
              <img src={`data:${activeScene.mime};base64,${activeScene.data}`} alt={activeScene.name} />
              {renderVisualMarkings(imageIndex)}
              {renderHeatmap()}
              {renderCrosshair()}
            </div>
          </div>
        )}

        {/* Cartographic Coordinate Graticule Overlay */}
        {showGraticule && (
          <div className="cartographic-graticule">
            <div className="graticule-grid-lines" />
            <div className="north-arrow-badge">
              <Compass size={22} className="north-compass-icon" />
              <span>N</span>
            </div>
            <div className="scale-bar-container">
              <div className="scale-bar-ruler" />
              <span>{Math.round(500 / zoom)} m</span>
            </div>
          </div>
        )}
      </div>

      {/* Real-time Telemetry Coordinates HUD (Docked at canvas bottom) */}
      <div className="canvas-bottom-telemetry">
        <div className="telemetry-item">
          <span className="dot active" />
          <span className="label">GEO-DATUM:</span>
          <b>{isroData?.telemetry?.[0]?.crs || 'EPSG:32636 (WGS 84 / UTM 36N)'}</b>
        </div>
        {crosshairPos ? (
          <>
            <div className="telemetry-item highlight">
              <span className="label">LAT:</span>
              <b>{crosshairPos.lat}</b>
            </div>
            <div className="telemetry-item highlight">
              <span className="label">LON:</span>
              <b>{crosshairPos.lon}</b>
            </div>
            <div className="telemetry-item">
              <span className="label">PIXEL:</span>
              <b>[{crosshairPos.pctX.toFixed(1)}%, {crosshairPos.pctY.toFixed(1)}%]</b>
            </div>
          </>
        ) : (
          <div className="telemetry-item">
            <span className="label">CURSOR TRACKING:</span>
            <span className="cursor-prompt">Hover canvas to track coordinates</span>
          </div>
        )}
        <div className="telemetry-item">
          <span className="label">GSD:</span>
          <b>{isroData?.telemetry?.[0]?.resolutionGsd || '30 m'}</b>
        </div>
        <div className="telemetry-item">
          <span className="label">RMS ERROR:</span>
          <b>{isroData?.telemetry?.[0]?.coregistrationRms || '0.18 px'}</b>
        </div>
        <div className="telemetry-item right-align">
          <span className="label">SENSORS:</span>
          <b>{scenes.map(s => s.name).join(' ↔ ')}</b>
        </div>
      </div>
    </div>
  );
}
