import React, { useEffect, useRef, useState } from 'react';
import { Camera, Image as ImageIcon, Play, Square, Upload, Video } from 'lucide-react';
import {
  HealthStatus,
  PredictionResponse,
  detectLiveFrameApi,
  detectLiveFrameClientPreview,
  predictClientPreview,
  predictDebrisApi,
  segmentImageApi,
  segmentImageClientFallback,
} from '../api/client';
import { CURATED_SAMPLES, MODELS_BENCHMARK, ModelId } from '../data/benchmarkData';

const DEMO_SAMPLES = CURATED_SAMPLES.filter((s) => s.is_debris_eval);
type InputMode = 'samples' | 'live';

export function InferenceTab({ health }: { health: HealthStatus | null }) {
  const [mode, setMode] = useState<InputMode>('samples');

  const [selectedId, setSelectedId] = useState<string>(DEMO_SAMPLES[0].id);
  const [origPreview, setOrigPreview] = useState<string>(DEMO_SAMPLES[0].orig_url);
  const [segPreview, setSegPreview] = useState<string>(DEMO_SAMPLES[0].seg_url);
  const [trueLabel, setTrueLabel] = useState<string | undefined>(DEMO_SAMPLES[0].true_class);
  const [isCustom, setIsCustom] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const loopTimerRef = useRef<number | null>(null);
  const busyRef = useRef(false);

  const [liveSource, setLiveSource] = useState<'none' | 'camera' | 'video'>('none');
  const [liveRunning, setLiveRunning] = useState(false);
  const [liveBbox, setLiveBbox] = useState<[number, number, number, number] | null>(null);
  const [liveDim, setLiveDim] = useState<[number, number]>([224, 224]);

  const [modelId, setModelId] = useState<ModelId>('a5_opt');
  const [isRunning, setIsRunning] = useState(false);
  const [isSegmenting, setIsSegmenting] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const [bbox, setBbox] = useState<[number, number, number, number]>([36, 36, 188, 188]);
  const [dragStart, setDragStart] = useState<[number, number] | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  const stopLiveMedia = () => {
    setLiveRunning(false);
    if (loopTimerRef.current !== null) {
      window.clearInterval(loopTimerRef.current);
      loopTimerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  useEffect(() => () => stopLiveMedia(), []);

  const switchMode = (next: InputMode) => {
    if (next === mode) return;
    setErrorMsg(null);
    if (next === 'samples') {
      stopLiveMedia();
      setLiveSource('none');
      setLiveBbox(null);
      selectSample(DEMO_SAMPLES[0].id);
    } else {
      setResult(null);
    }
    setMode(next);
  };

  const selectSample = (id: string) => {
    const s = DEMO_SAMPLES.find((x) => x.id === id);
    if (!s) return;
    setIsCustom(false);
    setSelectedId(s.id);
    setOrigPreview(s.orig_url);
    setSegPreview(s.seg_url);
    setTrueLabel(s.true_class);
    setErrorMsg(null);
  };

  useEffect(() => {
    if (mode === 'samples') void runStaticInference();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, modelId, mode]);

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>, target: 'orig' | 'seg') => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const url = String(reader.result);
      setIsCustom(true);
      setSelectedId('custom');
      setTrueLabel(undefined);
      setResult(null);
      if (target === 'orig') {
        setOrigPreview(url);
        setSegPreview(url);
      } else {
        setSegPreview(url);
      }
    };
    reader.readAsDataURL(file);
  };

  const startCamera = async () => {
    stopLiveMedia();
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      setLiveSource('camera');
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setLiveRunning(true);
    } catch (err) {
      setErrorMsg(err instanceof Error ? `Camera error: ${err.message}` : 'Unable to open camera');
    }
  };

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    stopLiveMedia();
    setErrorMsg(null);
    const url = URL.createObjectURL(file);
    setLiveSource('video');
    setTrueLabel(undefined);
    setTimeout(() => {
      if (videoRef.current) {
        videoRef.current.srcObject = null;
        videoRef.current.src = url;
        void videoRef.current.play();
        setLiveRunning(true);
      }
    }, 50);
  };

  useEffect(() => {
    if (!liveRunning || mode !== 'live') {
      if (loopTimerRef.current !== null) {
        window.clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
      return;
    }

    const tick = async () => {
      const v = videoRef.current;
      if (!v || v.readyState < 2 || v.paused || busyRef.current) return;
      busyRef.current = true;
      try {
        const vw = v.videoWidth || 224;
        const vh = v.videoHeight || 224;
        setLiveDim([vw, vh]);

        const liveTorch = Boolean(health?.torch_available && health?.weights_loaded?.[modelId]);
        if (liveTorch) {
          const cap = document.createElement('canvas');
          cap.width = vw;
          cap.height = vh;
          cap.getContext('2d')!.drawImage(v, 0, 0, vw, vh);
          const blob = await new Promise<Blob | null>((resolve) => cap.toBlob(resolve, 'image/jpeg', 0.85));
          if (blob) {
            const res = await detectLiveFrameApi(blob, modelId);
            setResult(res);
            setLiveBbox(res.bbox);
            setSegPreview(res.segmented_data_url);
          }
        } else {
          const res = await detectLiveFrameClientPreview(v, modelId);
          setResult(res);
          setLiveBbox(res.bbox);
          setSegPreview(res.segmented_data_url);
        }
      } catch (err) {
        setErrorMsg(err instanceof Error ? err.message : 'Live frame error');
      } finally {
        busyRef.current = false;
      }
    };

    void tick();
    loopTimerRef.current = window.setInterval(() => void tick(), 650);
    return () => {
      if (loopTimerRef.current !== null) {
        window.clearInterval(loopTimerRef.current);
        loopTimerRef.current = null;
      }
    };
  }, [liveRunning, mode, modelId, health]);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!isCustom || !boxRef.current) return;
    const rect = boxRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(224, ((e.clientX - rect.left) / rect.width) * 224));
    const y = Math.max(0, Math.min(224, ((e.clientY - rect.top) / rect.height) * 224));
    setDragStart([x, y]);
    setBbox([x, y, x + 4, y + 4]);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!dragStart || !boxRef.current) return;
    const rect = boxRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(224, ((e.clientX - rect.left) / rect.width) * 224));
    const y = Math.max(0, Math.min(224, ((e.clientY - rect.top) / rect.height) * 224));
    setBbox([
      Math.min(dragStart[0], x),
      Math.min(dragStart[1], y),
      Math.max(dragStart[0], x),
      Math.max(dragStart[1], y),
    ]);
  };

  const handleSegment = async () => {
    setIsSegmenting(true);
    setErrorMsg(null);
    try {
      if (health) {
        const blob = await (await fetch(origPreview)).blob();
        const segRes = await segmentImageApi(blob, bbox);
        setSegPreview(segRes.segmented_data_url);
      } else {
        const segRes = await segmentImageClientFallback(origPreview, bbox);
        setSegPreview(segRes.segmented_data_url);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Segmentation failed');
    } finally {
      setIsSegmenting(false);
    }
  };

  const runStaticInference = async () => {
    setIsRunning(true);
    setErrorMsg(null);
    try {
      const liveTorch = Boolean(health?.torch_available && health?.weights_loaded?.[modelId]);
      if (liveTorch) {
        const [origBlob, segBlob] = await Promise.all([
          (await fetch(origPreview)).blob(),
          (await fetch(segPreview)).blob(),
        ]);
        const res = await predictDebrisApi(origBlob, segBlob, modelId);
        setResult(res);
      } else {
        const res = await predictClientPreview(origPreview, segPreview, modelId, trueLabel);
        setResult(res);
      }
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Classification failed');
    } finally {
      setIsRunning(false);
    }
  };

  const liveWeights = Boolean(health?.torch_available && health?.weights_loaded?.[modelId]);

  return (
    <div className="stack">
      {/* Mode + Samples Bar */}
      <div className="card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem', marginBottom: mode === 'samples' ? '0.9rem' : 0 }}>
          <div className="segmented">
            <button className={mode === 'samples' ? 'active' : ''} onClick={() => switchMode('samples')}>
              <ImageIcon size={14} /> Image
            </button>
            <button className={mode === 'live' ? 'active' : ''} onClick={() => switchMode('live')}>
              <Video size={14} /> Live Camera / Video
            </button>
          </div>

          {mode === 'samples' ? (
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <label className="pill-btn pill-outline">
                <Upload size={14} /> Upload Image
                <input type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => handleUpload(e, 'orig')} />
              </label>
              {isCustom && (
                <label className="pill-btn pill-outline">
                  <Upload size={14} /> Upload Mask
                  <input type="file" accept="image/*" style={{ display: 'none' }} onChange={(e) => handleUpload(e, 'seg')} />
                </label>
              )}
            </div>
          ) : (
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              <button className="pill-btn pill-dark" onClick={startCamera}>
                <Camera size={14} /> Start Webcam
              </button>
              <label className="pill-btn pill-outline">
                <Upload size={14} /> Load Video
                <input type="file" accept="video/*" style={{ display: 'none' }} onChange={handleVideoUpload} />
              </label>
              {liveSource !== 'none' && (
                <button className="pill-btn pill-sage" onClick={() => setLiveRunning((v) => !v)}>
                  {liveRunning ? <><Square size={13} /> Pause</> : <><Play size={13} /> Resume</>}
                </button>
              )}
            </div>
          )}
        </div>

        {mode === 'samples' && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '0.65rem' }}>
            {DEMO_SAMPLES.map((s) => {
              const active = !isCustom && selectedId === s.id;
              return (
                <button
                  key={s.id}
                  onClick={() => selectSample(s.id)}
                  style={{
                    textAlign: 'left',
                    padding: '0.45rem',
                    borderRadius: '12px',
                    border: active ? '2px solid var(--charcoal)' : '1px solid var(--border)',
                    background: active ? 'var(--sage-tint)' : 'var(--surface)',
                    cursor: 'pointer',
                    fontFamily: 'var(--font-sans)',
                  }}
                >
                  <img
                    src={s.orig_url}
                    alt={s.title}
                    style={{ width: '100%', height: '74px', objectFit: 'cover', borderRadius: '8px', display: 'block', marginBottom: '0.35rem' }}
                  />
                  <div style={{ fontSize: '0.77rem', fontWeight: 600, color: 'var(--ink)' }}>{s.true_class}</div>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Main Workspace */}
      <div className="two-col">
        {mode === 'samples' ? (
          <div className="card">
            <h2 className="section-heading" style={{ marginBottom: '0.85rem' }}>Dual-Stream Input</h2>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem', marginBottom: '1rem' }}>
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--ink-muted)', marginBottom: '0.35rem' }}>
                  Original RGB
                </div>
                <div
                  ref={boxRef}
                  onPointerDown={onPointerDown}
                  onPointerMove={onPointerMove}
                  onPointerUp={() => setDragStart(null)}
                  style={{
                    position: 'relative',
                    aspectRatio: '1 / 1',
                    borderRadius: '12px',
                    overflow: 'hidden',
                    border: '1px solid var(--border)',
                    cursor: isCustom ? 'crosshair' : 'default',
                    background: '#000',
                  }}
                >
                  <img src={origPreview} alt="Original input" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', pointerEvents: 'none' }} />
                  {isCustom && (
                    <div
                      style={{
                        position: 'absolute',
                        left: `${(bbox[0] / 224) * 100}%`,
                        top: `${(bbox[1] / 224) * 100}%`,
                        width: `${((bbox[2] - bbox[0]) / 224) * 100}%`,
                        height: `${((bbox[3] - bbox[1]) / 224) * 100}%`,
                        border: '2px dashed #ffffff',
                        pointerEvents: 'none',
                      }}
                    />
                  )}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--ink-muted)', marginBottom: '0.35rem' }}>
                  SAM Mask
                </div>
                <div style={{ aspectRatio: '1 / 1', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', background: '#000' }}>
                  <img src={segPreview} alt="SAM segmented input" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
              {isCustom && (
                <button className="pill-btn pill-sage" onClick={handleSegment} disabled={isSegmenting}>
                  {isSegmenting ? 'Isolating...' : 'Isolate Mask'}
                </button>
              )}
              <button className="pill-btn pill-dark" onClick={runStaticInference} disabled={isRunning}>
                {isRunning ? 'Classifying...' : 'Classify Debris'}
              </button>
            </div>
          </div>
        ) : (
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
              <h2 className="section-heading" style={{ margin: 0 }}>Live Detection Feed</h2>
              {liveRunning && <span className="tag tag-green">Live</span>}
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1.35fr 0.65fr', gap: '0.85rem', alignItems: 'start' }}>
              <div
                style={{
                  position: 'relative',
                  aspectRatio: '4 / 3',
                  borderRadius: '12px',
                  overflow: 'hidden',
                  border: '1px solid var(--border)',
                  background: '#181b17',
                  display: 'grid',
                  placeItems: 'center',
                }}
              >
                <video
                  ref={videoRef}
                  loop
                  muted
                  playsInline
                  controls={liveSource === 'video'}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: liveSource === 'none' ? 'none' : 'block',
                  }}
                />
                {liveSource === 'none' && (
                  <div style={{ color: '#a3a89f', fontSize: '0.84rem', textAlign: 'center', padding: '1rem' }}>
                    Click <strong>Start Webcam</strong> or <strong>Load Video</strong> above.
                  </div>
                )}
                {liveSource !== 'none' && liveBbox && (
                  <div
                    style={{
                      position: 'absolute',
                      left: `${(liveBbox[0] / liveDim[0]) * 100}%`,
                      top: `${(liveBbox[1] / liveDim[1]) * 100}%`,
                      width: `${((liveBbox[2] - liveBbox[0]) / liveDim[0]) * 100}%`,
                      height: `${((liveBbox[3] - liveBbox[1]) / liveDim[1]) * 100}%`,
                      border: '2px solid #dce7d0',
                      borderRadius: '4px',
                      pointerEvents: 'none',
                    }}
                  >
                    {result && (
                      <div
                        className="mono"
                        style={{
                          position: 'absolute',
                          top: '-22px',
                          left: '-2px',
                          background: '#181b17',
                          color: '#ffffff',
                          fontSize: '0.7rem',
                          fontWeight: 600,
                          padding: '2px 6px',
                          borderRadius: '4px 4px 4px 0',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {result.predicted_class} {(result.confidence * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--ink-muted)', marginBottom: '0.35rem' }}>
                  Isolated Mask
                </div>
                <div style={{ aspectRatio: '1 / 1', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', background: '#000' }}>
                  {liveSource !== 'none' && (
                    <img src={segPreview} alt="Live segmented frame" style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Prediction Card */}
        <div className="card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.85rem' }}>
            <h2 className="section-heading" style={{ margin: 0 }}>Prediction</h2>
            <span className={`tag ${liveWeights ? 'tag-green' : 'tag-amber'}`}>
              {liveWeights ? 'PyTorch' : 'Preview'}
            </span>
          </div>

          <select
            value={modelId}
            onChange={(e) => setModelId(e.target.value as ModelId)}
            style={{
              width: '100%',
              padding: '0.52rem 0.7rem',
              borderRadius: '10px',
              border: '1px solid var(--border-strong)',
              background: 'var(--surface)',
              color: 'var(--ink)',
              fontFamily: 'var(--font-sans)',
              fontSize: '0.85rem',
              marginBottom: '1rem',
            }}
          >
            {MODELS_BENCHMARK.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))}
          </select>

          {errorMsg && <p style={{ color: 'var(--red)', fontSize: '0.83rem' }}>{errorMsg}</p>}

          {result ? (
            <>
              <div
                style={{
                  padding: '0.9rem 1rem',
                  borderRadius: '12px',
                  background: 'var(--sage-tint)',
                  border: '1px solid var(--border)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'baseline',
                  marginBottom: '1rem',
                }}
              >
                <div>
                  <div style={{ fontSize: '0.73rem', color: 'var(--ink-subtle)' }}>Predicted Class</div>
                  <div style={{ fontSize: '1.45rem', fontWeight: 700, color: 'var(--ink)' }}>
                    {result.predicted_class}
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div className="mono" style={{ fontSize: '1.15rem', fontWeight: 700 }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                  {trueLabel && (
                    <div style={{ fontSize: '0.74rem', color: result.predicted_class === trueLabel ? 'var(--green)' : 'var(--red)' }}>
                      True: {trueLabel}
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {result.probabilities.map((p) => {
                  const pct = p.probability * 100;
                  const isTop = p.class_name === result.predicted_class;
                  return (
                    <div key={p.class_name}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.79rem', marginBottom: '0.16rem' }}>
                        <span style={{ fontWeight: isTop ? 600 : 400 }}>{p.class_name}</span>
                        <span className="mono" style={{ color: 'var(--ink-muted)' }}>{pct.toFixed(1)}%</span>
                      </div>
                      <div style={{ height: '6px', borderRadius: '999px', background: 'var(--alabaster)', overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${pct}%`,
                            height: '100%',
                            background: isTop ? 'var(--charcoal)' : '#cfd4cb',
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <p style={{ fontSize: '0.84rem', color: 'var(--ink-subtle)' }}>
              Start a camera or load a video to classify frames live.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
